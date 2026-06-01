"""
johnson_noise_simulation.py
===========================
Full end-to-end simulation of the Johnson noise experiment.
Experiment: vary R at fixed T, extract kB from linear fit of <V^2> vs R.

Key insight: op-amp choice matters strongly for your R range.
  - LT1028: ultra-low voltage noise (0.85 nV/√Hz) but high current noise (1 pA/√Hz)
            -> current noise i_n*R dominates for R > ~1 kΩ. BAD for this experiment.
  - LT1012: higher voltage noise (15 nV/√Hz) but extremely low current noise (25 fA/√Hz)
            -> flat noise floor across 10 kΩ – 1 MΩ. CORRECT choice for stage 1.
  - OP07:   moderate both. Good for stage 2 (signal already amplified, its noise irrelevant).

Noise sources simulated:
  - Johnson noise of the resistor (signal)
  - Amplifier voltage noise (stage 1 + stage 2)
  - Amplifier current noise (stage 1, flows through R)
  - MyDAQ quantisation noise
  - 50 Hz mains interference + harmonics
  - 1/f (flicker) noise of the resistor

Analysis pipeline:
  - Welch PSD estimation
  - Mains notch filtering in frequency domain
  - Amplifier noise floor subtraction (short-circuit measurement)
  - Divide by G^2 to refer back to resistor input
  - Linear fit of <V^2> vs R -> extract kB
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Physical
kB_true   = 1.380649e-23   # true Boltzmann constant [J/K]
T         = 293.0          # temperature [K]

# Resistors to sweep
R_values  = [10e3, 47e3, 100e3, 270e3, 470e3, 1e6]  # Ohms

# Amplifier: two stages
G1, G2    = 100.0, 10.0    # gains of stage 1 and stage 2
G_total   = G1 * G2        # = 1000

# Stage 1 op-amp: LT1012 — best choice for R in 10kΩ–1MΩ range
# (low current noise 25 fA/√Hz means i_n*R stays small across all R values)
en_stage1 = 15.0e-9        # LT1012 voltage noise density [V/√Hz]
in_stage1 = 25.0e-15       # LT1012 current noise density [A/√Hz]

# Stage 2 op-amp: OP07 — its noise is divided by G1 when referred to input, negligible
en_stage2 = 10.0e-9        # OP07 voltage noise density [V/√Hz]

# MyDAQ
fs        = 50000          # sample rate [Hz]
V_range   = 2.0            # MyDAQ ±V_range input range [V]
bits      = 16             # ADC resolution
t_measure = 60.0           # measurement duration per resistor [seconds]

# Analysis band — chosen to avoid 50 Hz hum and stay in flat amplifier region
f_low     = 1000.0         # [Hz]
f_high    = 5000.0         # [Hz]
nperseg   = 8192           # Welch segment length

# Mains interference — simulated at the resistor input level (very small)
mains_freq      = 50.0
mains_harmonics = 5
mains_amplitude = 0.5e-9   # [V] input-referred — represents well-shielded setup

# Resistor 1/f (flicker) noise corner frequency
flicker_corner  = 200.0    # [Hz] — below f_low, so minimal impact in band

# ─────────────────────────────────────────────────────────────────────────────
# NOISE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_time_series(R, T, fs, duration, G_total, G1,
                     en1, in1, en2,
                     mains_amp, mains_freq, n_harmonics,
                     flicker_corner, V_range, bits, rng):
    """
    Generate simulated MyDAQ time series for a resistor of value R.
    All noise sources generated independently in time domain.
    Returns: array of length N = int(fs * duration) [Volts at MyDAQ input]
    """
    N  = int(fs * duration)
    dt = 1.0 / fs
    t  = np.arange(N) * dt

    # ── Johnson noise ──────────────────────────────────────────────────────
    # PSD = 4*kB*T*R [V^2/Hz], so sample std = sqrt(PSD * fs/2)
    johnson = rng.normal(0, np.sqrt(4 * kB_true * T * R * fs / 2), N)

    # ── 1/f flicker noise (generated in frequency domain) ─────────────────
    freqs        = np.fft.rfftfreq(N, d=dt)
    freqs[0]     = 1e-10
    flicker_psd  = (4 * kB_true * T * R) * (flicker_corner / freqs)
    flicker_amp  = np.sqrt(flicker_psd * fs / 2 / N)
    flicker_fd   = flicker_amp * np.exp(1j * rng.uniform(0, 2*np.pi, len(freqs)))
    flicker_fd[0] = 0
    flicker      = np.fft.irfft(flicker_fd, n=N)

    # ── Stage 1 voltage noise (referred to input) ──────────────────────────
    amp_v1 = rng.normal(0, np.sqrt(en1**2 * fs / 2), N)

    # ── Stage 1 current noise (flows through R, becomes voltage) ──────────
    # Noise density = i_n * R [V/√Hz]
    amp_i1 = rng.normal(0, np.sqrt((in1 * R)**2 * fs / 2), N)

    # ── Stage 2 voltage noise (referred back to system input via /G1) ──────
    amp_v2 = rng.normal(0, np.sqrt((en2 / G1)**2 * fs / 2), N)

    # ── Total input signal ─────────────────────────────────────────────────
    v_input     = johnson + flicker + amp_v1 + amp_i1 + amp_v2
    v_amplified = v_input * G_total

    # ── Mains interference (added at MyDAQ level) ──────────────────────────
    for h in range(1, n_harmonics + 1):
        v_amplified += (mains_amp * G_total
                        * np.sin(2*np.pi * h * mains_freq * t
                                 + rng.uniform(0, 2*np.pi)))

    # ── MyDAQ quantisation ─────────────────────────────────────────────────
    LSB         = (2 * V_range) / (2**bits)
    v_amplified = np.round(v_amplified / LSB) * LSB

    return v_amplified


def measure_psd(signal, fs, nperseg, f_low, f_high,
                notch_bw=5.0, mains_freq=50.0):
    """
    Estimate PSD with Welch, integrate over [f_low, f_high]
    with mains harmonics notched out.
    Returns: f, Pxx, V2_integrated
    """
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg,
                   window='hann', scaling='density')

    mask = (f >= f_low) & (f <= f_high)

    # Notch mains harmonics
    h = 1
    while h * mains_freq <= f_high * 1.1:
        mask &= np.abs(f - h * mains_freq) > notch_bw
        h += 1

    V2 = np.trapezoid(Pxx[mask], f[mask])
    return f, Pxx, V2, mask


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(seed=42)

print("=" * 65)
print("  JOHNSON NOISE EXPERIMENT — FULL SIMULATION")
print(f"  T = {T} K ({T-273:.0f}°C)  |  G = {G_total:.0f}  |  "
      f"Band = {f_low/1e3:.0f}–{f_high/1e3:.0f} kHz")
print(f"  Stage 1: LT1012  en={en_stage1*1e9:.0f} nV/√Hz  "
      f"in={in_stage1*1e15:.0f} fA/√Hz")
print(f"  Stage 2: OP07    en={en_stage2*1e9:.0f} nV/√Hz")
print(f"  Duration: {t_measure:.0f} s per resistor")
print("=" * 65)

# ── Step 1: Short-circuit (R≈0) — measure amplifier noise floor ──────────
print("\n[1/3] Short-circuit measurement (amp noise floor)...")
v_short = make_time_series(
    R=0.01, T=T, fs=fs, duration=t_measure, G_total=G_total, G1=G1,
    en1=en_stage1, in1=in_stage1, en2=en_stage2,
    mains_amp=mains_amplitude, mains_freq=mains_freq, n_harmonics=mains_harmonics,
    flicker_corner=flicker_corner, V_range=V_range, bits=bits, rng=rng
)
f_sc, Pxx_sc, V2_floor, _ = measure_psd(v_short, fs, nperseg, f_low, f_high)
# Refer floor PSD back to input
Pxx_sc_input = Pxx_sc / G_total**2
floor_density = np.mean(Pxx_sc_input[(f_sc >= f_low) & (f_sc <= f_high)])
print(f"   Floor PSD (input-referred): {np.sqrt(floor_density)*1e9:.2f} nV/√Hz")
print(f"   Floor <V²> over band:       {V2_floor:.3e} V² (at MyDAQ)")

# ── Step 2: Measure each resistor ────────────────────────────────────────
print(f"\n[2/3] Measuring Johnson noise...")
print(f"{'R':>10}  {'V²_raw':>12}  {'V²_corrected':>14}  "
      f"{'V_rms_meas':>12}  {'V_rms_theory':>13}  {'error':>7}")

V2_corrected = []
psd_data     = {}

for R in R_values:
    v_sig = make_time_series(
        R=R, T=T, fs=fs, duration=t_measure, G_total=G_total, G1=G1,
        en1=en_stage1, in1=in_stage1, en2=en_stage2,
        mains_amp=mains_amplitude, mains_freq=mains_freq, n_harmonics=mains_harmonics,
        flicker_corner=flicker_corner, V_range=V_range, bits=bits, rng=rng
    )
    f_r, Pxx_r, V2_raw, mask = measure_psd(v_sig, fs, nperseg, f_low, f_high)
    psd_data[R] = (f_r, Pxx_r, mask)

    # Subtract amp floor, refer to input
    V2_corr = (V2_raw - V2_floor) / G_total**2
    V2_corrected.append(V2_corr)

    # Theory: only Johnson noise (no amp noise)
    df_eff       = f_high - f_low
    V2_theory    = 4 * kB_true * T * R * df_eff
    V_rms_theory = np.sqrt(V2_theory)
    V_rms_meas   = np.sqrt(abs(V2_corr))
    err_pct      = (V_rms_meas - V_rms_theory) / V_rms_theory * 100

    label = f"{R/1e3:.0f} kΩ" if R < 1e6 else "1 MΩ"
    print(f"  R={label:>6}:  {V2_raw:.3e} V²  {V2_corr:.3e} V²  "
          f"{V_rms_meas*1e6:.3f} μV  {V_rms_theory*1e6:.3f} μV  {err_pct:+.1f}%")

# ── Step 3: Linear fit ────────────────────────────────────────────────────
print(f"\n[3/3] Linear fit: <V²> = 4·kB·T·Δf · R")

R_arr  = np.array(R_values)
V2_arr = np.array(V2_corrected)
df_eff = f_high - f_low

slope, intercept, r_value, _, std_err = linregress(R_arr, V2_arr)
kB_meas = slope / (4 * T * df_eff)
kB_err  = std_err / (4 * T * df_eff)
err_pct = (kB_meas - kB_true) / kB_true * 100

print(f"\n   Slope         = {slope:.4e} V²/Ω")
print(f"   Intercept     = {intercept:.4e} V²  (should be ≈ 0)")
print(f"   R²            = {r_value**2:.6f}")
print(f"\n   kB (measured) = {kB_meas:.4e} ± {kB_err:.2e} J/K")
print(f"   kB (true)     = {kB_true:.4e} J/K")
print(f"   Error         = {err_pct:+.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(R_values)))

# ── Plot 1: PSD for all resistors ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
for (R, (f_r, Pxx_r, mask)), col in zip(psd_data.items(), colors):
    Pxx_in = Pxx_r / G_total**2
    label  = f"{R/1e3:.0f} kΩ" if R < 1e6 else "1 MΩ"
    ax1.semilogy(f_r / 1e3, Pxx_in * 1e18, color=col, lw=1.1, label=f"R = {label}")

Pxx_floor_in = Pxx_sc / G_total**2
ax1.semilogy(f_sc / 1e3, Pxx_floor_in * 1e18, 'k--', lw=1.5, label='Amp floor (R≈0)')

# Shade integration band
ax1.axvspan(f_low/1e3, f_high/1e3, alpha=0.08, color='dodgerblue', label='Integration band')

# Theory lines (flat PSD = 4*kB*T*R)
for R, col in zip(R_values, colors):
    theory_psd = 4 * kB_true * T * R
    ax1.axhline(theory_psd * 1e18, color=col, lw=0.7, ls=':', alpha=0.6)

ax1.set_xlabel("Frequency (kHz)", fontsize=11)
ax1.set_ylabel("Input-referred PSD (aV²/Hz)", fontsize=11)
ax1.set_title("Welch PSD — input referred  (dotted lines = theory)", fontsize=11)
ax1.legend(fontsize=8, ncol=2, loc='upper right')
ax1.set_xlim(0, 10)
ax1.grid(True, which='both', alpha=0.25)

# ── Plot 2: <V²> vs R with linear fit ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
R_fit  = np.linspace(0, max(R_values) * 1.1, 300)
V2_fit = slope * R_fit + intercept

ax2.scatter(R_arr / 1e3, V2_arr * 1e12, color='royalblue',
            zorder=5, s=70, label='Simulation (measured)')

# Theory points
V2_theory_pts = np.array([4 * kB_true * T * R * df_eff for R in R_values])
ax2.scatter(R_arr / 1e3, V2_theory_pts * 1e12, marker='x',
            color='coral', s=80, zorder=6, lw=2, label='Theory (no amp noise)')

ax2.plot(R_fit / 1e3, V2_fit * 1e12, 'r-', lw=1.8,
         label=f'Fit  ($R^2$ = {r_value**2:.4f})')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel("R (kΩ)", fontsize=11)
ax2.set_ylabel(r"$\langle V^2 \rangle$ (pV²)", fontsize=11)
ax2.set_title(r"Johnson Noise: $\langle V^2 \rangle$ vs R", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Plot 3: kB result ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
labels = ['Measured\n$k_B$', 'True\n$k_B$']
vals   = [kB_meas * 1e23, kB_true * 1e23]
bars   = ax3.bar(labels, vals, color=['royalblue', 'coral'],
                 edgecolor='k', width=0.4)
ax3.errorbar(0, kB_meas * 1e23, yerr=kB_err * 1e23,
             fmt='none', color='k', capsize=8, lw=2)
ax3.set_ylabel("$k_B$ (× 10⁻²³ J/K)", fontsize=11)
ax3.set_title(f"Extracted $k_B$   (error = {err_pct:+.2f}%)", fontsize=11)
ax3.set_ylim(0.8 * kB_true * 1e23, 1.2 * kB_true * 1e23)
ax3.axhline(kB_true * 1e23, color='coral', lw=1, ls='--', alpha=0.5)
ax3.grid(True, axis='y', alpha=0.3)

fig.suptitle(
    f"Johnson Noise Simulation  |  T = {T} K  |  G = {G_total:.0f}  |  "
    f"Band = {f_low/1e3:.0f}–{f_high/1e3:.0f} kHz  |  "
    f"{t_measure:.0f} s/resistor  |  Stage 1: LT1012",
    fontsize=11, fontweight='bold'
)

plt.savefig("/tmp/johnson_simulation.png", dpi=150, bbox_inches='tight')
plt.savefig("/tmp/johnson_simulation.pdf", bbox_inches='tight')
print("\nPlots saved.")
