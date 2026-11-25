"""rcfilter: analysis library for the cascaded RC filter project (Physics 4BL).

Sections
    1. Reading recordings (daq.py files and SerialPlot CSV recordings)
    2. Filter models and the parameter formulas from the proposal
    3. Monte Carlo predictions from the multimeter readings
    4. Least-squares fitting helpers
    5. Sinusoid and phasor tools
    6. ESP32 calibration (sampling rate, channel delay, ADC linearity)
    7. Frequency sweeps (power ratio, fits, model-free checks)
    8. Phase points
    9. Square-wave bursts
   10. Comparison table with z-scores

Conventions
    Frequencies f are in Hz and omega = 2 pi f. Time constants are in seconds.
    Band-pass:  H = j K w / (1 - c w^2 + j b w)
    Low-pass 2: H = 1 / (1 - c w^2 + j b w)
    Power ratio P/P0 = |H|^2.
    "Sweep view" parameters include the generator resistance Rg, because the sweep is divided by the
    open-circuit reference. "Terminal view" parameters set Rg = 0, because the two-channel method
    measures the voltage at the circuit's input terminal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal

TWO_PI = 2.0 * np.pi


# =============================================================================================
# 1. Reading recordings
# =============================================================================================
def read_meta(path, max_lines=400):
    """Collect 'key=value' tokens from lines that start with '#'."""
    meta = {}
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta.setdefault(k, v)
    return meta


def load_recording(path, columns=None):
    """Load a recording as a float array of shape (n_samples, n_channels).

    Works for daq.py files and SerialPlot recordings: comment lines ('#'), header lines and broken
    rows are dropped. `columns` selects columns by index, for example columns=[1, 2] when a
    SerialPlot timestamp column comes first.
    """
    df = pd.read_csv(path, comment="#", header=None, dtype=str, skip_blank_lines=True,
                     on_bad_lines="skip", engine="python" if str(path).endswith(".txt") else "c")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    data = df.to_numpy(dtype=float)
    if columns is not None:
        data = data[:, columns]
    return data


def freq_from_name(path):
    """Read a frequency such as '700Hz' from a file name."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", str(path))
    return float(m.group(1)) if m else None


# =============================================================================================
# 2. Models and parameter formulas
# =============================================================================================
def omega(f):
    return TWO_PI * np.asarray(f, dtype=float)


def H_lp1(f, tau):
    return 1.0 / (1.0 + 1j * omega(f) * tau)


def H_lp2(f, b, c):
    w = omega(f)
    return 1.0 / (1.0 - c * w**2 + 1j * b * w)


def H_bp(f, K, b, c):
    w = omega(f)
    return 1j * K * w / (1.0 - c * w**2 + 1j * b * w)


def P_lp1(f, tau):
    return 1.0 / (1.0 + (omega(f) * tau) ** 2)


def P_lp2(f, b, c):
    w = omega(f)
    return 1.0 / ((1.0 - c * w**2) ** 2 + (b * w) ** 2)


def P_bp(f, K, b, c):
    w = omega(f)
    return (K * w) ** 2 / ((1.0 - c * w**2) ** 2 + (b * w) ** 2)


def phase_lp1_deg(f, tau):
    return -np.degrees(np.arctan(omega(f) * tau))


def phase_bp_deg(f, b, c):
    w = omega(f)
    return 90.0 - np.degrees(np.angle(1.0 - c * w**2 + 1j * b * w))


def tau_lp1(R1, C1, Rg=0.0):
    return (R1 + Rg) * C1


def bc_lp2(R1, C1, R2, C2, Rg=0.0, loaded=True):
    """Two-stage low-pass: loaded b = Ra C1 + R2 C2 + Ra C2, ideal drops Ra C2 (Ra = R1 + Rg)."""
    Ra = R1 + Rg
    b = Ra * C1 + R2 * C2 + (Ra * C2 if loaded else 0.0)
    c = Ra * C1 * R2 * C2
    return b, c


def Kbc_bp(R3, R4, R5, C3, C4, Rg=0.0, loaded=True):
    """Band-pass K, b, c. With Rg = 0 these are the proposal's formulas.

    Including the generator resistance in series with C3 keeps the response second order:
        loaded:   b = (RB+Rg) C3 + R3 C4 + RB C4,   c = (RB+Rg) C3 R3 C4 + Rg RB C3 C4
        buffered: b = (RB+Rg) C3 + R3 C4,           c = (RB+Rg) C3 R3 C4
    K = RB C3 in both. The extra Rg RB C3 C4 term is the 5 % change in c quoted in the hypothesis.
    """
    RB = R4 * R5 / (R4 + R5)
    K = RB * C3
    if loaded:
        b = (RB + Rg) * C3 + R3 * C4 + RB * C4
        c = (RB + Rg) * C3 * R3 * C4 + Rg * RB * C3 * C4
    else:
        b = (RB + Rg) * C3 + R3 * C4
        c = (RB + Rg) * C3 * R3 * C4
    return K, b, c


def lp2_half_power(b, c):
    """Frequency where (1 - c w^2)^2 + (b w)^2 = 2."""
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    B = b**2 - 2.0 * c
    u = (-B + np.sqrt(B**2 + 4.0 * c**2)) / (2.0 * c**2)
    return np.sqrt(u) / TWO_PI


def bp_edges(b, c):
    """Half-power edges of the band-pass, which are also its +45 and -45 degree points."""
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    root = np.sqrt(b**2 + 4.0 * c)
    return (root - b) / (2.0 * c) / TWO_PI, (root + b) / (2.0 * c) / TWO_PI


def bp_derived(K, b, c):
    K, b, c = (np.asarray(v, float) for v in (K, b, c))
    f_lo, f_hi = bp_edges(b, c)
    peak = K / b
    df = b / (TWO_PI * c)
    return {
        "f0": 1.0 / (TWO_PI * np.sqrt(c)),
        "df": df,
        "Q": np.sqrt(c) / b,
        "peak_amp": peak,
        "peak_power": peak**2,
        "product": peak * df,
        "f_lo": f_lo,
        "f_hi": f_hi,
    }


# =============================================================================================
# 3. Monte Carlo predictions
# =============================================================================================
def part(value, pct=0.0, counts=0, resolution=0.0):
    """A multimeter reading with stated accuracy of +/-(pct % of reading + counts x resolution)."""
    return {"value": float(value), "bound": float(value) * pct / 100.0 + counts * resolution}


def draw_parts(parts, n=10000, seed=1, bound_is="rectangular"):
    """Draw component values.

    bound_is = "rectangular": the stated accuracy is a bound and values are uniform inside it
                              (standard uncertainty bound / sqrt(3), the GUM convention).
    bound_is = "one_sigma":   the stated accuracy is treated as one standard deviation.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in parts.items():
        if bound_is == "rectangular":
            out[name] = p["value"] + rng.uniform(-p["bound"], p["bound"], n)
        elif bound_is == "one_sigma":
            out[name] = p["value"] + rng.normal(0.0, p["bound"], n)
        else:
            raise ValueError("bound_is must be 'rectangular' or 'one_sigma'")
    return out


def predict(p, f_check=20.0, square_f0=500.0, harmonics=(1, 3, 5, 7), phase_f=2000.0):
    """Every prediction in the hypothesis, from component values p (floats or Monte Carlo arrays).

    Keys use prefixes lp1 (single stage), lp2 (two stage), bpL (band-pass loaded), bpB (buffered).
    """
    R1, R2, R3, R4, R5 = (p[k] for k in ("R1", "R2", "R3", "R4", "R5"))
    C1, C2, C3, C4, Rg = (p[k] for k in ("C1", "C2", "C3", "C4", "Rg"))
    out = {}

    out["lp1.tau_sweep"] = tau_lp1(R1, C1, Rg)
    out["lp1.tau_phase"] = tau_lp1(R1, C1, 0.0)
    out["lp1.fc_sweep"] = 1.0 / (TWO_PI * out["lp1.tau_sweep"])
    out["lp1.fc_phase"] = 1.0 / (TWO_PI * out["lp1.tau_phase"])

    bL, cL = bc_lp2(R1, C1, R2, C2, Rg, loaded=True)
    bI, cI = bc_lp2(R1, C1, R2, C2, Rg, loaded=False)
    out["lp2.b_loaded"], out["lp2.b_ideal"], out["lp2.c"] = bL, bI, cL
    out["lp2.fhalf_loaded"] = lp2_half_power(bL, cL)
    out["lp2.fhalf_ideal"] = lp2_half_power(bI, cI)

    sweep = {}
    for tag, loaded in (("bpL", True), ("bpB", False)):
        K, b, c = Kbc_bp(R3, R4, R5, C3, C4, Rg, loaded)
        sweep[tag] = (K, b, c)
        out[f"{tag}.K"], out[f"{tag}.b"], out[f"{tag}.c"] = K, b, c
        for k, v in bp_derived(K, b, c).items():
            out[f"{tag}.{k}"] = v
        Kt, bt, ct = Kbc_bp(R3, R4, R5, C3, C4, 0.0, loaded)  # terminal view
        out[f"{tag}.Kt"], out[f"{tag}.bt"], out[f"{tag}.ct"] = Kt, bt, ct
        lo, hi = bp_edges(bt, ct)
        out[f"{tag}.ph_f0"] = 1.0 / (TWO_PI * np.sqrt(ct))
        out[f"{tag}.ph_p45"], out[f"{tag}.ph_m45"] = lo, hi
        out[f"{tag}.ph_{int(phase_f)}Hz"] = phase_bp_deg(phase_f, bt, ct)
        Hk = [H_bp(k * square_f0, Kt, bt, ct) for k in harmonics]
        ph = [np.degrees(np.angle(h)) for h in Hk]
        for k, h, a in zip(harmonics, Hk, ph):
            fk = int(k * square_f0)
            out[f"{tag}.H{fk}"] = np.abs(h)
            out[f"{tag}.phase{fk}"] = a
            if k != harmonics[0]:
                out[f"{tag}.norm{fk}"] = np.abs(h) / np.abs(Hk[0])
        out[f"{tag}.spread_deg"] = np.max(np.stack(ph), axis=0) - np.min(np.stack(ph), axis=0)

    out["bp.c_change_pct"] = 100.0 * (sweep["bpL"][2] / sweep["bpB"][2] - 1.0)
    out[f"bp.ratio_{int(f_check)}Hz"] = P_bp(f_check, *sweep["bpL"]) / P_bp(f_check, *sweep["bpB"])
    return out


def summarize(samples):
    """Mean and standard deviation of every Monte Carlo prediction."""
    rows = [(k, float(np.mean(v)), float(np.std(v, ddof=1)) if np.ndim(v) else 0.0)
            for k, v in samples.items()]
    return pd.DataFrame(rows, columns=["quantity", "mean", "std"]).set_index("quantity")


# =============================================================================================
# 4. Fitting helpers
# =============================================================================================
@dataclass
class Fit:
    names: list
    p: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    extra: dict = field(default_factory=dict)

    @property
    def perr(self):
        return np.sqrt(np.diag(self.cov))

    @property
    def chi2_red(self):
        return self.chi2 / self.dof if self.dof > 0 else np.nan

    @property
    def perr_scaled(self):
        """Errors inflated by sqrt(chi2_red) when chi2_red > 1."""
        return self.perr * np.sqrt(max(1.0, self.chi2_red))

    def __getitem__(self, name):
        i = self.names.index(name)
        return self.p[i], self.perr[i]

    def table(self):
        return pd.DataFrame({"value": self.p, "error": self.perr, "error_scaled": self.perr_scaled},
                            index=self.names)

    def __repr__(self):
        body = ", ".join(f"{n}={v:.5g}±{e:.2g}" for n, v, e in zip(self.names, self.p, self.perr))
        return f"Fit({body}; chi2_red={self.chi2_red:.3g}, dof={self.dof})"


def fit_scaled(fn, x, y, sigma, p0, scales, names, bounds=None):
    """curve_fit in scaled units (p / scales) for good conditioning; results in SI units."""
    scales = np.asarray(scales, float)
    p0s = np.asarray(p0, float) / scales

    def g(xx, *ps):
        return fn(xx, *(np.asarray(ps) * scales))

    if bounds is None:
        bnds = (-np.inf, np.inf)
    else:
        bnds = (np.asarray(bounds[0], float) / scales, np.asarray(bounds[1], float) / scales)
    ps, covs = optimize.curve_fit(g, x, y, p0=p0s, sigma=sigma, absolute_sigma=True, bounds=bnds,
                                  maxfev=20000)
    p = ps * scales
    cov = covs * np.outer(scales, scales)
    r = (y - fn(x, *p)) / sigma
    return Fit(list(names), p, cov, float(np.sum(r**2)), len(x) - len(p))


def chi2_at(fn, x, y, sigma, params):
    """Chi-square of a model at fixed (predicted) parameters, with dof = number of points."""
    r = (y - fn(x, *params)) / sigma
    chi2 = float(np.sum(r**2))
    return {"chi2": chi2, "dof": len(x), "chi2_red": chi2 / len(x)}


def mvn_draws(fit, n=20000, seed=2):
    rng = np.random.default_rng(seed)
    d = rng.multivariate_normal(fit.p, fit.cov, size=n)
    return d[np.all(d > 0, axis=1)]


# =============================================================================================
# 5. Sinusoid and phasor tools
# =============================================================================================
def fft_peak_freq(x, fs=1.0, fmin=0.0):
    """Peak frequency of a Hann-windowed FFT with parabolic interpolation."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    X[freqs < max(fmin, freqs[1])] = 0.0
    k = int(np.argmax(X))
    if 1 <= k < len(X) - 1:
        a, b, c = np.log(X[k - 1] + 1e-30), np.log(X[k] + 1e-30), np.log(X[k + 1] + 1e-30)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * fs / n


def phasors(x, t, freqs):
    """Linear least squares x(t) = dc + sum_k Re[Z_k exp(j w_k t)].

    Returns dc, the complex amplitudes Z, their amplitude and phase standard errors, and the
    residual RMS. Over whole periods this equals the DFT at exactly the harmonic frequencies.
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    freqs = np.atleast_1d(np.asarray(freqs, float))
    cols = [np.ones_like(t)]
    for f in freqs:
        w = TWO_PI * f
        cols += [np.cos(w * t), np.sin(w * t)]
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    resid = x - M @ coef
    dof = max(1, len(x) - M.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(M.T @ M)
    a, b = coef[1::2], coef[2::2]
    Z = a - 1j * b
    amp = np.abs(Z)
    s_amp = np.empty_like(amp)
    s_ph = np.empty_like(amp)
    for i in range(len(freqs)):
        C = cov[1 + 2 * i:3 + 2 * i, 1 + 2 * i:3 + 2 * i]
        ga = np.array([a[i], b[i]]) / amp[i]
        gp = np.array([b[i], -a[i]]) / amp[i] ** 2
        s_amp[i] = np.sqrt(ga @ C @ ga)
        s_ph[i] = np.sqrt(gp @ C @ gp)
    return {"dc": coef[0], "Z": Z, "s_amp": s_amp, "s_phase": s_ph,
            "resid_rms": float(np.sqrt(np.mean(resid**2))), "model": M @ coef}


def fit_frequency(x, t, f_guess):
    """Nonlinear fit of x = d + a cos(w t) + b sin(w t) with the frequency free."""
    x = np.asarray(x, float)
    tc = np.asarray(t, float) - np.mean(t)
    ph = phasors(x, tc, [f_guess])
    Z = ph["Z"][0]
    p0 = [0.0, Z.real, -Z.imag, ph["dc"]]

    w0 = TWO_PI * f_guess

    def model(tt, ppm, a, b, d):  # frequency as a deviation from f_guess in parts per million
        w = w0 * (1.0 + 1e-6 * ppm)
        return d + a * np.cos(w * tt) + b * np.sin(w * tt)

    def jac(tt, ppm, a, b, d):
        w = w0 * (1.0 + 1e-6 * ppm)
        c, s = np.cos(w * tt), np.sin(w * tt)
        return np.column_stack([(b * c - a * s) * tt * w0 * 1e-6, c, s, np.ones_like(tt)])

    p, cov = optimize.curve_fit(model, tc, x, p0=p0, jac=jac, maxfev=20000)
    return float(f_guess * (1.0 + 1e-6 * p[0])), float(f_guess * 1e-6 * np.sqrt(cov[0, 0]))


# =============================================================================================
# 6. Calibration
# =============================================================================================
def sampling_rate(x, f_true):
    """Sampling rate from a recording of a sine of known frequency f_true (Hz)."""
    x = np.asarray(x, float)
    n = np.arange(len(x), dtype=float)
    f0 = fft_peak_freq(x, 1.0)
    fc, sfc = fit_frequency(x, n, f0)  # cycles per sample
    fs = f_true / fc
    return fs, fs * sfc / fc


def channel_delay(records, fs):
    """Time offset between the paired samples of the two channels.

    records: list of (f_nominal, xA, xB) with both pins on the same generator signal.
    Fits dphi = 2 pi f dt through the origin. Also returns the gain ratio |B|/|A|.
    """
    rows = []
    for f_nom, xA, xB in records:
        t = np.arange(len(xA)) / fs
        f, _ = fit_frequency(xA, t, f_nom)
        pa, pb = phasors(xA, t, [f]), phasors(xB, t, [f])
        r = pb["Z"][0] / pa["Z"][0]
        dphi = float(np.angle(r))
        s = float(np.hypot(pa["s_phase"][0], pb["s_phase"][0]))
        rows.append((f, dphi, s, float(np.abs(r))))
    df = pd.DataFrame(rows, columns=["f", "dphi_rad", "s_dphi_rad", "gain_B_over_A"])
    w = 1.0 / df.s_dphi_rad**2
    x = TWO_PI * df.f
    dt = float(np.sum(w * x * df.dphi_rad) / np.sum(w * x**2))
    sdt = float(1.0 / np.sqrt(np.sum(w * x**2)))
    chi2 = float(np.sum(w * (df.dphi_rad - x * dt) ** 2))
    chi2_red = chi2 / max(1, len(df) - 1)
    df["resid_rad"] = df.dphi_rad - x * dt
    return {"dt": dt, "sdt": sdt, "sdt_scaled": sdt * np.sqrt(max(1.0, chi2_red)), "chi2_red": chi2_red,
            "gain_ratio": float(df.gain_B_over_A.mean()),
            "s_gain_ratio": float(df.gain_B_over_A.std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else 0.0,
            "table": df}


def _tri(theta):
    return (2.0 / np.pi) * np.arcsin(np.clip(np.sin(theta), -1.0, 1.0))


def triangle_linearity(x, fs, v_low=0.9, v_high=2.4, f_guess=None, corner_frac=0.05):
    """Fit an ideal triangle wave and report the ADC's deviation from linearity.

    Points within corner_frac of a period from each corner are excluded from the residuals.
    Returns the counts-per-volt calibration and the residuals in mV.
    """
    x = np.asarray(x, float)
    t = np.arange(len(x)) / fs
    f0 = f_guess or fft_peak_freq(x, fs)
    ph = phasors(x, t, [f0])
    phi0 = float(np.angle(ph["Z"][0])) + np.pi / 2
    A0 = 0.5 * (np.percentile(x, 99.5) - np.percentile(x, 0.5))

    def model(tt, A, f, phi, d):
        return d + A * _tri(TWO_PI * f * tt + phi)

    p, _ = optimize.curve_fit(model, t, x, p0=[A0, f0, phi0, np.mean(x)], maxfev=20000)
    A, f, phi, d = p
    if A < 0:
        A, phi = -A, phi + np.pi
    theta = np.mod(TWO_PI * f * t + phi, TWO_PI)
    near = np.minimum(np.abs(theta - np.pi / 2), np.abs(theta - 3 * np.pi / 2))
    mask = near > TWO_PI * corner_frac
    resid = x - model(t, A, f, phi, d)
    counts_per_volt = 2 * A / (v_high - v_low)
    resid_mV = resid / counts_per_volt * 1e3
    volts = v_low + (x - (d - A)) / counts_per_volt
    return {"counts_per_volt": counts_per_volt, "counts_at_0V": d - A - v_low * counts_per_volt,
            "f": f, "resid_rms_mV": float(np.sqrt(np.mean(resid_mV[mask] ** 2))),
            "resid_max_mV": float(np.max(np.abs(resid_mV[mask]))),
            "max_dev_pct_of_span": float(100 * np.max(np.abs(resid[mask])) / (2 * A)),
            "t": t, "x": x, "volts": volts, "resid_mV": resid_mV, "mask": mask}


def noise_variance(x):
    """ADC noise variance from a DC recording (generator at a constant 1.65 V)."""
    x = np.asarray(x, float)
    k = 51
    smooth = np.convolve(x, np.ones(k) / k, mode="valid")
    resid = x[k // 2:k // 2 + len(smooth)] - smooth
    return float(np.var(resid, ddof=1) * k / (k - 1))


# =============================================================================================
# 7. Frequency sweeps
# =============================================================================================
def log_sweep_freq(t, f1, f2, T):
    """f(t) = f1 (f2/f1)^(t/T)."""
    return f1 * (f2 / f1) ** (np.asarray(t, float) / T)


def sweep_starts(x, fs, f1, f2, T, seg_s=0.1, f_valid=None, min_snr=10.0, return_diag=False):
    """Find the start time (s) of every complete sweep in a recording.

    A spectrogram gives the peak frequency of each 0.1 s segment. Each segment with a clean peak
    below f_valid (default 0.4 fs, above which aliasing can fold the peak) predicts a start time
    t0 = t - T ln(f/f1) / ln(f2/f1). The predictions cluster at the true starts, and the median of
    each cluster is returned. Only sweeps lying completely inside the recording are kept.
    """
    x = np.asarray(x, float)
    if f_valid is None:  # below fs - f2 an aliased (folded) peak cannot occur
        f_valid = min(0.4 * fs, fs - f2) if f2 > 0.5 * fs else 0.4 * fs
    nper = int(seg_s * fs)
    freqs, tc, S = signal.spectrogram(x - x.mean(), fs, window="hann", nperseg=nper,
                                      noverlap=nper // 2, detrend="constant", mode="magnitude")
    S[:3] = 0.0
    k = np.argmax(S, axis=0)
    k = np.clip(k, 1, len(freqs) - 2)
    idx = np.arange(S.shape[1])
    la, lb, lc = (np.log(S[k + d, idx] + 1e-30) for d in (-1, 0, 1))
    shift = 0.5 * (la - lc) / (la - 2 * lb + lc)
    fpk = (k + shift) * (freqs[1] - freqs[0])
    snr = S[k, idx] / (np.median(S, axis=0) + 1e-30)
    good = (snr > min_snr) & (fpk > max(10 * freqs[1], 2 * f1)) & (fpk < f_valid) & (fpk < f2)
    t0c = tc[good] - T * np.log(fpk[good] / f1) / np.log(f2 / f1)
    starts = []
    if len(t0c):
        bins = np.arange(t0c.min() - 0.1, t0c.max() + 0.1, 0.02)
        h, e = np.histogram(t0c, bins)
        peaks, _ = signal.find_peaks(np.r_[0, h, 0], height=max(5, 0.25 * h.max()), distance=int(0.5 * T / 0.02))
        for pk in peaks - 1:
            centre = 0.5 * (e[pk] + e[pk + 1])
            near = t0c[np.abs(t0c - centre) < 0.06]
            starts.append(float(np.median(near)))
    duration = len(x) / fs
    starts = [s for s in sorted(starts) if s >= 0 and s + T <= duration]
    if return_diag:
        return starts, {"t": tc, "fpk": fpk, "good": good, "t0_candidates": t0c}
    return starts


def sweep_power(x, fs, t0, f1, f2, T, n_periods=1.0, min_half=4, noise_var=0.0):
    """Trim one sweep, subtract the mean, square, and smooth over n_periods of the local period.

    Returns (f, P): the instantaneous sweep frequency and smoothed mean-square signal per sample.
    """
    x = np.asarray(x, float)
    i0 = int(np.ceil(t0 * fs))
    n = int(np.floor(T * fs)) - 1
    seg = x[i0:i0 + n]
    if len(seg) < n:
        raise ValueError("sweep runs past the end of the recording")
    t = (i0 + np.arange(n)) / fs - t0
    f = log_sweep_freq(t, f1, f2, T)
    y2 = (seg - seg.mean()) ** 2
    half = np.maximum(np.round(0.5 * n_periods * fs / f).astype(int), min_half)
    S = np.concatenate([[0.0], np.cumsum(y2)])
    i = np.arange(n)
    lo = np.clip(i - half, 0, n)
    hi = np.clip(i + half + 1, 0, n)
    P = (S[hi] - S[lo]) / (hi - lo) - noise_var
    return f, P


def sweeps_from_files(paths, fs, f1, f2, T, column=0, **kw):
    """All complete sweeps in all files, as a list of (f, P), plus the start times found."""
    out, found = [], []
    for path in paths:
        x = load_recording(path)[:, column]
        starts = sweep_starts(x, fs, f1, f2, T)
        found.append((str(path), starts))
        for t0 in starts:
            out.append(sweep_power(x, fs, t0, f1, f2, T, **kw))
    return out, found


def _bin_means(f, P, edges):
    j = np.digitize(f, edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1)
    s = np.bincount(j[ok], weights=P[ok], minlength=len(edges) - 1)
    c = np.bincount(j[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def sweep_ratio(outs, refs, f_lo, f_hi, n_bins=120, rel_floor=0.0):
    """Power ratio in log-frequency bins, with uncertainty from the scatter of repeated sweeps.

    ratio = mean(out sweeps) / mean(reference sweeps); the relative error adds the standard error
    of each mean in quadrature. rel_floor adds a fractional floor if the repeats agree too well.
    """
    edges = np.geomspace(f_lo, f_hi, n_bins + 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    O = np.array([_bin_means(f, P, edges) for f, P in outs])
    R = np.array([_bin_means(f, P, edges) for f, P in refs])
    mo, mr = O.mean(0), R.mean(0)

    def sem(A):
        return A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros(A.shape[1])

    ratio = mo / mr
    rel = np.sqrt((sem(O) / mo) ** 2 + (sem(R) / mr) ** 2 + rel_floor**2)
    ok = np.isfinite(ratio) & np.isfinite(rel) & (rel > 0)
    # sub-grid uniform in log f inside each bin (a log sweep spends equal time per log-f step),
    # so the fits can compare each bin with the model averaged over that bin
    u = (np.arange(9) + 0.5) / 9
    f_sub = np.exp(np.log(edges[:-1])[:, None] + u[None, :] * np.diff(np.log(edges))[:, None])
    return {"f": fc[ok], "ratio": ratio[ok], "sigma": (ratio * rel)[ok], "f_sub": f_sub[ok],
            "per_sweep": (O / mr)[:, ok], "n_out": len(O), "n_ref": len(R)}


def bin_avg(model):
    """Wrap a power model so it returns the average over each bin's sub-grid (rows of f_sub)."""
    def wrapped(F, *p):
        return model(F, *p).mean(axis=1)
    return wrapped


def fit_lp1_sweep(sw, tau0=1e-4):
    return fit_scaled(bin_avg(P_lp1), sw["f_sub"], sw["ratio"], sw["sigma"], [tau0], [1e-4], ["tau"],
                      bounds=([1e-7], [1e-1]))


def fit_lp2_sweep(sw, b0=3e-4, c0=1e-8):
    return fit_scaled(bin_avg(P_lp2), sw["f_sub"], sw["ratio"], sw["sigma"], [b0, c0], [1e-4, 1e-8], ["b", "c"],
                      bounds=([1e-7, 1e-12], [1e-1, 1e-4]))


def fit_bp_sweep(sw, K0=5e-4, b0=8e-4, c0=5e-8):
    return fit_scaled(bin_avg(P_bp), sw["f_sub"], sw["ratio"], sw["sigma"], [K0, b0, c0], [1e-4, 1e-4, 1e-8],
                      ["K", "b", "c"], bounds=([1e-7, 1e-7, 1e-12], [1e-1, 1e-1, 1e-4]))


def bp_derived_from_fit(fit, n=20000, seed=3):
    """f0, df, Q, peak K/b, (K/b)^2, (K/b) df and the edges, with errors from the fit covariance."""
    d = mvn_draws(fit, n, seed)
    der = bp_derived(d[:, 0], d[:, 1], d[:, 2])
    best = bp_derived(*fit.p)
    return {k: (float(best[k]), float(np.std(v, ddof=1))) for k, v in der.items()}


def lp2_half_power_from_fit(fit, n=20000, seed=4):
    d = mvn_draws(fit, n, seed)
    return float(lp2_half_power(*fit.p)), float(np.std(lp2_half_power(d[:, 0], d[:, 1]), ddof=1))


def half_power_from_curve(f, r):
    """Model-free half-power edges of a band-pass curve: peak by a parabola in log f, then linear
    interpolation in log f at half of the peak."""
    f = np.asarray(f, float)
    r = np.asarray(r, float)
    lf = np.log(f)
    i = int(np.nanargmax(r))
    sl = slice(max(0, i - 4), min(len(r), i + 5))
    a, b, c = np.polyfit(lf[sl], r[sl], 2)
    lpk = -b / (2 * a) if a < 0 else lf[i]
    rp = float(np.polyval([a, b, c], lpk)) if a < 0 else float(r[i])
    half = rp / 2.0

    def cross(rng):
        for j in rng:
            j2 = j + (1 if rng.step < 0 else -1)
            if r[j] < half <= r[j2] or r[j2] >= half > r[j]:
                return float(np.exp(np.interp(half, [r[j], r[j2]], [lf[j], lf[j2]])))
        return np.nan

    lo = cross(range(i - 1, -1, -1))
    hi = cross(range(i + 1, len(r)))
    return {"f_lo": lo, "f_hi": hi, "peak_power": rp, "f_peak": float(np.exp(lpk)), "df": hi - lo}


def band_at_level(f, r, level):
    """Frequencies where a band-pass curve crosses a fixed absolute level of P/P0.

    Comparing both forms at the same absolute level shows whether loading passes more signal
    anywhere, which the half-power definition (relative to each curve's own peak) cannot show.
    """
    f = np.asarray(f, float)
    r = np.asarray(r, float)
    lf = np.log(f)
    i = int(np.nanargmax(r))

    def cross(idx, step):
        for j in idx:
            jp = j - step  # neighbour on the peak side
            if r[j] < level <= r[jp]:
                return float(np.exp(np.interp(level, [r[j], r[jp]], [lf[j], lf[jp]])))
        return np.nan

    lo = cross(range(i - 1, -1, -1), -1)
    hi = cross(range(i + 1, len(r)), +1)
    return {"f_lo": lo, "f_hi": hi, "width": hi - lo}


def model_free_edges(sw):
    """Half-power edges of the mean curve and of each sweep (for the standard error)."""
    each = pd.DataFrame([half_power_from_curve(sw["f"], r) for r in sw["per_sweep"]])
    mean = half_power_from_curve(sw["f"], sw["ratio"])
    sem = each.std(ddof=1) / np.sqrt(len(each)) if len(each) > 1 else each.std() * np.nan
    return {k: (mean[k], float(sem[k])) for k in mean}


def loaded_over_buffered(swL, swB):
    """Ratio of the loaded to the buffered power curves on the shared frequency bins."""
    f = np.intersect1d(np.round(swL["f"], 6), np.round(swB["f"], 6))
    iL = np.isin(np.round(swL["f"], 6), f)
    iB = np.isin(np.round(swB["f"], 6), f)
    q = swL["ratio"][iL] / swB["ratio"][iB]
    sq = q * np.hypot(swL["sigma"][iL] / swL["ratio"][iL], swB["sigma"][iB] / swB["ratio"][iB])
    z = (q - 1.0) / sq
    j = int(np.argmax(q))
    return {"f": swL["f"][iL], "q": q, "sigma": sq, "max_q": float(q[j]), "max_q_sigma": float(sq[j]),
            "f_at_max": float(swL["f"][iL][j]), "n_above_1_by_2sigma": int(np.sum(z > 2.0)),
            "lowest_f": float(swL["f"][iL][0]), "q_lowest": float(q[0]), "s_q_lowest": float(sq[0])}


# =============================================================================================
# 8. Phase points
# =============================================================================================
def phase_point(xA, xB, fs, f_guess, dt=0.0, sdt=0.0, gain_ratio=1.0):
    """Phase and amplitude of output (channel B) relative to input (channel A) at one frequency.

    The frequency is fitted on the input, both channels are fitted at that frequency, and the
    delay between paired samples is removed: phase = arg(Z_B / Z_A) - 2 pi f dt.
    """
    t = np.arange(len(xA)) / fs
    f, _ = fit_frequency(xA, t, f_guess)
    pa, pb = phasors(xA, t, [f]), phasors(xB, t, [f])
    H = pb["Z"][0] / pa["Z"][0] / gain_ratio * np.exp(-1j * TWO_PI * f * dt)
    s_ph = np.sqrt(pa["s_phase"][0] ** 2 + pb["s_phase"][0] ** 2 + (TWO_PI * f * sdt) ** 2)
    amp = np.abs(H)
    s_amp = amp * np.hypot(pa["s_amp"][0] / np.abs(pa["Z"][0]), pb["s_amp"][0] / np.abs(pb["Z"][0]))
    return {"f": f, "phase_deg": float(np.degrees(np.angle(H))), "s_phase_deg": float(np.degrees(s_ph)),
            "amp": float(amp), "s_amp": float(s_amp)}


def phase_set(paths, fs, dt=0.0, sdt=0.0, gain_ratio=1.0, phase_floor_deg=0.0):
    """Phase points from files named like lp1_700Hz.csv (columns ch34, ch35)."""
    rows = []
    for path in sorted(paths, key=lambda p: freq_from_name(p) or 0):
        d = load_recording(path)
        r = phase_point(d[:, 0], d[:, 1], fs, freq_from_name(path), dt, sdt, gain_ratio)
        r["s_phase_deg"] = float(np.hypot(r["s_phase_deg"], phase_floor_deg))
        r["file"] = str(path)
        rows.append(r)
    return pd.DataFrame(rows)


def fit_lp1_phase(ph, tau0=1e-4):
    return fit_scaled(phase_lp1_deg, ph.f.to_numpy(), ph.phase_deg.to_numpy(), ph.s_phase_deg.to_numpy(),
                      [tau0], [1e-4], ["tau"], bounds=([1e-7], [1e-1]))


def fit_bp_phase(ph, b0=8e-4, c0=5e-8):
    return fit_scaled(phase_bp_deg, ph.f.to_numpy(), ph.phase_deg.to_numpy(), ph.s_phase_deg.to_numpy(),
                      [b0, c0], [1e-4, 1e-8], ["b", "c"], bounds=([1e-7, 1e-12], [1e-1, 1e-4]))


def bp_phase_marks(fit, f_eval=2000.0, n=20000, seed=5):
    """Zero crossing, +45 and -45 degree frequencies and the phase at f_eval, with errors."""
    d = mvn_draws(fit, n, seed)

    def marks(b, c):
        lo, hi = bp_edges(b, c)
        return {"f_zero": 1.0 / (TWO_PI * np.sqrt(c)), "f_p45": lo, "f_m45": hi,
                f"phase_{int(f_eval)}Hz": phase_bp_deg(f_eval, b, c)}

    best, draws = marks(*fit.p), marks(d[:, 0], d[:, 1])
    return {k: (float(best[k]), float(np.std(draws[k], ddof=1))) for k in best}


def generator_resistance(tau_sweep, s_tau_sweep, tau_phase, s_tau_phase, R1, s_R1):
    """Rg = R1 (tau_sweep / tau_phase - 1), the ratio of the two fitted single-stage cutoffs."""
    q = tau_sweep / tau_phase
    sq = q * np.hypot(s_tau_sweep / tau_sweep, s_tau_phase / tau_phase)
    Rg = R1 * (q - 1.0)
    return Rg, float(np.hypot(s_R1 * (q - 1.0), R1 * sq))


# =============================================================================================
# 9. Square-wave bursts
# =============================================================================================
def square_burst(xA, xB, fs, f0_guess=500.0, dt=0.0, gain_ratio=1.0, harmonics=None):
    """Harmonic transfer of one two-channel square-wave burst.

    The fundamental is fitted on the input, the burst is trimmed to a whole number of periods,
    and every harmonic below the Nyquist frequency is measured at exactly k f0 (equivalent to the
    DFT of the trimmed record). H_k = Z_out,k / Z_in,k with the channel delay removed.
    """
    xA = np.asarray(xA, float)
    xB = np.asarray(xB, float)
    t = np.arange(len(xA)) / fs
    f0, _ = fit_frequency(xA, t, f0_guess)
    n_per = int(np.floor(len(xA) * f0 / fs))
    N = int(np.round(n_per * fs / f0))
    N = min(N, len(xA))
    t, xA, xB = t[:N], xA[:N], xB[:N]
    kmax = int(np.floor((0.5 * fs - 1.0) / f0))
    k = np.arange(1, kmax + 1) if harmonics is None else np.asarray(harmonics)
    pa, pb = phasors(xA, t, k * f0), phasors(xB, t, k * f0)
    H = pb["Z"] / pa["Z"] / gain_ratio * np.exp(-1j * TWO_PI * k * f0 * dt)
    return {"f0": f0, "k": k, "ZA": pa["Z"], "ZB": pb["Z"], "H": H, "dcA": pa["dc"], "dcB": pb["dc"],
            "t": t, "xA": xA, "xB": xB, "fs": fs, "dt": dt, "gain_ratio": gain_ratio}


def square_summary(bursts, harmonics=(1, 3, 5, 7)):
    """Mean and standard error over bursts of |H_k|, |H_k|/|H_1|, phase, and the phase spread."""
    recs = []
    for bu in bursts:
        rec = {}
        Hs = {kk: bu["H"][list(bu["k"]).index(kk)] for kk in harmonics}
        f0 = bu["f0"]
        for kk in harmonics:
            fk = int(round(kk * 500)) if abs(f0 - 500) < 5 else int(round(kk * f0))
            rec[f"H{fk}"] = np.abs(Hs[kk])
            rec[f"phase{fk}"] = np.degrees(np.angle(Hs[kk]))
            if kk != harmonics[0]:
                rec[f"norm{fk}"] = np.abs(Hs[kk]) / np.abs(Hs[harmonics[0]])
        ph = [np.degrees(np.angle(Hs[kk])) for kk in harmonics]
        rec["spread_deg"] = max(ph) - min(ph)
        recs.append(rec)
    df = pd.DataFrame(recs)
    n = len(df)
    return pd.DataFrame({"mean": df.mean(), "sem": df.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan})


def predict_square_output(bu, K, b, c):
    """Output predicted from the measured input harmonics below Nyquist and a model (terminal view).

    Returns the predicted output samples, the RMS residual, and the RMS residual as a fraction of
    the measured peak-to-peak output.
    """
    fk = bu["k"] * bu["f0"]
    Zp = bu["ZA"] * H_bp(fk, K, b, c) * bu["gain_ratio"]
    tB = bu["t"] + bu["dt"]
    y = bu["dcB"] + np.real(np.exp(1j * TWO_PI * np.outer(tB, fk)) @ Zp)
    resid = bu["xB"] - y
    rms = float(np.sqrt(np.mean(resid**2)))
    pp = float(np.ptp(bu["xB"]))
    return y, rms, rms / pp


# =============================================================================================
# 10. Comparison table
# =============================================================================================
def zscore(meas, s_meas, pred, s_pred):
    return (meas - pred) / np.sqrt(s_meas**2 + s_pred**2)


class Report:
    """Collects measured-versus-predicted comparisons and prints them with z-scores."""

    def __init__(self, mc=None):
        self.rows = []
        self.mc = mc  # dict of Monte Carlo prediction samples

    def add(self, name, meas, s_meas, pred=None, s_pred=None, key=None, unit="", scale=1.0):
        if key is not None:
            v = self.mc[key]
            pred, s_pred = float(np.mean(v)), float(np.std(v, ddof=1))
        z = zscore(meas, s_meas, pred, s_pred)
        self.rows.append({"quantity": name, "unit": unit, "measured": meas * scale, "s_meas": s_meas * scale,
                          "predicted": pred * scale, "s_pred": s_pred * scale, "z": z,
                          "within_2sigma": bool(abs(z) < 2)})

    def table(self):
        return pd.DataFrame(self.rows).set_index("quantity")
