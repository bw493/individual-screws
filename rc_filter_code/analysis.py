"""Full analysis of the cascaded RC filter data (Physics 4BL final project).

Follows the proposal's Reading Data steps 1 to 7 with rcfilter.py, with three changes forced by the
recordings themselves:
  * every file has its own sampling rate (the ESP32 could not hold the 25 us / 49 us timer periods,
    so each sample took about analogRead time + overhead); rates come from the data,
  * the sweeps are linear (20 Hz to 10 kHz), not logarithmic,
  * the channel delay is scaled with each file's measured pair period.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import rcfilter as rc

D = Path("data")
OUT = Path("out")
OUT.mkdir(exist_ok=True)
F1, F2 = 20.0, 10000.0
PHASE_F = 2000.0
N_MC = 10000
BOUND_IS = "rectangular"
RES = {}

# ------------------------------------------------------------------------------------------------
# Step 7 first: Monte Carlo predictions from the measured components
# ------------------------------------------------------------------------------------------------
RES_1K = dict(pct=0.8, counts=2, resolution=1.0)
RES_10K = dict(pct=0.8, counts=2, resolution=10.0)
CAP = dict(pct=3.0, counts=5, resolution=0.1e-9)
parts = {
    "R1": rc.part(987.0, **RES_1K), "R2": rc.part(987.0, **RES_1K), "R3": rc.part(987.0, **RES_1K),
    "R4": rc.part(9970.0, **RES_10K), "R5": rc.part(9970.0, **RES_10K),
    "C1": rc.part(99.7e-9, **CAP), "C2": rc.part(99.7e-9, **CAP),
    "C3": rc.part(99.7e-9, **CAP), "C4": rc.part(99.7e-9, **CAP),
    "Rg": rc.part(50.0, pct=2.0),
}
draws = rc.draw_parts(parts, N_MC, seed=1, bound_is=BOUND_IS)
mc = rc.predict(draws, phase_f=PHASE_F)
nominal = rc.predict({k: v["value"] for k, v in parts.items()}, phase_f=PHASE_F)
PM_ = {k: float(np.mean(v)) for k, v in mc.items()}
PS_ = {k: float(np.std(v, ddof=1)) for k, v in mc.items()}


def std_unc(name):
    b = parts[name]["bound"]
    return b / np.sqrt(3) if BOUND_IS == "rectangular" else b


# ------------------------------------------------------------------------------------------------
# Step 1: calibration
# ------------------------------------------------------------------------------------------------
x = rc.load_recording(D / "calib/stream_1500Hz.csv")[:, 0]
fs_stream, s_fs_stream = rc.sampling_rate(x, 1500.0)
b15 = rc.load_recording(D / "calib/burst_1500Hz.csv")
fsA, sA = rc.sampling_rate(b15[:, 0], 1500.0)
fsB, sB = rc.sampling_rate(b15[:, 1], 1500.0)
fs_pair = (fsA / sA**2 + fsB / sB**2) / (1 / sA**2 + 1 / sB**2)
s_fs_pair = 1 / np.sqrt(1 / sA**2 + 1 / sB**2)
T_PAIR_CAL = 1.0 / fs_pair
RES["calib"] = {"fs_stream": fs_stream, "s_fs_stream": s_fs_stream, "fs_pair": fs_pair, "s_fs_pair": s_fs_pair,
                "aR_stream": float(rc.read_meta(D / "calib/stream_1500Hz.csv")["analogRead_us"]),
                "aR_burst": float(rc.read_meta(D / "calib/burst_1500Hz.csv")["analogRead_us"])}


def file_rate(d, f_nom, col=0):
    """Pair (or stream) rate of one file from its known generator frequency."""
    return rc.sampling_rate(d[:, col], f_nom)


# channel delay, each delay file with its own pair rate, from short windows (see windowed_H below)
rows = []
for p in sorted((D / "calib").glob("delay_*Hz.csv"), key=rc.freq_from_name):
    f_nom = rc.freq_from_name(p)
    d = rc.load_recording(p)
    fs_f, _ = file_rate(d, f_nom)
    rw = []
    for i in range(len(d) // 1000):
        seg = d[i * 1000:(i + 1) * 1000]
        n = np.arange(len(seg), dtype=float)
        fc, _ = rc.fit_frequency(seg[:, 0], n, f_nom / fs_f)
        rw.append(rc.phasors(seg[:, 1], n, [fc])["Z"][0] / rc.phasors(seg[:, 0], n, [fc])["Z"][0])
    rw = np.array(rw)
    rm = rw.mean()
    dphi = float(np.angle(rm))
    s_dphi = float(np.angle(rw / rm).std(ddof=1) / np.sqrt(len(rw)))
    rows.append((f_nom, dphi, s_dphi, float(np.abs(rw).mean()), fs_f))
dl = pd.DataFrame(rows, columns=["f", "dphi_rad", "s_dphi_rad", "gain", "fs"])
# the three files share one pair period, so one delay fits all three: dphi = 2 pi f dt (GPIO 35 sampled later)
w = 1 / dl.s_dphi_rad**2
xx = 2 * np.pi * dl.f
dt_cal = float(np.sum(w * xx * dl.dphi_rad) / np.sum(w * xx**2))
s_dt_stat = float(1 / np.sqrt(np.sum(w * xx**2)))
chi2 = float(np.sum(w * (dl.dphi_rad - xx * dt_cal) ** 2))
chi2_red_dt = chi2 / (len(dl) - 1)
s_dt = s_dt_stat * np.sqrt(max(1, chi2_red_dt))
GAIN = float(dl.gain.mean())
S_GAIN = float(dl.gain.std(ddof=1) / np.sqrt(len(dl)))
T_PAIR_DELAY = float(np.mean(1 / dl.fs))
dl["resid_deg"] = np.degrees(dl.dphi_rad - xx * dt_cal)
RES["delay"] = {"dt_us": dt_cal * 1e6, "s_dt_us": s_dt * 1e6, "chi2_red": chi2_red_dt, "gain": GAIN, "s_gain": S_GAIN,
                "T_pair_us": T_PAIR_DELAY * 1e6, "table": dl.to_dict("list")}


def delay_for(fs_file):
    """Delay between paired samples scaled to a file's pair period (half of the change goes to each read)."""
    return dt_cal + 0.5 * (1.0 / fs_file - T_PAIR_DELAY)


# ADC linearity from the 1 Hz triangle (200 us per pair, no late pairs)
tri_d = rc.load_recording(D / "calib/triangle_1Hz.csv")
TRI = {}
for col, pin in ((0, 34), (1, 35)):
    tri = rc.triangle_linearity(tri_d[:, col], 5000.0, 0.9, 2.4)
    TRI[pin] = tri
RES["triangle"] = {pin: {k: TRI[pin][k] for k in ("counts_per_volt", "counts_at_0V", "f", "resid_rms_mV",
                                                    "resid_max_mV", "max_dev_pct_of_span")} for pin in TRI}

noise_x = rc.load_recording(D / "calib/noise_dc.csv")[:, 0]
NOISE_VAR = rc.noise_variance(noise_x)
RES["noise"] = {"var_counts2": NOISE_VAR, "rms_counts": float(np.sqrt(NOISE_VAR)),
                "rms_mV": float(np.sqrt(NOISE_VAR) / TRI[35]["counts_per_volt"] * 1e3)}

# ------------------------------------------------------------------------------------------------
# Step 2: linear sweeps -> power ratios
# ------------------------------------------------------------------------------------------------
R_NOM = (F2 - F1) / 20.0


def chirp_track(x, nper=1024):
    """Spectrogram peak (cycles per sample) of each segment, with parabolic interpolation."""
    f, t, S = signal.spectrogram(x - x.mean(), 1.0, nperseg=nper, noverlap=nper // 2, mode="magnitude")
    S[:3] = 0
    k = np.clip(S.argmax(0), 1, len(f) - 2)
    idx = np.arange(S.shape[1])
    la, lb, lc = (np.log(S[k + dd, idx] + 1e-30) for dd in (-1, 0, 1))
    fpk = (k + 0.5 * (la - lc) / (la - 2 * lb + lc)) * (f[1] - f[0])
    snr = S[k, idx] / (np.median(S, 0) + 1e-30)
    return t, fpk, snr


def find_sweeps(x, fs_guess):
    """Start sample and chirp slope (cycles/sample^2) of every rising sweep segment."""
    t, fpk, snr = chirp_track(x)
    fHz = fpk * fs_guess
    good = (snr > 10) & (fHz > 150) & (fHz < 0.4 * fs_guess)
    n0 = t[good] - (fHz[good] - F1) / R_NOM * fs_guess
    order = np.sort(n0)
    clusters, cur = [], [order[0]]
    for v in order[1:]:
        if v - cur[-1] < 0.3 * fs_guess:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    out = []
    for c in clusters:
        if len(c) < 50:
            continue
        c0 = np.median(c)
        sel = good & (np.abs((t - (fHz - F1) / R_NOM * fs_guess) - c0) < 0.3 * fs_guess)
        A = np.vstack([t[sel], np.ones(sel.sum())]).T
        (sl, ic), *_ = np.linalg.lstsq(A, fpk[sel], rcond=None)
        if sel.sum() >= 50 and sl > 0:
            out.append({"sl": sl, "ic": ic, "npts": int(sel.sum())})
    return out


# Reference: its sampling rate equals the stream calibration (same analogRead time to 0.03 us).
SWEEP_INFO = {}
raw = {}
for name in ("ref", "lp1", "lp2", "bpL", "bpB"):
    raw[name] = rc.load_recording(D / f"sweeps/{name}.csv")[:, 0]

meta_ref = rc.read_meta(D / "sweeps/ref.csv")
aR_ref = float(meta_ref["analogRead_us"])
# correct the stream rate for the 0.03 us analogRead difference (period = analogRead + overhead)
overhead_stream = 1e6 / fs_stream - RES["calib"]["aR_stream"]
fs_ref = 1e6 / (aR_ref + overhead_stream)
segs = find_sweeps(raw["ref"], fs_ref)
sl_ref = np.mean([s["sl"] for s in segs])
R_TRUE = sl_ref * fs_ref**2           # real chirp rate of the generator (Hz/s)
P_SWEEP = (F2 - F1) / R_TRUE           # real sweep duration (s)
RES["sweep_cal"] = {"fs_ref": fs_ref, "R_true": R_TRUE, "P_sweep": P_SWEEP, "overhead_stream_us": overhead_stream}


def sweeps_of(name):
    x = raw[name]
    aR = float(rc.read_meta(D / f"sweeps/{name}.csv")["analogRead_us"])
    fs0 = 1e6 / (aR + overhead_stream)
    segs = find_sweeps(x, fs0)
    sl = np.mean([s["sl"] for s in segs])
    fs = np.sqrt(R_TRUE / sl) if name != "ref" else fs_ref
    N = P_SWEEP * fs
    starts = []
    for s in segs:
        n0 = (F1 / fs - s["ic"]) / s["sl"]
        if n0 >= 0 and n0 + N <= len(x):
            starts.append(n0)
    spacing = np.diff([(F1 / fs - s["ic"]) / s["sl"] for s in segs])
    SWEEP_INFO[name] = {"aR_us": aR, "fs": float(fs), "fs_from_aR": float(fs0), "starts_s": [float(v / fs) for v in starts],
                        "n_complete": len(starts), "spacing_s": [float(v / fs) for v in spacing]}
    # additive noise of this recording: high-pass (> 800 Hz) the first 0.2 s of each sweep (20 to 120 Hz),
    # where the tone has no power above 800 Hz, and extrapolate the white level to the full band
    sos = signal.butter(6, 800, "hp", fs=fs, output="sos")
    nv = []
    for n0 in starts:
        seg = x[int(n0):int(n0) + int(0.2 * fs)].astype(float)
        y = signal.sosfiltfilt(sos, seg - seg.mean())[200:-200]
        nv.append(y.var() * (fs / 2) / (fs / 2 - 800))
    noise_var = float(np.mean(nv))
    SWEEP_INFO[name]["noise_var"] = noise_var
    SWEEP_INFO[name]["noise_var_each"] = [float(v) for v in nv]
    out = []
    for n0 in starts:
        i0 = int(np.ceil(n0))
        n = int(np.floor(N)) - 1
        seg = x[i0:i0 + n]
        f = F1 + R_TRUE * ((i0 + np.arange(n)) - n0) / fs
        y2 = (seg - seg.mean()) ** 2
        half = np.maximum(np.round(0.5 * fs / f).astype(int), 4)     # one local period
        S = np.concatenate([[0.0], np.cumsum(y2)])
        i = np.arange(n)
        lo = np.clip(i - half, 0, n)
        hi = np.clip(i + half + 1, 0, n)
        P = (S[hi] - S[lo]) / (hi - lo) - noise_var
        out.append((f, P))
    return out


def make_edges(f_lo, f_hi, frac=0.045):
    """Bins at least frac wide and at least as wide as one-period smoothing (R/f) of a linear sweep."""
    e = [f_lo]
    while e[-1] < f_hi:
        e.append(e[-1] * (1 + max(frac, R_TRUE / e[-1] ** 2)))
    e[-1] = f_hi
    return np.array(e)


def bin_means(f, P, edges):
    j = np.digitize(f, edges) - 1
    ok = (j >= 0) & (j < len(edges) - 1)
    s = np.bincount(j[ok], weights=P[ok], minlength=len(edges) - 1)
    c = np.bincount(j[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


FS_MIN = None
SW_RAW = {name: sweeps_of(name) for name in ("ref", "lp1", "lp2", "bpL", "bpB")}
FS_MIN = min(v["fs"] for v in SWEEP_INFO.values())
FIT_LO, FIT_HI = 50.0, round(0.4 * FS_MIN, -2)
EDGES = make_edges(FIT_LO, FIT_HI)


def sweep_ratio(name):
    outs, refs = SW_RAW[name], SW_RAW["ref"]
    fc = np.sqrt(EDGES[:-1] * EDGES[1:])
    O = np.array([bin_means(f, P, EDGES) for f, P in outs])
    Rr = np.array([bin_means(f, P, EDGES) for f, P in refs])
    mo, mr = O.mean(0), Rr.mean(0)
    sem = lambda A: A.std(0, ddof=1) / np.sqrt(len(A))
    ratio = mo / mr
    rel = np.sqrt((sem(O) / mo) ** 2 + (sem(Rr) / mr) ** 2)
    ok = np.isfinite(ratio) & np.isfinite(rel) & (rel > 0)
    u = (np.arange(9) + 0.5) / 9      # linear sweep: equal time per Hz, so the sub-grid is linear in f
    f_sub = EDGES[:-1, None] + u[None, :] * np.diff(EDGES)[:, None]
    return {"f": fc[ok], "ratio": ratio[ok], "sigma": (ratio * rel)[ok], "f_sub": f_sub[ok],
            "per_sweep": (O / mr)[:, ok], "n_out": len(O), "n_ref": len(Rr)}


# Generator amplitude in each recording block, from the windowed input-channel (GPIO 34) amplitude at
# 100 and 200 Hz, where the circuits barely load the generator. Block A (reference) uses the 200 Hz delay
# file and the reference sweep itself near 100 and 200 Hz.
def win_amp(path, f_nom, col=0):
    d = rc.load_recording(path)
    fs_f, _ = file_rate(d, f_nom)
    a = []
    for i in range(len(d) // 1000):
        seg = d[i * 1000:(i + 1) * 1000]
        n = np.arange(1000.0)
        fc, _ = rc.fit_frequency(seg[:, 0], n, f_nom / fs_f)
        a.append(abs(rc.phasors(seg[:, col], n, [fc])["Z"][0]))
    return float(np.mean(a))


ref_amp = []
for f, P in SW_RAW["ref"]:
    for f0 in (100.0, 200.0):
        ref_amp.append(np.sqrt(2 * np.mean(P[np.abs(f - f0) < 5])))
A_REF_SWEEP = float(np.mean(ref_amp)) / GAIN
A_DELAY200 = win_amp(D / "calib/delay_200Hz.csv", 200.0)
A_BLOCK = {"A": 0.5 * (A_REF_SWEEP + A_DELAY200)}
for blk, key in (("B", "lp1"), ("D", "bpL"), ("E", "bpB")):
    A_BLOCK[blk] = float(np.mean([win_amp(D / f"phase/{key}_{f}Hz.csv", f) for f in (100, 200)]))
S_AMP_REL = abs(A_REF_SWEEP - A_DELAY200) / 2 / A_BLOCK["A"]          # repeatability within block A
S_G = 2 * np.sqrt(2) * S_AMP_REL                                     # relative uncertainty of a power factor
G = {"lp1": (A_BLOCK["B"] / A_BLOCK["A"]) ** 2, "bpL": (A_BLOCK["D"] / A_BLOCK["A"]) ** 2,
     "bpB": (A_BLOCK["E"] / A_BLOCK["A"]) ** 2, "lp2": None}
RES["gen_amp"] = {"A_ref_sweep": A_REF_SWEEP, "A_delay200": A_DELAY200, "blocks": A_BLOCK, "s_amp_rel": S_AMP_REL,
                  "G": G, "s_G_rel": S_G}

SW_UNCORR = {k: sweep_ratio(k) for k in ("lp1", "lp2", "bpL", "bpB")}
SW = {}
for k, s in SW_UNCORR.items():
    g = G[k] or 1.0
    SW[k] = dict(s, ratio=s["ratio"] / g, sigma=s["sigma"] / g, per_sweep=s["per_sweep"] / g)
RES["sweeps"] = {"info": SWEEP_INFO, "fit_lo": FIT_LO, "fit_hi": FIT_HI, "n_bins": len(EDGES) - 1,
                 "median_rel_sigma": {k: float(np.median(v["sigma"] / v["ratio"])) for k, v in SW.items()}}

# ------------------------------------------------------------------------------------------------
# Step 3: low-pass fits
# ------------------------------------------------------------------------------------------------
def scaled(fit):
    return np.sqrt(max(1.0, fit.chi2_red))


fit1 = rc.fit_scaled(rc.bin_avg(rc.P_lp1), SW["lp1"]["f_sub"], SW["lp1"]["ratio"], SW["lp1"]["sigma"], [1e-4], [1e-4],
                     ["tau"], bounds=([1e-7], [1e-1]))
tau1, s_tau1 = fit1.p[0], fit1.perr_scaled[0]
fc1, s_fc1 = 1 / (2 * np.pi * tau1), s_tau1 / (2 * np.pi * tau1**2)
# single stage, free-scale check
fit1A = rc.fit_scaled(rc.bin_avg(lambda f, A, tau: A * rc.P_lp1(f, tau)), SW["lp1"]["f_sub"], SW["lp1"]["ratio"],
                      SW["lp1"]["sigma"], [1.0, 1.4e-4], [1.0, 1e-4], ["A", "tau"])
# two stage: block C has no two-channel recording, so the generator amplitude is unknown there and the
# overall scale A is left free (the sweep sits at P/P0 = 0.91 at 50 to 100 Hz instead of 0.99)
P_lp2A = lambda f, A, b, c: A * rc.P_lp2(f, b, c)
fit2_fixed = rc.fit_scaled(rc.bin_avg(rc.P_lp2), SW["lp2"]["f_sub"], SW["lp2"]["ratio"], SW["lp2"]["sigma"], [3e-4, 1e-8],
                           [1e-4, 1e-8], ["b", "c"], bounds=([1e-7, 1e-12], [1e-1, 1e-4]))
fit2A = rc.fit_scaled(rc.bin_avg(P_lp2A), SW["lp2"]["f_sub"], SW["lp2"]["ratio"], SW["lp2"]["sigma"], [0.92, 3e-4, 1e-8],
                      [1.0, 1e-4, 1e-8], ["A", "b", "c"], bounds=([0.5, 1e-7, 1e-12], [1.5, 1e-1, 1e-4]))
fit2 = rc.Fit(["b", "c"], fit2A.p[1:], fit2A.cov[1:, 1:], fit2A.chi2, fit2A.dof)
lp2_models = {}
s2 = SW["lp2"]
for label, bkey in (("loaded", "lp2.b_loaded"), ("ideal", "lp2.b_ideal")):
    bb, cc = PM_[bkey], PM_["lp2.c"]
    fA = rc.fit_scaled(rc.bin_avg(lambda f, A: A * rc.P_lp2(f, bb, cc)), s2["f_sub"], s2["ratio"], s2["sigma"], [0.92], [1.0], ["A"])
    lp2_models[label] = {"chi2_red": fA.chi2_red, "A": float(fA.p[0])}
fh2, s_fh2 = rc.lp2_half_power_from_fit(fit2)
s_fh2 *= scaled(fit2)
RES["lp"] = {"tau1_us": tau1 * 1e6, "s_tau1_us": s_tau1 * 1e6, "fc1": fc1, "s_fc1": s_fc1, "chi2_red1": fit1.chi2_red,
             "dof1": fit1.dof, "b2": fit2.p[0], "s_b2": fit2.perr_scaled[0], "c2": fit2.p[1], "s_c2": fit2.perr_scaled[1],
             "chi2_red2": fit2.chi2_red, "dof2": fit2.dof, "chi2_red_at_pred": lp2_models, "fhalf2": fh2, "s_fhalf2": s_fh2,
             "A2": fit2A.p[0], "s_A2": fit2A.perr_scaled[0], "fixed_scale_fit": {"b": fit2_fixed.p[0], "c": fit2_fixed.p[1],
             "chi2_red": fit2_fixed.chi2_red, "fhalf": rc.lp2_half_power(*fit2_fixed.p)},
             "lp1_free_scale": {"A": fit1A.p[0], "tau_us": fit1A.p[1] * 1e6, "fc": 1 / (2 * np.pi * fit1A.p[1]),
                                "chi2_red": fit1A.chi2_red}}

# ------------------------------------------------------------------------------------------------
# Step 4: band-pass fits and model-free checks
# ------------------------------------------------------------------------------------------------
def model_free_edges(sw):
    each = pd.DataFrame([rc.half_power_from_curve(sw["f"], r) for r in sw["per_sweep"]])
    mean = rc.half_power_from_curve(sw["f"], sw["ratio"])
    sem = each.std(ddof=1) / np.sqrt(len(each))
    return {k: (float(mean[k]), float(sem[k])) for k in mean}


BP = {}
for key in ("bpL", "bpB"):
    s = SW[key]
    fit = rc.fit_scaled(rc.bin_avg(rc.P_bp), s["f_sub"], s["ratio"], s["sigma"], [5e-4, 8e-4, 5e-8],
                        [1e-4, 1e-4, 1e-8], ["K", "b", "c"], bounds=([1e-7, 1e-7, 1e-12], [1e-1, 1e-1, 1e-4]))
    der = rc.bp_derived_from_fit(fit)
    der = {n: (v, e * scaled(fit)) for n, (v, e) in der.items()}
    # the generator-amplitude correction is uncertain by S_G in power: K-dependent quantities carry it
    for n, pw in (("peak_amp", 0.5), ("product", 0.5), ("peak_power", 1.0)):
        v, e = der[n]
        der[n] = (v, float(np.hypot(e, v * pw * S_G)))
    BP[key] = {"fit": fit, "der": der, "free": model_free_edges(s)}
cL, cB = BP["bpL"]["fit"]["c"], BP["bpB"]["fit"]["c"]
c_change = 100 * (cL[0] / cB[0] - 1)
s_c_change = 100 * cL[0] / cB[0] * np.hypot(BP["bpL"]["fit"].perr_scaled[2] / cL[0], BP["bpB"]["fit"].perr_scaled[2] / cB[0])

# loaded / buffered on the shared bins
sL, sB_ = SW["bpL"], SW["bpB"]
q = sL["ratio"] / sB_["ratio"]
sq = q * np.sqrt((sL["sigma"] / sL["ratio"]) ** 2 + (sB_["sigma"] / sB_["ratio"]) ** 2 + S_G**2)
zq = (q - 1) / sq
jq = int(np.argmax(q))
fl = float(sL["f"][0])
qpred = rc.P_bp(fl, mc["bpL.K"], mc["bpL.b"], mc["bpL.c"]) / rc.P_bp(fl, mc["bpB.K"], mc["bpB.b"], mc["bpB.c"])
LB = {"f": sL["f"], "q": q, "sigma": sq, "max_q": float(q[jq]), "max_q_sigma": float(sq[jq]), "f_at_max": float(sL["f"][jq]),
      "n_above_1_by_2sigma": int(np.sum(zq > 2)), "lowest_f": fl, "q_lowest": float(q[0]), "s_q_lowest": float(sq[0]),
      "q_pred_lowest": float(np.mean(qpred)), "s_q_pred_lowest": float(np.std(qpred, ddof=1))}
LEVEL = BP["bpL"]["free"]["peak_power"][0] / 2
BAND = {k: rc.band_at_level(SW[k]["f"], SW[k]["ratio"], LEVEL) for k in ("bpL", "bpB")}
RES["bp"] = {k: {"fit": dict(zip(BP[k]["fit"].names, map(float, BP[k]["fit"].p))),
                 "fit_err": dict(zip(BP[k]["fit"].names, map(float, BP[k]["fit"].perr_scaled))),
                 "chi2_red": BP[k]["fit"].chi2_red, "dof": BP[k]["fit"].dof,
                 "der": BP[k]["der"], "free": BP[k]["free"], "band_at_level": BAND[k]} for k in BP}
RES["bp"]["c_change"] = (c_change, s_c_change)
RES["bp"]["LB"] = {k: v for k, v in LB.items() if not isinstance(v, np.ndarray)}
RES["bp"]["level"] = LEVEL

# ------------------------------------------------------------------------------------------------
# Step 5: phase points
# ------------------------------------------------------------------------------------------------
W_PAIRS = 1000   # about 82 ms per window: the pair rate is steady inside a window but shifts within a burst


def windowed_H(d, f_nom, harmonics=(1,)):
    """Transfer function from short windows. Each window gets its own fitted frequency (in cycles per pair),
    so a shift of the pair rate within the burst does not smear the phasors. Both channels share the
    window's time base, and H_w = Z_out / Z_in is averaged over the windows."""
    fs_f, _ = file_rate(d, f_nom)
    dt_f = delay_for(fs_f)
    k = np.asarray(harmonics)
    Hs, ZA, ZB = [], [], []
    nwin = len(d) // W_PAIRS
    for i in range(nwin):
        seg = d[i * W_PAIRS:(i + 1) * W_PAIRS]
        n = np.arange(len(seg), dtype=float)
        fc, _ = rc.fit_frequency(seg[:, 0], n, f_nom / fs_f)
        pa, pb = rc.phasors(seg[:, 0], n, k * fc), rc.phasors(seg[:, 1], n, k * fc)
        Hs.append(pb["Z"] / pa["Z"])
        ZA.append(pa["Z"])
        ZB.append(pb["Z"])
    Hs = np.array(Hs) / GAIN * np.exp(-1j * 2 * np.pi * k * f_nom * dt_f)[None, :]
    return {"H_w": Hs, "fs": fs_f, "dt": dt_f, "ZA": np.array(ZA), "ZB": np.array(ZB), "nwin": nwin}


def summarize_H(Hw):
    """Mean transfer function and standard errors (amplitude, phase) from the scatter of the windows."""
    n = Hw.shape[0]
    Hm = Hw.mean(0)
    amp_w, ph_w = np.abs(Hw), np.angle(Hw / Hm[None, :])
    amp = np.abs(Hm)
    s_amp = amp_w.std(0, ddof=1) / np.sqrt(n)
    s_ph = ph_w.std(0, ddof=1) / np.sqrt(n)
    return Hm, amp, s_amp, s_ph


def phase_set(key):
    rows = []
    for p in sorted((D / "phase").glob(f"{key}_*Hz.csv"), key=rc.freq_from_name):
        f_nom = rc.freq_from_name(p)
        r = windowed_H(rc.load_recording(p), f_nom)
        Hm, amp, s_amp, s_ph = summarize_H(r["H_w"])
        s_ph_tot = np.sqrt(s_ph[0] ** 2 + (2 * np.pi * f_nom * s_dt) ** 2)
        s_amp_tot = np.sqrt(s_amp[0] ** 2 + (amp[0] * S_GAIN / GAIN) ** 2)
        rows.append({"f": f_nom, "fs": r["fs"], "dt_us": r["dt"] * 1e6, "phase_deg": float(np.degrees(np.angle(Hm[0]))),
                     "s_phase_deg": float(np.degrees(s_ph_tot)), "amp": float(amp[0]), "s_amp": float(s_amp_tot),
                     "in_amp": float(np.abs(r["ZA"][:, 0]).mean()), "out_amp": float(np.abs(r["ZB"][:, 0]).mean()),
                     "nwin": r["nwin"]})
    return pd.DataFrame(rows)


PH = {k: phase_set(k) for k in ("lp1", "bpL", "bpB")}
fitp1 = rc.fit_lp1_phase(PH["lp1"])
taup, s_taup = fitp1.p[0], fitp1.perr_scaled[0]
fcp, s_fcp = 1 / (2 * np.pi * taup), s_taup / (2 * np.pi * taup**2)
Rg_m, s_Rg_m = rc.generator_resistance(tau1, s_tau1, taup, s_taup, parts["R1"]["value"], std_unc("R1"))
PMK = {}
for key in ("bpL", "bpB"):
    fit = rc.fit_bp_phase(PH[key])
    marks = rc.bp_phase_marks(fit, PHASE_F)
    PMK[key] = {"fit": fit, "marks": {n: (v, e * scaled(fit)) for n, (v, e) in marks.items()}}
# amplitude fit of the single stage from the phase files (terminal view), a further cross-check
fit_amp1 = rc.fit_scaled(lambda f, tau: np.abs(rc.H_lp1(f, tau)), PH["lp1"].f.to_numpy(), PH["lp1"].amp.to_numpy(),
                         PH["lp1"].s_amp.to_numpy(), [1e-4], [1e-4], ["tau"], bounds=([1e-7], [1e-1]))
RES["phase"] = {"lp1": {"tau_us": taup * 1e6, "s_tau_us": s_taup * 1e6, "fc": fcp, "s_fc": s_fcp,
                        "chi2_red": fitp1.chi2_red, "amp_fc": 1 / (2 * np.pi * fit_amp1.p[0]),
                        "amp_s_fc": fit_amp1.perr_scaled[0] / (2 * np.pi * fit_amp1.p[0] ** 2), "amp_chi2_red": fit_amp1.chi2_red},
                "Rg": (Rg_m, s_Rg_m),
                "bp": {k: {"b": PMK[k]["fit"].p[0], "c": PMK[k]["fit"].p[1], "chi2_red": PMK[k]["fit"].chi2_red,
                           "marks": PMK[k]["marks"]} for k in PMK},
                "tables": {k: PH[k].to_dict("list") for k in PH}}

# ------------------------------------------------------------------------------------------------
# Step 6: square-wave bursts
# ------------------------------------------------------------------------------------------------
HARM = (1, 3, 5, 7)


def square_burst(d):
    fs_f, _ = file_rate(d, 500.0)
    kmax = int(np.floor((0.5 * fs_f - 1.0) / 500.0))
    k = np.arange(1, kmax + 1)
    r = windowed_H(d, 500.0, harmonics=k)
    Hm = r["H_w"].mean(0)
    # first window, for the waveform comparison
    seg = d[:W_PAIRS]
    n = np.arange(len(seg), dtype=float)
    fc, _ = rc.fit_frequency(seg[:, 0], n, 500.0 / fs_f)
    pa, pb = rc.phasors(seg[:, 0], n, k * fc), rc.phasors(seg[:, 1], n, k * fc)
    fs_w = 500.0 / fc
    return {"f0": 500.0, "k": k, "H": Hm, "H_w": r["H_w"], "ZA": pa["Z"], "dcA": pa["dc"], "dcB": pb["dc"],
            "t": n / fs_w, "xA": seg[:, 0].astype(float), "xB": seg[:, 1].astype(float), "fs": fs_f, "dt": r["dt"],
            "gain_ratio": GAIN}


def square_summary(bursts):
    recs = []
    for bu in bursts:
        rec = {}
        Hs = {kk: bu["H"][kk - 1] for kk in HARM}
        for kk in HARM:
            fk = kk * 500
            rec[f"H{fk}"] = abs(Hs[kk])
            rec[f"phase{fk}"] = np.degrees(np.angle(Hs[kk]))
            if kk != 1:
                rec[f"norm{fk}"] = abs(Hs[kk]) / abs(Hs[1])
        ph = [np.degrees(np.angle(Hs[kk])) for kk in HARM]
        rec["spread_deg"] = max(ph) - min(ph)
        recs.append(rec)
    df = pd.DataFrame(recs)
    return pd.DataFrame({"mean": df.mean(), "sem": df.std(ddof=1) / np.sqrt(len(df))}), df


def predict_square_output(bu, K, b, c):
    fk = bu["k"] * bu["f0"]
    Zp = bu["ZA"] * rc.H_bp(fk, K, b, c) * bu["gain_ratio"]
    tB = bu["t"] + bu["dt"]
    y = bu["dcB"] + np.real(np.exp(1j * 2 * np.pi * np.outer(tB, fk)) @ Zp)
    resid = bu["xB"] - y
    rms = float(np.sqrt(np.mean(resid**2)))
    return y, rms, rms / float(np.ptp(bu["xB"]))


SQ = {k: [square_burst(rc.load_recording(p)) for p in sorted((D / "square").glob(f"{k}_*.csv"))] for k in ("bpL", "bpB")}
SQS, SQRAW = {}, {}
for k in SQ:
    SQS[k], SQRAW[k] = square_summary(SQ[k])
SQP = {}
for k in SQ:
    Kt, bt, ct = (PM_[f"{k}.{n}"] for n in ("Kt", "bt", "ct"))
    SQP[k] = [predict_square_output(bu, Kt, bt, ct) for bu in SQ[k]]
RES["square"] = {k: {"summary": SQS[k].to_dict(), "pred_resid_pct": [100 * r[2] for r in SQP[k]],
                     "fs": [bu["fs"] for bu in SQ[k]], "kmax": int(SQ[k][0]["k"][-1])} for k in SQ}

# ------------------------------------------------------------------------------------------------
# Step 7: comparison table
# ------------------------------------------------------------------------------------------------
rows = []


def add(group, name, meas, s_meas, key=None, pred=None, s_pred=None, unit="", scale=1.0, nd=None):
    if key is not None:
        pred, s_pred = PM_[key], PS_[key]
    z = rc.zscore(meas, s_meas, pred, s_pred)
    rows.append({"group": group, "quantity": name, "unit": unit, "measured": meas * scale, "s_meas": s_meas * scale,
                 "predicted": pred * scale, "s_pred": s_pred * scale, "z": float(z), "within_2sigma": bool(abs(z) < 2)})


add("Single stage", "f_c from sweep", fc1, s_fc1, key="lp1.fc_sweep", unit="Hz")
add("Single stage", "f_c from phase", fcp, s_fcp, key="lp1.fc_phase", unit="Hz")
add("Single stage", "generator resistance R_g", Rg_m, s_Rg_m, pred=50.0, s_pred=std_unc("Rg"), unit="Ω")
add("Two stage", "b vs loaded model", fit2.p[0], fit2.perr_scaled[0], key="lp2.b_loaded", unit="10⁻⁴ s", scale=1e4)
add("Two stage", "b vs ideal model", fit2.p[0], fit2.perr_scaled[0], key="lp2.b_ideal", unit="10⁻⁴ s", scale=1e4)
add("Two stage", "c", fit2.p[1], fit2.perr_scaled[1], key="lp2.c", unit="10⁻⁸ s²", scale=1e8)
add("Two stage", "half-power frequency", fh2, s_fh2, key="lp2.fhalf_loaded", unit="Hz")
for key, form in (("bpL", "Loaded"), ("bpB", "Buffered")):
    fit, der, free = BP[key]["fit"], BP[key]["der"], BP[key]["free"]
    g = f"{form} band-pass"
    add(g, "b", fit.p[1], fit.perr_scaled[1], key=f"{key}.b", unit="10⁻⁴ s", scale=1e4)
    add(g, "c", fit.p[2], fit.perr_scaled[2], key=f"{key}.c", unit="10⁻⁸ s²", scale=1e8)
    for n, lab, u in (("f0", "center frequency f₀", "Hz"), ("df", "bandwidth Δf", "Hz"), ("Q", "Q", ""),
                      ("peak_power", "peak power ratio", ""), ("product", "(K/b)·Δf", "Hz"),
                      ("f_lo", "low edge", "Hz"), ("f_hi", "high edge", "Hz")):
        add(g, lab, *der[n], key=f"{key}.{n}", unit=u)
    for n, lab in (("f_lo", "low edge, model-free"), ("f_hi", "high edge, model-free"), ("df", "bandwidth, model-free")):
        add(g, lab, *free[n], key=f"{key}.{n}", unit="Hz")
    marks = PMK[key]["marks"]
    add(g, "phase 0° crossing", *marks["f_zero"], key=f"{key}.ph_f0", unit="Hz")
    add(g, "+45° frequency", *marks["f_p45"], key=f"{key}.ph_p45", unit="Hz")
    add(g, "−45° frequency", *marks["f_m45"], key=f"{key}.ph_m45", unit="Hz")
    add(g, "phase at 2 kHz", *marks["phase_2000Hz"], key=f"{key}.ph_2000Hz", unit="°")
    S = SQS[key]
    for f in (500, 1500, 2500, 3500):
        add(g, f"|H| at {f} Hz", S.loc[f"H{f}", "mean"], S.loc[f"H{f}", "sem"], key=f"{key}.H{f}")
    for f in (1500, 2500, 3500):
        add(g, f"|H_{f}| / |H_500|", S.loc[f"norm{f}", "mean"], S.loc[f"norm{f}", "sem"], key=f"{key}.norm{f}")
    add(g, "phase spread of 4 harmonics", S.loc["spread_deg", "mean"], S.loc["spread_deg", "sem"], key=f"{key}.spread_deg", unit="°")
add("Loading effect", "change in c", c_change, s_c_change, key="bp.c_change_pct", unit="%")
add("Loading effect", f"P_loaded / P_buffered at {fl:.0f} Hz", LB["q_lowest"], LB["s_q_lowest"], pred=LB["q_pred_lowest"],
    s_pred=LB["s_q_pred_lowest"])
hL = PH["bpL"].set_index("f"); hB = PH["bpB"].set_index("f")
r100 = hL.loc[100.0, "amp"] / hB.loc[100.0, "amp"]
s_r100 = r100 * np.hypot(hL.loc[100.0, "s_amp"] / hL.loc[100.0, "amp"], hB.loc[100.0, "s_amp"] / hB.loc[100.0, "amp"])
pr = np.abs(rc.H_bp(100.0, mc["bpL.Kt"], mc["bpL.bt"], mc["bpL.ct"])) / np.abs(rc.H_bp(100.0, mc["bpB.Kt"], mc["bpB.bt"], mc["bpB.ct"]))
add("Loading effect", "|H_loaded| / |H_buffered| at 100 Hz (two-channel)", r100, s_r100, pred=float(np.mean(pr)), s_pred=float(np.std(pr, ddof=1)))
TABLE = pd.DataFrame(rows)
TABLE.to_csv(OUT / "comparison_table.csv", index=False)
RES["table_summary"] = {"n": len(TABLE), "n_within": int(TABLE.within_2sigma.sum())}
RES["predictions_nominal"] = {k: float(v) for k, v in nominal.items()}
RES["pred_mc"] = {k: (PM_[k], PS_[k]) for k in PM_}


def tojson(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


with open(OUT / "results.json", "w") as fh:
    json.dump(RES, fh, default=tojson, indent=1)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(json.dumps({k: RES[k] for k in ("calib", "noise", "sweep_cal", "gen_amp")}, default=tojson, indent=1)); print({k: v for k, v in RES["delay"].items() if k != "table"})
    print(json.dumps(RES["triangle"], default=tojson, indent=1))
    print({k: (v["fs"], v["noise_var"], v["n_complete"]) for k, v in RES["sweeps"]["info"].items()})
    print(json.dumps(RES["lp"], default=tojson, indent=1))
    for k in ("bpL", "bpB"):
        print(k, BP[k]["fit"])
    print(json.dumps({k: RES["bp"][k] for k in ("c_change", "LB", "level")}, default=tojson, indent=1))
    for k in PH:
        print(k); print(PH[k].round(4).to_string())
    print(json.dumps({k: v for k, v in RES["phase"].items() if k != "tables"}, default=tojson, indent=1))
    for k in SQS:
        print(k); print(SQS[k].round(4).to_string()); print(RES["square"][k]["pred_resid_pct"], RES["square"][k]["kmax"])
    print(TABLE.round(4).to_string())
    print(RES["table_summary"])
