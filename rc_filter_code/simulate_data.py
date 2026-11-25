"""simulate_data.py: make a complete fake data set in the same format as daq.py.

Use it to run the whole notebook before any real recording exists, and to confirm that the
analysis recovers known parameters. The circuits are simulated from their state equations
(including the generator's 50 ohm resistance), the ESP32 is modelled with an ADC offset, gain,
slight nonlinearity, noise, quantization, a sample-clock error, and a delay between the two
channels. The square-wave bursts are exact time-domain solutions, so aliasing is realistic.

    python simulate_data.py                 # SerialPlot ASCII stream rate (15.6 kS/s)
    python simulate_data.py --binary        # daq.py binary stream rate (40 kS/s)
    python simulate_data.py --out data_sim --seed 7
"""
import argparse
import json
import os

import numpy as np

TWO_PI = 2 * np.pi
F1, F2, T_SWEEP = 20.0, 10000.0, 20.0
PHASE_FREQS = [100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000]


# ---------------------------------------------------------------- circuits as state equations
def state_space(kind, p, Rg):
    """Return A, B, C, D with input u = generator EMF and outputs [V_terminal, V_out]."""
    if kind == "lp1":
        Ra = p["R1"] + Rg
        A = np.array([[-1 / (Ra * p["C1"])]])
        B = np.array([[1 / (Ra * p["C1"])]])
        C = np.array([[Rg / Ra], [1.0]])
        D = np.array([[p["R1"] / Ra], [0.0]])
    elif kind == "lp2":
        Ra, R2, C1, C2 = p["R1"] + Rg, p["R2"], p["C1"], p["C2"]
        A = np.array([[-1 / (Ra * C1) - 1 / (R2 * C1), 1 / (R2 * C1)], [1 / (R2 * C2), -1 / (R2 * C2)]])
        B = np.array([[1 / (Ra * C1)], [0.0]])
        C = np.array([[Rg / Ra, 0.0], [0.0, 1.0]])
        D = np.array([[p["R1"] / Ra], [0.0]])
    elif kind in ("bpL", "bpB"):
        R3, C3, C4 = p["R3"], p["C3"], p["C4"]
        RB = p["R4"] * p["R5"] / (p["R4"] + p["R5"])
        # states: v3 (across C3, terminal side minus node A), v4 (output)
        if kind == "bpL":
            G = 1 / RB + 1 / R3
            den = 1 + G * Rg
            # i3 = [G (u - v3) - v4/R3] / den ; VA = u - Rg i3 - v3
            i3 = np.array([-G / den, -1 / (R3 * den)])
            i3u = G / den
        else:
            # i3 = (u - v3) / (RB + Rg)
            i3 = np.array([-1 / (RB + Rg), 0.0])
            i3u = 1 / (RB + Rg)
        VA = np.array([-1.0, 0.0]) - Rg * i3
        VAu = 1.0 - Rg * i3u
        A = np.vstack([i3 / C3, (VA - np.array([0.0, 1.0])) / (R3 * C4)])
        B = np.array([[i3u / C3], [VAu / (R3 * C4)]])
        C = np.array([-Rg * i3, [0.0, 1.0]])
        D = np.array([[1 - Rg * i3u], [0.0]])
    else:
        raise ValueError(kind)
    return A, B, C, D


def freq_response(kind, p, Rg, f):
    """Phasors of [V_terminal, V_out] per unit EMF."""
    A, B, C, D = state_space(kind, p, Rg)
    out = np.empty((2, len(f)), complex)
    I = np.eye(A.shape[0])
    for i, fi in enumerate(f):
        x = np.linalg.solve(1j * TWO_PI * fi * I - A, B)
        out[:, i] = (C @ x + D)[:, 0]
    return out


def square_response(kind, p, Rg, t, f0, amp):
    """Exact periodic steady state of [V_terminal, V_out] for a square EMF of +/- amp (AC part)."""
    A, B, C, D = state_space(kind, p, Rg)
    lam, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    Tp = 1.0 / f0
    h = Tp / 2
    xs_p = -np.linalg.solve(A, B[:, 0] * amp)
    xs_m = -xs_p
    E = (V @ np.diag(np.exp(lam * h)) @ Vi).real
    I = np.eye(A.shape[0])
    rhs = xs_m + E @ xs_p - E @ E @ xs_p - E @ xs_m
    x0 = np.linalg.solve(I - E @ E, rhs)
    xh = xs_p + E @ (x0 - xs_p)
    tau = np.mod(t, Tp)
    first = tau < h
    out = np.empty((2, len(t)))
    for sel, xstart, xss, tt, u in ((first, x0, xs_p, tau, amp), (~first, xh, xs_m, tau - h, -amp)):
        d = Vi @ (xstart - xss)
        X = (V @ (d[:, None] * np.exp(np.outer(lam, tt[sel])))).real + xss[:, None]
        out[:, sel] = C @ X + D * u
    return out


# ---------------------------------------------------------------- ESP32 model
class ESP32:
    def __init__(self, rng, fs_stream, pair_us=49, dt=11.7e-6):
        self.rng = rng
        clock = 1 + 23e-6  # crystal error: the ESP32 runs slightly slow, so true rates are lower
        self.fs_stream = fs_stream / clock
        self.fs_pair = 1e6 / pair_us / clock
        self.dt = dt
        self.gain = {34: 1770.0 * 1.004, 35: 1770.0}
        self.offset = {34: -205.0, 35: -212.0}

    def adc(self, v, pin):
        c = self.offset[pin] + self.gain[pin] * v + 3.0 * (v - 1.65) ** 2
        c = c + self.rng.normal(0, 3.0, np.shape(v))
        return np.clip(np.round(c), 0, 4095).astype(int)


def write(path, meta, header, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.column_stack(cols)
    with open(path, "w") as fh:
        for k, v in meta.items():
            fh.write(f"# {k}={v}\n")
        fh.write(header + "\n")
        np.savetxt(fh, data, fmt="%d", delimiter=",")


def sweep_signal(t, offset, amp, Hfun):
    tau = np.mod(t - offset, T_SWEEP)
    k = F2 / F1
    f = F1 * k ** (tau / T_SWEEP)
    theta = TWO_PI * F1 * T_SWEEP / np.log(k) * (k ** (tau / T_SWEEP) - 1)
    H = Hfun(f)
    return amp * np.abs(H) * np.sin(theta + np.angle(H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_sim")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--sweep-seconds", type=float, default=85.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fs_nom = 40000.0 if a.binary else 15625.0
    esp = ESP32(rng, fs_nom)
    out = a.out

    measured = {"R1": 987.0, "R2": 987.0, "R3": 987.0, "R4": 9970.0, "R5": 9970.0,
                "C1": 99.7e-9, "C2": 99.7e-9, "C3": 99.7e-9, "C4": 99.7e-9}
    true = {k: v * (1 + rng.normal(0, 0.003)) for k, v in measured.items()}
    Rg = 50.0
    vbias = 3.3 * true["R5"] / (true["R4"] + true["R5"])
    with open(os.path.join(_mk(out), "truth.json"), "w") as fh:
        json.dump({"true": true, "Rg": Rg, "dt": esp.dt, "fs_stream": esp.fs_stream,
                   "fs_pair": esp.fs_pair}, fh, indent=1)

    # ---- calibration
    def burst(name, VA, VB, extra=None):
        meta = {"mode": "burst", "pairs": len(VA), "period_us": 49}
        meta.update(extra or {})
        write(os.path.join(out, name), meta, "ch34,ch35", [esp.adc(VA, 34), esp.adc(VB, 35)])

    n = int(25 * esp.fs_stream)
    t = np.arange(n) / esp.fs_stream
    v = 1.65 + 0.75 * np.sin(TWO_PI * 1500 * t + 0.3)
    write(os.path.join(out, "calib", "stream_1500Hz.csv"), {"mode": "stream"}, "ch35", [esp.adc(v, 35)])
    write(os.path.join(out, "calib", "noise_dc.csv"), {"mode": "stream"}, "ch35",
          [esp.adc(np.full(int(3 * esp.fs_stream), 1.65), 35)])

    tp = np.arange(16000) / esp.fs_pair
    for f in [1500] + [200, 1000, 3000]:
        ph = rng.uniform(0, TWO_PI)
        VA = 1.65 + 0.75 * np.sin(TWO_PI * f * tp + ph)
        VB = 1.65 + 0.75 * np.sin(TWO_PI * f * (tp + esp.dt) + ph)
        burst(os.path.join("calib", f"burst_{f}Hz.csv" if f == 1500 else f"delay_{f}Hz.csv"), VA, VB)

    ttri = np.arange(25000) * 200e-6 * (1 + 23e-6)
    tri = lambda tt: 1.65 + 0.75 * (2 / np.pi) * np.arcsin(np.sin(TWO_PI * 1.0 * tt + 0.4))
    write(os.path.join(out, "calib", "triangle_1Hz.csv"), {"mode": "burst", "period_us": 200}, "ch34,ch35",
          [esp.adc(tri(ttri), 34), esp.adc(tri(ttri + esp.dt), 35)])

    # ---- sweeps (EMF reference and outputs), one long recording each with several sweeps
    ns = int(a.sweep_seconds * esp.fs_stream)
    ts = np.arange(ns) / esp.fs_stream
    configs = {"ref": None, "lp1": "lp1", "lp2": "lp2", "bpL": "bpL", "bpB": "bpB"}
    for name, kind in configs.items():
        off = rng.uniform(0, T_SWEEP)
        if kind is None:
            Hf = lambda f: np.ones_like(f, complex)
            dc = 1.65
        else:
            fgrid = np.geomspace(F1, F2, 4000)
            Hg = freq_response(kind, true, Rg, fgrid)[1]
            Hf = lambda f, Hg=Hg, fgrid=fgrid: (np.interp(np.log(f), np.log(fgrid), Hg.real)
                                               + 1j * np.interp(np.log(f), np.log(fgrid), Hg.imag))
            dc = vbias if kind.startswith("bp") else 1.65
        v = dc + sweep_signal(ts, off, 0.75, Hf)
        write(os.path.join(out, "sweeps", f"{name}.csv"), {"mode": "stream", "true_offset_s": f"{off:.4f}"},
              "ch35", [esp.adc(v, 35)])

    # ---- phase points
    for kind in ("lp1", "bpL", "bpB"):
        dc_out = vbias if kind.startswith("bp") else 1.65
        R = freq_response(kind, true, Rg, np.array(PHASE_FREQS, float))
        for i, f in enumerate(PHASE_FREQS):
            ph = rng.uniform(0, TWO_PI)
            VA = 1.65 + 0.75 * np.abs(R[0, i]) * np.sin(TWO_PI * f * tp + ph + np.angle(R[0, i]))
            VB = dc_out + 0.75 * np.abs(R[1, i]) * np.sin(TWO_PI * f * (tp + esp.dt) + ph + np.angle(R[1, i]))
            burst(os.path.join("phase", f"{kind}_{f}Hz.csv"), VA, VB)

    # ---- square-wave bursts (exact time-domain response)
    for kind in ("bpL", "bpB"):
        for r in range(1, 4):
            t0 = rng.uniform(0, 1e-3)
            yA = square_response(kind, true, Rg, tp + t0, 500.0, 0.625)
            yB = square_response(kind, true, Rg, tp + t0 + esp.dt, 500.0, 0.625)
            burst(os.path.join("square", f"{kind}_{r}.csv"), 1.65 + yA[0], vbias + yB[1])
    print(f"simulated data written to {out}/ (stream rate {esp.fs_stream:.1f} S/s)")


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    main()
