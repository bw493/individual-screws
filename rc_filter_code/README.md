# Cascaded RC filter project: code

| File | Purpose |
|---|---|
| `rc_daq/rc_daq.ino` | ESP32 sketch: timer-paced single-pin stream for sweeps and two-pin bursts for everything else |
| `daq.py` | Records from the sketch straight to CSV, including guided sets for the delay, phase and square-wave steps |
| `rcfilter.py` | Analysis library: models, Monte Carlo predictions, calibration, sweeps, fits, phase, harmonics, z-scores |
| `analysis.py` | Final analysis of the recorded data in `data/`, with per-file sampling rates, linear sweeps and windowed bursts. It reproduces every number in the report |

## Setup

1. Arduino IDE with the ESP32 board package (tested on cores 2.0.17 and 3.3.12). Board: "ESP32 Dev Module". Upload `rc_daq/rc_daq.ino`.
2. Python packages, in a Python 3 virtual environment: `python3 -m venv .venv`, then `source .venv/bin/activate`, then `python -m pip install numpy scipy pandas pyserial`.
3. Final results from the recorded data: `python analysis.py`. It should end with `{'n': 58, 'n_within': 42}` and writes `out/comparison_table.csv` and `out/results.json`.

## Wiring

GPIO 34 reads the input terminal (the generator side of R1 or C3) and GPIO 35 reads Vout. Circuits 3 and 4 also need the GPIO 34 connection at the generator side of C3 for the phase points and square-wave bursts. For the reference sweeps, connect GPIO 35 directly to the generator so the reference and the outputs use the same ADC channel. Keep the generator in High-Z mode.

## Recording (replace COM5 with your port)

| Proposal step | Command | Saved as |
|---|---|---|
| 2, stream sampling rate | `python daq.py --port COM5 stream --seconds 25 --out data/calib/stream_1500Hz.csv` (add `--binary` if you use binary sweeps) | `calib/stream_1500Hz.csv` |
| 2, two-channel rate | `python daq.py --port COM5 burst --out data/calib/burst_1500Hz.csv` | `calib/burst_1500Hz.csv` |
| 2, delay Δt | `python daq.py --port COM5 delay-set` | `calib/delay_<f>Hz.csv` |
| 2, ADC linearity | `python daq.py --port COM5 burst --pairs 25000 --period 200 --out data/calib/triangle_1Hz.csv` | `calib/triangle_1Hz.csv` |
| 2, ADC noise (optional) | generator at 1.65 V DC, `stream --seconds 3 --out data/calib/noise_dc.csv` | `calib/noise_dc.csv` |
| 3, sweeps | `python daq.py --port COM5 stream --seconds 85 --binary --out data/sweeps/ref.csv` and the same for `lp1`, `lp2`, `bpL`, `bpB` | `sweeps/<name>.csv` |
| 4, phase points | `python daq.py --port COM5 phase-set --name lp1` (then `bpL`, `bpB`) | `phase/<name>_<f>Hz.csv` |
| 5, square waves | `python daq.py --port COM5 square-set --name bpL` (then `bpB`) | `square/<name>_<n>.csv` |

An 85 s recording always contains at least three complete 20 s sweeps, and the analysis finds and uses every complete sweep in each file. Enter each part's own multimeter reading and the meter's stated accuracy in the `parts` dictionary near the top of `analysis.py`, then run `python analysis.py`. The script expects linear sweeps and calibrates every file's sampling rate from its known signal frequency, as described in the main README's "Notes on the data", so new recordings made with different settings need matching changes.

### Using SerialPlot instead

Port settings: 921600 baud, ASCII, comma delimiter. The board starts streaming GPIO 35 at power-up (1 channel). For a burst, set 2 channels and send `b` followed by a newline from the Commands panel. Save each recording with the file names above; `rc.load_recording` reads SerialPlot CSV files the same way (use `columns=[...]` if you enable timestamps).

## Key figures

The planned figures, drawn from simulated data before recording, are in `example success outcome/` at the top of the repository. The report contains the figures drawn from the recorded data.

* **Figure A** shows the phenomenon. Its left panel normalizes each band-pass curve to its own peak with the half-power edges marked. Its right panel compares both forms at one fixed absolute level.
* **Figure B** shows the advantage of each form. Its left panel gives the square-wave harmonics normalized to the fundamental. Its right panel gives their absolute transmission.

## Design notes

* **Stream rates.** ASCII streaming fits 15.6 kS/s through the 921600 baud link, so sweep data above 0.4 f_s (about 6.2 kHz) are excluded from the fits (no anti-alias filter on the ESP32). The binary stream through `daq.py` runs at 40 kS/s and covers the full 20 Hz to 10 kHz sweep.
* **Burst period of 49 µs.** At 50 µs per pair (20 kS/s), the aliases of the square wave's high harmonics land exactly on 500, 1500, 2500 and 3500 Hz and bias the harmonic ratios by up to 3 %. At 49 µs they fall between harmonics and contribute under 0.05 %.
* **Two-stage model comparison.** With $b$ and $c$ both free, the loaded and ideal models have the same functional form and the same $c$, so their free fits are identical. The analysis therefore tests the fitted $b$ against each prediction and reports each model's reduced $\chi^2$ at its predicted parameters.
* **Generator resistance in the band-pass.** Including $R_g$ in series with C3 keeps the response second order with $K=R_BC_3$, $b=(R_B+R_g)C_3+R_3C_4+R_BC_4$ and $c=(R_B+R_g)C_3R_3C_4+R_gR_BC_3C_4$ for the loaded form. These formulas were checked against a direct circuit solution to machine precision.
* **Uncertainties.** Multimeter accuracy is treated as a rectangular bound by default (`BOUND_IS = "one_sigma"` treats it as one standard deviation). Fit errors are inflated by $\sqrt{\chi^2_\text{red}}$ when the reduced $\chi^2$ exceeds 1.
