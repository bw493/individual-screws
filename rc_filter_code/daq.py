"""daq.py: record data from rc_daq.ino straight to CSV files (an alternative to SerialPlot).

Requires pyserial (pip install pyserial). Find the port in Arduino IDE (Tools > Port),
for example COM5 on Windows or /dev/cu.usbserial-0001 on macOS.

Examples
--------
Single recordings:
    python daq.py --port COM5 info
    python daq.py --port COM5 stream --seconds 85 --binary --out data/sweeps/ref.csv
    python daq.py --port COM5 burst --out data/calib/burst_1500Hz.csv
    python daq.py --port COM5 burst --pairs 25000 --period 200 --out data/calib/triangle_1Hz.csv

Guided sets (the script tells you what to set on the generator, then records):
    python daq.py --port COM5 delay-set             # 200, 1000, 3000 Hz with both pins on the generator
    python daq.py --port COM5 phase-set --name lp1  # 9 phase frequencies, saved as data/phase/lp1_<f>Hz.csv
    python daq.py --port COM5 square-set --name bpL # three 500 Hz square-wave bursts

Every file starts with '# key=value' metadata lines, then a header line, then one row per sample.
The analysis notebook reads these files and SerialPlot recordings in the same way.
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover
    sys.exit("pyserial is missing: pip install pyserial")

BAUD = 921600
PHASE_FREQS = [100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000]
DELAY_FREQS = [200, 1000, 3000]


def open_port(port):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUD
    ser.timeout = 0.2
    ser.dtr = False  # leave the ESP32 running; do not hold it in reset
    ser.rts = False
    ser.open()
    time.sleep(1.0)  # some boards reset when the port opens; let them boot
    stop(ser)
    return ser


def stop(ser):
    ser.write(b"x\n")
    time.sleep(0.3)
    ser.reset_input_buffer()


def read_until(ser, marker, timeout):
    buf = bytearray()
    t_end = time.time() + timeout
    while time.time() < t_end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if marker in buf:
                return bytes(buf)
    raise TimeoutError(f"did not see {marker!r} within {timeout} s")


def command(ser, text, marker=b"\n", timeout=3.0):
    ser.reset_input_buffer()
    ser.write(text.encode() + b"\n")
    return read_until(ser, marker, timeout).decode(errors="replace")


def decode_binary(raw):
    """Decode the self-synchronizing 2-byte stream: high byte 1hhhhhhh, low byte 0lllllll."""
    a = np.frombuffer(raw, dtype=np.uint8)
    hi = np.flatnonzero(((a[:-1] & 0x80) != 0) & ((a[1:] & 0x80) == 0))
    return ((a[hi].astype(np.uint16) & 0x1F) << 7) | a[hi + 1].astype(np.uint16)


def write_csv(path, meta, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        for k, v in meta.items():
            fh.write(f"# {k}={v}\n")
        fh.write(header + "\n")
        if rows.ndim == 1:
            np.savetxt(fh, rows, fmt="%d")
        else:
            np.savetxt(fh, rows, fmt="%d", delimiter=",")
    n = rows.shape[0]
    print(f"saved {path} ({n} samples)")


def record_stream(ser, seconds, out, pin=35, binary=False, period=None):
    command(ser, "f b" if binary else "f a", b"stream_format")
    if period:
        command(ser, f"p {period}", b"stream_period_us")
    info = command(ser, "i", b"analogRead_us")
    ser.reset_input_buffer()
    ser.write(f"s {pin}\n".encode())
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < seconds:
        buf += ser.read(ser.in_waiting or 1)
    ser.write(b"x\n")
    buf += read_until(ser, b"late=", 5.0)
    buf += ser.read(200)
    marker = buf.rfind(b"\n# stream stopped")
    tail = buf[marker:].decode(errors="replace")
    body = bytes(buf[:marker])
    if binary:
        values = decode_binary(body)
    else:
        lines = body.decode(errors="replace").splitlines()
        values = np.array([int(s) for s in lines if s.strip().isdigit()], dtype=np.int64)
    meta = {"mode": "stream", "pin": pin, "format": "binary" if binary else "ascii",
            "wall_seconds": f"{seconds:.1f}"}
    for tok in (info + " " + tail).replace("\n", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            meta.setdefault(k, v)
    if "late" in meta and meta["late"] != "0":
        print(f"warning: {meta['late']} samples were late; consider a longer period (p <us>)")
    write_csv(out, meta, f"ch{pin}", values)
    return values


def record_burst(ser, out, pairs=16000, period=49):
    text = command(ser, f"b {pairs} {period}", b"# end", timeout=30.0)
    meta = {"mode": "burst"}
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        elif line and line[0].isdigit():
            parts = line.split(",")
            if len(parts) == 2:
                rows.append((int(parts[0]), int(parts[1])))
    rows = np.array(rows, dtype=np.int64)
    if meta.get("late", "0") != "0":
        print(f"warning: {meta['late']} pairs were late; use a longer period")
    write_csv(out, meta, "ch34,ch35", rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--data", default="data", help="data folder for the guided sets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    s = sub.add_parser("stream")
    s.add_argument("--seconds", type=float, default=85.0)
    s.add_argument("--pin", type=int, default=35)
    s.add_argument("--binary", action="store_true", help="40 kS/s binary stream (full 10 kHz sweep)")
    s.add_argument("--period", type=int, default=None)
    s.add_argument("--out", required=True)
    b = sub.add_parser("burst")
    b.add_argument("--pairs", type=int, default=16000)
    b.add_argument("--period", type=int, default=49)
    b.add_argument("--out", required=True)
    sub.add_parser("delay-set")
    p = sub.add_parser("phase-set")
    p.add_argument("--name", required=True, help="lp1, bpL or bpB")
    q = sub.add_parser("square-set")
    q.add_argument("--name", required=True, help="bpL or bpB")
    q.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    ser = open_port(args.port)
    try:
        if args.cmd == "info":
            print(command(ser, "i", b"analogRead_us"))
        elif args.cmd == "stream":
            record_stream(ser, args.seconds, args.out, args.pin, args.binary, args.period)
        elif args.cmd == "burst":
            record_burst(ser, args.out, args.pairs, args.period)
        elif args.cmd == "delay-set":
            for f in DELAY_FREQS:
                input(f"Both pins on the generator. Set a {f} Hz sine, 1.5 Vpp, 1.65 V offset, then press Enter.")
                record_burst(ser, os.path.join(args.data, "calib", f"delay_{f}Hz.csv"))
        elif args.cmd == "phase-set":
            for f in PHASE_FREQS:
                input(f"[{args.name}] Set a {f} Hz sine, 1.5 Vpp, 1.65 V offset, then press Enter.")
                record_burst(ser, os.path.join(args.data, "phase", f"{args.name}_{f}Hz.csv"))
        elif args.cmd == "square-set":
            input(f"[{args.name}] Set a 500 Hz square wave, 1.25 Vpp, 1.65 V offset, then press Enter.")
            for i in range(1, args.repeats + 1):
                record_burst(ser, os.path.join(args.data, "square", f"{args.name}_{i}.csv"))
                time.sleep(0.5)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
