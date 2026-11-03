"""First-formant (F1) shift DSP, extracted from experiments/F1-EXP-020/run.py.

The constants and functions below are copied from experiments/F1-EXP-020/run.py
unchanged (same names, same bodies, same values). Only the import block at the top
of this file differs: run.py's side-effecting top level (ffmpeg checks, output
directory creation, file loading, export) is intentionally NOT included.

Method (from the original): LPC pole rotation of the first formant (F1) on
reliable voiced frames, applied as a gain envelope on the STFT magnitude
spectrum with phase preserved.

Public API:
    apply_shift(x, sr, shift_hz) -> (y, stats)   original, unchanged return value
    shift_f1(x, sr, shift_hz) -> y               thin wrapper returning audio only
"""
import math

import numpy as np
from scipy import signal


STFT_N, STFT_H = 2048, 512


FMIN, FMAX = 90.0, 450.0


R_GATE_VOICED = 0.50


FLOOR_DB = 40.0


LPC_SR = 16000


LPC_ORDER = 14


F1_LO, F1_HI = 150.0, 500.0


BW_LO, BW_HI = 20.0, 600.0


SMOOTH_FRAMES = 2


def frame_f0_ac(win, sr):
    w = np.hanning(len(win)) * (win - win.mean())
    if not np.any(w):
        return float("nan"), 0.0
    c = signal.correlate(w, w, method="fft")[len(w) - 1:]
    if c[0] <= 0:
        return float("nan"), 0.0
    c = c / c[0]
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    if hi >= len(c):
        return float("nan"), 0.0
    k = int(np.argmax(c[lo:hi + 1])) + lo
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        k_f, r = k + d, float(y1)
    else:
        k_f, r = float(k), float(c[k])
    f0 = sr / k_f
    if not (FMIN <= f0 <= FMAX):
        return float("nan"), r
    return float(f0), float(r)


def levinson(r, order):
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0])
    if e <= 0:
        return a
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[1:i][::-1]))
        k = -acc / e
        anew = a.copy()
        for j in range(1, i):
            anew[j] = a[j] + k * a[i - j]
        anew[i] = k
        a = anew
        e *= (1.0 - k * k)
        if e <= 1e-12:
            break
    return a


def lpc_of(x16):
    xp = np.append(x16[0], x16[1:] - 0.63 * x16[:-1])
    xp = xp * np.hanning(len(xp))
    r = np.correlate(xp, xp, mode="full")[len(xp) - 1:len(xp) + LPC_ORDER]
    if r[0] <= 0:
        return None
    return levinson(r, LPC_ORDER)


def poles(a, sr=LPC_SR):
    roots = np.roots(a)
    freqs = np.angle(roots) * sr / (2.0 * np.pi)
    bws = -np.log(np.maximum(np.abs(roots), 1e-12)) * sr / np.pi
    return roots, freqs, bws


def f1_index(roots, freqs, bws):
    cand = [i for i in range(len(roots))
            if np.imag(roots[i]) > 0 and F1_LO <= freqs[i] <= F1_HI and BW_LO <= bws[i] <= BW_HI]
    return min(cand, key=lambda i: freqs[i]) if cand else None


def shift_gain(x_frame, sr, shift_hz):
    gcd = math.gcd(sr, LPC_SR)
    x16 = signal.resample_poly(x_frame, LPC_SR // gcd, sr // gcd)
    if len(x16) < LPC_ORDER * 4:
        return None
    a_old = lpc_of(x16)
    if a_old is None or len(a_old) < 3:
        return None
    roots, freqs, bws = poles(a_old)
    idx = f1_index(roots, freqs, bws)
    if idx is None:
        return None
    conj = int(np.argmin(np.abs(roots - np.conj(roots[idx]))))
    z = roots[idx]
    dw = shift_hz * 2.0 * np.pi / LPC_SR
    nz = np.abs(z) * np.exp(1j * (np.angle(z) + dw))
    rn = roots.copy()
    rn[idx] = nz
    rn[conj] = np.conj(nz)
    a_new = np.poly(rn).real
    a_new = a_new / a_new[0]
    w = np.linspace(0, np.pi, 401)
    k = np.arange(len(a_old))
    E = np.exp(-1j * np.outer(w, k))
    A_old = np.abs(E @ a_old)
    A_new = np.abs(E @ a_new)
    g = 20.0 * np.log10(np.maximum(A_old, 1e-10) / np.maximum(A_new, 1e-10))
    return w, g, float(freqs[idx]), float(bws[idx])


def apply_shift(x, sr, shift_hz):
    N, H = STFT_N, STFT_H
    ww = np.hanning(N)
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    nb = N // 2 + 1
    f_bins = np.arange(nb) * (sr / N)
    f16 = np.linspace(0, np.pi, 401) * (LPC_SR / (2.0 * np.pi))

    rms_db, rs = [], []
    for i in range(n_frames):
        fr = xp[i * H:i * H + N]
        rms_db.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        _, r = frame_f0_ac(fr, sr)
        rs.append(r)
    rms_db = np.array(rms_db)
    gate = rms_db.max() - FLOOR_DB

    gains = np.zeros((n_frames, nb))
    f1s, n_proc, n_fail = [], 0, 0
    for i in range(n_frames):
        if rms_db[i] < gate or rs[i] < R_GATE_VOICED:
            continue
        res = shift_gain(xp[i * H:i * H + N], sr, shift_hz)
        if res is None:
            n_fail += 1
            continue
        wv, g, f1o, _ = res
        gains[i] = np.interp(f_bins, f16, g, left=0.0, right=0.0)
        f1s.append(f1o)
        n_proc += 1

    if n_proc < 50:
        raise RuntimeError(f"可靠浊音帧过少 ({n_proc})，F1 无法可靠修改")

    sm = np.copy(gains)
    for i in range(n_frames):
        lo = max(0, i - SMOOTH_FRAMES // 2)
        hi = min(n_frames, i + SMOOTH_FRAMES // 2 + 1)
        sm[i] = gains[lo:hi].mean(axis=0)
    gains = sm

    y_ola = np.zeros(len(xp))
    norm = np.zeros(len(xp))
    for i in range(n_frames):
        fr = xp[i * H:i * H + N] * ww
        y = np.fft.irfft(np.fft.rfft(fr) * 10.0 ** (gains[i] / 20.0), N)
        s = i * H
        y_ola[s:s + N] += y * ww
        norm[s:s + N] += ww * ww
    y = (y_ola / np.maximum(norm, 1e-8))[pad:pad + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))

    stats = {"frames_total": int(n_frames), "vowel_frames": int(n_proc),
             "vowel_fraction": round(n_proc / n_frames, 3), "lpc_failures": int(n_fail),
             "f1_source_median_hz": round(float(np.median(f1s)), 1),
             "max_gain_db": round(float(np.max(np.abs(gains))), 2)}
    return y, stats


def shift_f1(x: np.ndarray, sr: int, shift_hz: float) -> np.ndarray:
    """Return only the shifted audio (float64 mono, same length as x)."""
    y, _stats = apply_shift(x, sr, shift_hz)
    return y
