"""Descriptive measurements (not listening scores) comparing a processed signal with its input. 22.05 kHz mono arrays."""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from colour_match import ColourConfig, ltas_db

SR = 22050
HOP = 128
BANDS = [(0, 500), (500, 1500), (1500, 4000), (4000, 8000)]


def block_db(x: np.ndarray, block: int = 2205) -> np.ndarray:
    n = len(x) // block
    return np.array([10 * np.log10(np.mean(x[i * block:(i + 1) * block] ** 2) + 1e-12) for i in range(n)])


def level_dips(inp: np.ndarray, out: np.ndarray) -> dict:
    """Worst 100 ms level change of the output vs the input over blocks within 45 dB of the loudest block."""
    n = min(len(inp), len(out))
    li, lo = block_db(inp[:n]), block_db(out[:n])
    act = li > li.max() - 45
    drop = (lo - li)[act]
    return {"worst_drop_db": round(float(drop.min()), 1), "seconds_below_-10db": round(float((drop < -10).sum() * 0.1), 1),
            "blocks": int(act.sum())}


def colour_shift(inp: np.ndarray, out: np.ndarray, same_timeline: bool = True) -> dict:
    cfg = ColourConfig(active_db=40)
    n = min(len(inp), len(out)) if same_timeline else None
    f, di, act = ltas_db(inp[:n] if n else inp, None, SR, cfg)
    _, do, _ = ltas_db(out[:n] if n else out, act if same_timeline else None, SR, cfg)
    vec = {f"{a}-{b}": round(float(np.mean((do - di)[(f >= a) & (f < b)])), 1) for a, b in BANDS}
    vec["nasal_norm_db"] = round(float(np.sqrt(np.mean(np.square(list(vec.values()))))), 1)
    return vec


def praat_pitch(y: np.ndarray):
    p = parselmouth.Sound(np.asarray(y, dtype=np.float64), SR).to_pitch_ac(
        time_step=HOP / SR, voicing_threshold=0.6, pitch_floor=80, pitch_ceiling=750)
    return p.selected_array["frequency"]


def fast_f0_cents(f0: np.ndarray, width: int = 11) -> float:
    ok = f0 > 0
    cents = np.where(ok, 1200 * np.log2(np.where(ok, f0, 1) / 440.0), np.nan)
    vals, i = [], 0
    while i < len(cents):
        if ok[i]:
            j = i
            while j + 1 < len(cents) and ok[j + 1]:
                j += 1
            if j - i + 1 >= 15:
                seg = cents[i:j + 1]
                sm = np.convolve(np.pad(seg, width // 2, mode="edge"), np.ones(width) / width, mode="valid")
                vals.append((seg - sm)[width:-width])
            i = j + 1
        else:
            i += 1
    return round(float(np.concatenate(vals).std()), 1) if vals else float("nan")


def voice_quality(y: np.ndarray) -> dict:
    s = parselmouth.Sound(np.asarray(y, dtype=np.float64), SR)
    h = s.to_harmonicity_cc(time_step=0.01, minimum_pitch=75).values[0]
    h = h[h > -200]
    pp = call(s, "To PointProcess (periodic, cc)", 80, 750)
    return {"hnr_median_db": round(float(np.median(h)), 1) if len(h) else float("nan"),
            "hnr_p10_db": round(float(np.percentile(h, 10)), 1) if len(h) else float("nan"),
            "jitter": round(float(call(pp, "Get jitter (local)", 0, 0, 1e-4, .02, 1.3)), 4),
            "shimmer": round(float(call([s, pp], "Get shimmer (local)", 0, 0, 1e-4, .02, 1.3, 1.6)), 4)}


def pitch_fidelity(inp: np.ndarray, out: np.ndarray) -> dict:
    """Same Praat extraction on both signals, same frame index: did the output keep the sung pitch?"""
    a, b = praat_pitch(inp), praat_pitch(out)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    both = (a > 0) & (b > 0)
    dev = np.abs(1200 * np.log2(b[both] / a[both])) if both.any() else np.array([np.nan])
    return {"median_dev_cents": round(float(np.median(dev)), 1), "p95_dev_cents": round(float(np.percentile(dev, 95)), 1),
            "voiced_kept": round(float(both.sum() / max(1, (a > 0).sum())), 3), "fast_f0_cents": fast_f0_cents(b)}


def all_metrics(inp: np.ndarray, out: np.ndarray) -> dict:
    return {**level_dips(inp, out), "colour": colour_shift(inp, out), **voice_quality(out), **pitch_fidelity(inp, out)}
