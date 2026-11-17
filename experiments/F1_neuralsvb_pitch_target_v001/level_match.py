"""Make the output follow the input's loudness contour (F1 does not touch dynamics unless asked to).

Measured on the user's recordings: after removing the clip-wide level difference, the acoustic model's output deviates
from the input by +-5 dB or more in about one frame in ten (p5..p95 spread 10-12 dB), including 10-25 dB collapses.
This module computes the per-frame deviation, ignores small natural differences (dead zone), limits the correction and
applies it as one smooth gain curve. Only the *level* of the model output changes; its waveform and timbre do not, and
no other signal is mixed in.

Pure numpy; same-length in and out; inputs are never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dip_repair import HOP, frame_db


@dataclass(frozen=True)
class LevelConfig:
    active_range_db: float = 35.0   # frames within this of the loud end (99th percentile of the input level) are matched
    floor_db: float = -70.0
    dead_zone_db: float = 2.0       # natural differences smaller than this are left to the model
    smooth_ms: float = 40.0
    max_boost_db: float = 15.0
    max_cut_db: float = 10.0


def _hann_smooth(x: np.ndarray, frames: int) -> np.ndarray:
    if frames < 2:
        return x
    k = np.hanning(frames * 2 + 1)
    k /= k.sum()
    return np.convolve(np.pad(x, frames, mode="edge"), k, mode="valid")


def level_correction_db(inp: np.ndarray, out: np.ndarray, sr: int, cfg: LevelConfig) -> tuple[np.ndarray, float]:
    """Per-hop-frame gain (dB) that makes `out` follow `inp`'s contour, and the clip-wide baseline offset (dB)."""
    n = min(len(inp), len(out))
    li, lo = frame_db(inp[:n]), frame_db(out[:n])
    active = (li > np.percentile(li, 99) - cfg.active_range_db) & (li > cfg.floor_db)
    if not active.any():
        return np.zeros(len(li)), 0.0
    dev = li - lo
    baseline = float(np.median(dev[active]))
    c = np.where(active, dev - baseline, 0.0)
    c = np.sign(c) * np.maximum(np.abs(c) - cfg.dead_zone_db, 0.0)
    c = np.clip(c, -cfg.max_cut_db, cfg.max_boost_db)
    return _hann_smooth(c, int(round(cfg.smooth_ms / 1000.0 * sr / HOP))), baseline


def level_match(inp: np.ndarray, out: np.ndarray, sr: int,
                config: LevelConfig | None = None) -> tuple[np.ndarray, dict]:
    cfg = config or LevelConfig()
    if len(inp) != len(out):
        raise ValueError(f"length mismatch: input {len(inp)} vs output {len(out)}")
    gain_db, baseline = level_correction_db(inp, out, sr, cfg)
    g = 10.0 ** (np.interp(np.arange(len(out)), np.arange(len(gain_db)) * HOP, gain_db) / 20.0)
    fixed = np.asarray(out, dtype=float) * g
    touched = np.abs(gain_db) > 0.5
    return fixed, {"baseline_offset_db": round(baseline, 2), "seconds_corrected": round(float(touched.sum() * HOP / sr), 2),
                   "max_boost_db": round(float(gain_db.max(initial=0.0)), 1), "max_cut_db": round(float(-gain_db.min(initial=0.0)), 1)}
