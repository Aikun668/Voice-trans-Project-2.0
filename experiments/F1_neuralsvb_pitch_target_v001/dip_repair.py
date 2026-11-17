"""Repair frames where the acoustic model output collapses far below the faithful resynthesis.

Observed on the user's recordings: the model output drops 10-27 dB below the input for tens of milliseconds, both in
quiet unvoiced passages (breaths, tails) and - after the F0 axis fix - inside loud sung notes, while the vocoder-only
resynthesis of the same mel stays within about 1 dB of the input. This module blends that faithful resynthesis back in
*only* at frames where the model output is at least `trigger_db` below it. No waveform of the original recording is
mixed in. `include_voiced=False` restricts the repair to frames whose target F0 is 0.

Pure numpy; same-length in and out; inputs are never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP = 128
WIN = 512


@dataclass(frozen=True)
class DipConfig:
    trigger_db: float = 8.0      # output this far below the faithful resynthesis starts to be restored
    full_db: float = 14.0        # ... and is fully restored at this drop
    floor_db: float = -70.0      # reference quieter than this is silence: nothing to restore
    active_range_db: float = 30.0  # only frames within this many dB of the loud end (99th percentile) are audible enough to repair
    include_voiced: bool = True  # also repair collapses inside sung (voiced) frames
    dilate_frames: int = 2       # grow the unvoiced mask a little so the blend can ramp inside it
    smooth_ms: float = 25.0      # raised-cosine smoothing of the blend weight


def frame_db(x: np.ndarray, hop: int = HOP, win: int = WIN) -> np.ndarray:
    """RMS level in dBFS of centred frames, one per hop."""
    n = len(x) // hop + 1
    pad = np.pad(np.asarray(x, dtype=float), (win // 2, win // 2))
    out = np.empty(n)
    for i in range(n):
        seg = pad[i * hop:i * hop + win]
        out[i] = 10.0 * np.log10(np.mean(seg ** 2) + 1e-12)
    return out


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask.copy()
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _smooth(w: np.ndarray, frames: int) -> np.ndarray:
    if frames < 2:
        return w
    kernel = np.hanning(frames * 2 + 1)
    kernel /= kernel.sum()
    return np.clip(np.convolve(np.pad(w, frames, mode="edge"), kernel, mode="valid"), 0.0, 1.0)


def baseline_offset_db(out: np.ndarray, ref: np.ndarray, cfg: DipConfig) -> float:
    """Clip-wide level difference ref - out (dB) over active frames. The model output is usually a few dB quieter than the
    faithful resynthesis everywhere; that constant is NOT a dip and must not count towards the trigger."""
    n = min(len(out), len(ref)) // HOP + 1
    lr, lo = frame_db(ref)[:n], frame_db(out)[:n]
    active = (lr > np.percentile(lr, 99) - cfg.active_range_db) & (lr > cfg.floor_db)
    return float(np.median((lr - lo)[active])) if active.any() else 0.0


def blend_weights(out: np.ndarray, ref: np.ndarray, target_f0: np.ndarray, sr: int,
                  cfg: DipConfig, offset_db: float | None = None) -> np.ndarray:
    """Per-hop-frame weight in [0, 1] for the (level-matched) faithful resynthesis `ref`."""
    n = min(len(out), len(ref)) // HOP + 1
    offset = baseline_offset_db(out, ref, cfg) if offset_db is None else offset_db
    lr = frame_db(ref)[:n]
    drop = lr - frame_db(out)[:n] - offset              # local excess drop only
    if cfg.include_voiced:
        allowed = np.ones(n, dtype=bool)
    else:
        allowed = _dilate(np.asarray(target_f0)[:n] <= 0, cfg.dilate_frames)
        if len(allowed) < n:
            allowed = np.pad(allowed, (0, n - len(allowed)), constant_values=False)
    audible = (lr > np.percentile(lr, 99) - cfg.active_range_db) & (lr > cfg.floor_db)
    w = np.clip((drop - cfg.trigger_db) / (cfg.full_db - cfg.trigger_db), 0.0, 1.0)
    w = np.where(allowed & audible, w, 0.0)
    return _smooth(w, int(round(cfg.smooth_ms / 1000.0 * sr / HOP)))


def repair_dips(out: np.ndarray, ref: np.ndarray, target_f0: np.ndarray, sr: int,
                config: DipConfig | None = None) -> tuple[np.ndarray, dict]:
    """Blend `ref` (vocoder-only, same F0), scaled to the model's overall level, into `out` where `out` collapsed locally."""
    cfg = config or DipConfig()
    if len(out) != len(ref):
        raise ValueError(f"length mismatch: out {len(out)} vs ref {len(ref)}")
    offset = baseline_offset_db(out, ref, cfg)
    w_frames = blend_weights(out, ref, target_f0, sr, cfg, offset)
    w = np.interp(np.arange(len(out)), np.arange(len(w_frames)) * HOP, w_frames)
    scaled_ref = np.asarray(ref, dtype=float) * 10.0 ** (-offset / 20.0)
    fixed = (1.0 - w) * np.asarray(out, dtype=float) + w * scaled_ref
    report = {"baseline_offset_db": round(offset, 2), "frames_touched": int((w_frames > 0.05).sum()),
              "max_weight": float(w_frames.max(initial=0.0)), "seconds_touched": float((w_frames > 0.05).sum() * HOP / sr)}
    return fixed, report
