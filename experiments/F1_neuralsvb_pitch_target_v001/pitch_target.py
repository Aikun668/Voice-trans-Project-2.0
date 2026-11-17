"""Build a pitch target for NeuralSVB from the singer's own F0 only (no reference recording).

F1 rule: no melody or rhythm change. This module therefore only *repairs* detector mistakes:
  1. octave errors (a short stretch read one octave off its neighbours),
  2. isolated spikes inside a sung note,
  3. short voiced dropouts inside a sung note (confirmed by an independent voicing hint when available).
Unvoiced frames (consonants, breaths, pauses) stay unvoiced. Optional light smoothing is off by default.

Pure numpy; never mutates its input; output has the same length as the input.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

OCTAVE_CENTS = 1200.0


@dataclass(frozen=True)
class TargetConfig:
    octave_window: int = 15          # frames each side used as local reference
    octave_trigger_cents: float = 900.0
    octave_accept_cents: float = 350.0
    spike_max_frames: int = 3        # voiced runs this short, far from both neighbours, are spikes
    spike_cents: float = 400.0
    fill_max_gap: int = 8            # unvoiced frames (~46 ms at hop 128 / 22.05 kHz)
    fill_flank_cents: float = 300.0  # flanking pitch must agree this closely to bridge a gap
    hint_min_voiced: float = 0.5     # share of gap frames the independent detector must call voiced
    smooth_frames: int = 0           # 0 = off (the pro reference is NOT smoother than amateurs)


def _to_cents(f0: np.ndarray) -> np.ndarray:
    out = np.full(f0.shape, np.nan)
    ok = f0 > 0
    out[ok] = 1200.0 * np.log2(f0[ok] / 440.0)
    return out


def _from_cents(c: np.ndarray) -> np.ndarray:
    out = np.zeros(c.shape)
    ok = ~np.isnan(c)
    out[ok] = 440.0 * 2.0 ** (c[ok] / 1200.0)
    return out


def _runs(mask: np.ndarray):
    """Yield (start, end_inclusive) for each run of True."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


def _fix_octaves(c: np.ndarray, cfg: TargetConfig) -> tuple[np.ndarray, int]:
    out, fixed = c.copy(), 0
    voiced = np.where(~np.isnan(c))[0]
    for i in voiced:
        lo, hi = max(0, i - cfg.octave_window), min(len(c), i + cfg.octave_window + 1)
        neigh = np.concatenate([c[lo:i], c[i + 1:hi]])
        neigh = neigh[~np.isnan(neigh)]
        if len(neigh) < 5:
            continue
        ref = float(np.median(neigh))
        if abs(c[i] - ref) <= cfg.octave_trigger_cents:
            continue
        for shift in (-OCTAVE_CENTS, OCTAVE_CENTS):
            if abs(c[i] + shift - ref) <= cfg.octave_accept_cents:
                out[i] = c[i] + shift
                fixed += 1
                break
    return out, fixed


def _rolling_median(seg: np.ndarray, width: int) -> np.ndarray:
    pad = np.pad(seg, width // 2, mode="edge")
    return np.array([np.median(pad[k:k + width]) for k in range(len(seg))])


def _remove_spikes(c: np.ndarray, cfg: TargetConfig) -> tuple[np.ndarray, int]:
    """Inside each voiced stretch, a run of <= spike_max_frames frames that sits far from the rolling
    median while both flanking frames agree with it is a detector spike. A real note step is never
    flagged: the frames on its new side form a run longer than spike_max_frames."""
    out, fixed = c.copy(), 0
    for s, e in _runs(~np.isnan(c)):
        if e - s + 1 < 9:
            continue
        seg = c[s:e + 1]
        flag = np.abs(seg - _rolling_median(seg, 9)) > cfg.spike_cents
        for fs, fe in _runs(flag):
            if fe - fs + 1 > cfg.spike_max_frames or fs == 0 or fe == len(seg) - 1:
                continue
            out[s + fs:s + fe + 1] = np.interp(np.arange(fs, fe + 1), [fs - 1, fe + 1], [seg[fs - 1], seg[fe + 1]])
            fixed += fe - fs + 1
    return out, fixed


def _fill_gaps(c: np.ndarray, hint: np.ndarray | None, cfg: TargetConfig) -> tuple[np.ndarray, int]:
    out, filled = c.copy(), 0
    for s, e in _runs(np.isnan(c)):
        if s == 0 or e == len(c) - 1 or e - s + 1 > cfg.fill_max_gap:
            continue
        left, right = c[s - 1], c[e + 1]
        if np.isnan(left) or np.isnan(right) or abs(left - right) > cfg.fill_flank_cents:
            continue
        if hint is not None and float(np.mean(hint[s:e + 1])) < cfg.hint_min_voiced:
            continue
        out[s:e + 1] = np.interp(np.arange(s, e + 1), [s - 1, e + 1], [left, right])
        filled += e - s + 1
    return out, filled


def _smooth(c: np.ndarray, width: int) -> np.ndarray:
    if width < 3:
        return c
    width |= 1
    out = c.copy()
    for s, e in _runs(~np.isnan(c)):
        seg = c[s:e + 1]
        pad = np.pad(seg, width // 2, mode="edge")
        out[s:e + 1] = np.array([np.median(pad[k:k + width]) for k in range(len(seg))])
    return out


def make_target(f0: np.ndarray, voiced_hint: np.ndarray | None = None,
                config: TargetConfig | None = None) -> tuple[np.ndarray, dict]:
    """Return (target_f0_hz, report). `voiced_hint` is an independent per-frame voicing decision (bool)."""
    cfg = config or TargetConfig()
    f0 = np.asarray(f0, dtype=float)
    hint = None if voiced_hint is None else np.asarray(voiced_hint, dtype=bool)
    if hint is not None and len(hint) != len(f0):
        raise ValueError(f"voiced_hint length {len(hint)} != f0 length {len(f0)}")
    c = _to_cents(f0)
    c, octave = _fix_octaves(c, cfg)
    c, spikes = _remove_spikes(c, cfg)
    c, filled = _fill_gaps(c, hint, cfg)
    c = _smooth(c, cfg.smooth_frames)
    target = _from_cents(c)
    report = {"octave_fixed_frames": octave, "spike_frames": spikes, "gap_filled_frames": filled,
              "voiced_in": int((f0 > 0).sum()), "voiced_out": int((target > 0).sum())}
    return target, report


def with_smoothing(config: TargetConfig, frames: int) -> TargetConfig:
    return replace(config, smooth_frames=frames)
