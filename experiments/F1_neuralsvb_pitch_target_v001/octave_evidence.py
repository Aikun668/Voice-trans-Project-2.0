"""Fix octave errors of a pitch tracker using independent evidence from the waveform itself.

A stretch is suspicious when the tracker (Praat, what the model receives) and an independent tracker (pYIN) disagree by
one octave for >= `min_frames` frames. The decision is made on the original waveform with a harmonic-comb test:

    score(f) = mean over k=1..8 of  log|S(k*f)| - log|S((k-0.5)*f)|        (a true fundamental has energy ON its
                                                                           harmonics and little BETWEEN them)

Only when the octave candidate beats the tracker's value by `margin` are the stretch's frames multiplied by 2 or 0.5.
Melody is never changed: the corrected pitch is the same note, one octave away from a detector mistake.

Pure numpy; same-length in and out; inputs are never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OctaveConfig:
    min_frames: int = 3
    octave_tol: float = 0.12      # |log2(ratio)| within this of 1 counts as an octave disagreement
    margin: float = 0.3           # comb-score advantage required to overrule the tracker
    win_s: float = 0.05
    extend_frames: int = 6        # continue the correction into adjacent frames that carry the same wrong octave
    extend_tol: float = 0.15
    max_harmonic_hz: float = 9000.0


def comb_score(wav: np.ndarray, sr: int, f: float, t: float, win_s: float = 0.05,
               max_hz: float = 9000.0) -> float:
    i, half = int(t * sr), int(win_s * sr / 2)
    seg = np.asarray(wav[max(i - half, 0):i + half], dtype=float)
    if len(seg) < 256 or f <= 0:
        return float("nan")
    seg = seg * np.hanning(len(seg))
    n = 1 << 15
    spec = np.abs(np.fft.rfft(seg, n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    def level(freq: float) -> float:
        sel = np.abs(freqs - freq) < 0.03 * freq
        return float(np.log(spec[sel].max() + 1e-9)) if sel.any() else float("nan")

    ks = [k for k in range(1, 9) if k * f < max_hz]
    return float(np.nanmean([level(k * f) - level((k - 0.5) * f) for k in ks]))


def _stretches(flag: np.ndarray, min_frames: int):
    """Runs of True allowing a single-frame hole; yields (start, end_inclusive)."""
    n, i = len(flag), 0
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and (flag[j + 1] or (j + 2 < n and flag[j + 2])):
                j += 1
            if j - i + 1 >= min_frames:
                yield i, j
            i = j + 1
        else:
            i += 1


def fix_octaves(f0: np.ndarray, ref_f0: np.ndarray, wav: np.ndarray, sr: int, hop: int,
                config: OctaveConfig | None = None, time_offset_frames: float = 0.0):
    """Return (corrected_f0, report). `ref_f0` must be on the same frame axis as `f0` (0 = unvoiced).

    `time_offset_frames` converts a frame index of the tracks to waveform time: t = (i - offset) * hop / sr.
    """
    cfg = config or OctaveConfig()
    f0 = np.asarray(f0, dtype=float)
    ref = np.asarray(ref_f0, dtype=float)
    if len(ref) != len(f0):
        raise ValueError(f"ref_f0 length {len(ref)} != f0 length {len(f0)}")
    both = (f0 > 0) & (ref > 0)
    ratio = np.zeros(len(f0))
    ratio[both] = np.log2(ref[both] / f0[both])
    flag = both & (np.abs(np.abs(ratio) - 1.0) < cfg.octave_tol)
    out, events = f0.copy(), []
    for s, e in _stretches(flag, cfg.min_frames):
        idx = np.arange(s, e + 1)
        idx = idx[flag[idx]]
        tracker = float(np.median(f0[idx]))
        shift = 2.0 if np.median(ratio[idx]) > 0 else 0.5
        times = [(i - time_offset_frames) * hop / sr for i in idx[::2]]
        s_tracker = float(np.nanmean([comb_score(wav, sr, tracker, t, cfg.win_s, cfg.max_harmonic_hz) for t in times]))
        s_alt = float(np.nanmean([comb_score(wav, sr, tracker * shift, t, cfg.win_s, cfg.max_harmonic_hz) for t in times]))
        event = {"start_frame": int(s), "end_frame": int(e), "tracker_hz": round(tracker, 1), "alt_hz": round(tracker * shift, 1),
                 "comb_tracker": round(s_tracker, 2), "comb_alt": round(s_alt, 2), "applied": False}
        if s_alt - s_tracker >= cfg.margin:
            lo, hi = s, e
            while lo - 1 >= 0 and s - lo < cfg.extend_frames and f0[lo - 1] > 0 and abs(np.log2(f0[lo - 1] / tracker)) < cfg.extend_tol:
                lo -= 1
            while hi + 1 < len(f0) and hi - e < cfg.extend_frames and f0[hi + 1] > 0 and abs(np.log2(f0[hi + 1] / tracker)) < cfg.extend_tol:
                hi += 1
            span = np.arange(lo, hi + 1)
            span = span[f0[span] > 0]
            out[span] = f0[span] * shift
            event.update(applied=True, frames_changed=int(len(span)), shift=shift)
        events.append(event)
    return out, {"stretches": events, "frames_changed": int(sum(v.get("frames_changed", 0) for v in events))}
