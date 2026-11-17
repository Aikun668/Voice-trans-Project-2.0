"""Remove the global spectral colour the acoustic model adds to a recording.

Measured on the user's recordings: the full beautifier lowers <0.5 kHz by 1.5-2.5 dB, raises 0.5-4 kHz by 1-2 dB and
lowers 4-8 kHz by 2-7 dB relative to the input - the classic 'nasal / honky' curve. This module estimates that
long-term difference on the active frames and applies the inverse as one smooth zero-phase EQ. Local changes the model
makes (anything that is not a stable spectral tilt) are kept.

Pure numpy/scipy; same-length in and out; inputs are never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin2, fftconvolve


@dataclass(frozen=True)
class ColourConfig:
    n_fft: int = 2048
    hop: int = 512
    active_db: float = 30.0       # frames within this many dB of the loudest (100-4000 Hz) count as active
    smooth_octaves: float = 1 / 3  # smoothing of the gain curve on a log-frequency axis
    max_gain_db: float = 9.0
    f_lo: float = 80.0            # full correction from f_lo up to f_hi; ramps to zero over the octave below f_lo
    f_hi: float = 8000.0          # ... and from f_hi to f_end (the mel stops at 11 kHz)
    f_end: float = 11000.0
    strength: float = 1.0         # 0 = off, 1 = full match
    ref_lo: float = 300.0         # the curve is levelled to 0 dB over this range first: overall loudness is not 'colour'
    ref_hi: float = 3000.0
    taps: int = 2049


def voiced_frame_mask(f0: np.ndarray, n_frames: int, cfg: ColourConfig, f0_hop: int = 128, min_frac: float = 0.8) -> np.ndarray:
    """STFT frames (cfg.hop, cfg.n_fft) in which at least `min_frac` of the F0 frames (hop `f0_hop`, frame k centred at k*f0_hop) are voiced."""
    f0 = np.asarray(f0, dtype=float)
    out = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        lo, hi = i * cfg.hop // f0_hop, (i * cfg.hop + cfg.n_fft) // f0_hop
        seg = f0[lo:hi]
        out[i] = len(seg) > 0 and float(np.mean(seg > 0)) >= min_frac
    return out


def ltas_db(y: np.ndarray, ref_active: np.ndarray | None, sr: int, cfg: ColourConfig, frame_mask: np.ndarray | None = None):
    """Long-term average power spectrum (dB) over active frames. `ref_active` lets two signals share the frame set."""
    y = np.asarray(y, dtype=float)
    win = np.hanning(cfg.n_fft)
    n = 1 + max(0, (len(y) - cfg.n_fft) // cfg.hop)
    frames = np.stack([y[i * cfg.hop:i * cfg.hop + cfg.n_fft] * win for i in range(n) if i * cfg.hop + cfg.n_fft <= len(y)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
    if ref_active is None:
        band = (freqs >= 100) & (freqs < 4000)
        level = 10.0 * np.log10(power[:, band].sum(1) + 1e-12)
        ref_active = level > level.max() - cfg.active_db
        if frame_mask is not None:
            m = np.zeros(len(ref_active), dtype=bool)
            m[:min(len(m), len(frame_mask))] = frame_mask[:len(m)]
            ref_active = ref_active & m
    avg = power[ref_active].mean(0) if ref_active.any() else power.mean(0)
    return freqs, 10.0 * np.log10(avg + 1e-12), ref_active


def _smooth_log_freq(freqs: np.ndarray, values: np.ndarray, octaves: float) -> np.ndarray:
    out = np.copy(values)
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        lo, hi = f * 2 ** (-octaves / 2), f * 2 ** (octaves / 2)
        sel = (freqs >= lo) & (freqs <= hi)
        out[i] = values[sel].mean()
    return out


def match_gain_db(reference: np.ndarray, processed: np.ndarray, sr: int, cfg: ColourConfig | None = None,
                  voiced_f0: np.ndarray | None = None):
    """(freqs, gain_db) that moves the long-term spectrum of `processed` onto that of `reference`.
    With `voiced_f0` (Hz per 128-sample frame, 0 = unvoiced) only sung frames are used to estimate the curve."""
    cfg = cfg or ColourConfig()
    n = min(len(reference), len(processed))
    mask = None
    if voiced_f0 is not None:
        n_frames = 1 + max(0, (n - cfg.n_fft) // cfg.hop)
        mask = voiced_frame_mask(voiced_f0, n_frames, cfg)
    freqs, ref_db, active = ltas_db(reference[:n], None, sr, cfg, mask)
    _, out_db, _ = ltas_db(processed[:n], active, sr, cfg)
    gain = _smooth_log_freq(freqs, ref_db - out_db, cfg.smooth_octaves)
    gain = gain - float(np.median(gain[(freqs >= cfg.ref_lo) & (freqs <= cfg.ref_hi)]))
    f = np.maximum(freqs, 1.0)
    low = np.clip(np.log2(f / (cfg.f_lo / 2.0)), 0.0, 1.0)
    high = np.clip(np.log2(cfg.f_end / f) / np.log2(cfg.f_end / cfg.f_hi), 0.0, 1.0)
    taper = np.minimum(low, high)
    gain = np.clip(gain, -cfg.max_gain_db, cfg.max_gain_db) * taper * cfg.strength
    return freqs, gain


def apply_eq(y: np.ndarray, sr: int, freqs: np.ndarray, gain_db: np.ndarray, taps: int = 2049) -> np.ndarray:
    """Zero-phase (linear-phase, delay-compensated) EQ with the given magnitude curve."""
    fir = firwin2(taps, freqs / (sr / 2.0), 10.0 ** (gain_db / 20.0))
    return fftconvolve(np.asarray(y, dtype=float), fir, mode="same")


def colour_match(reference: np.ndarray, processed: np.ndarray, sr: int,
                 config: ColourConfig | None = None, voiced_f0: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    cfg = config or ColourConfig()
    freqs, gain = match_gain_db(reference, processed, sr, cfg, voiced_f0)
    out = apply_eq(processed, sr, freqs, gain, cfg.taps)
    edges = [(0, 500), (500, 1500), (1500, 4000), (4000, 8000)]
    report = {"band_gain_db": {f"{a}-{b}": round(float(gain[(freqs >= a) & (freqs < b)].mean()), 2) for a, b in edges},
              "max_abs_gain_db": round(float(np.abs(gain).max()), 2)}
    return out, report
