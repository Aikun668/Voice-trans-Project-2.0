"""Descriptive features; frequency energy never diagnoses perceptual quality."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def read_wav(path):
    info = sf.info(str(path))
    if info.format not in ('WAV', 'WAVEX') or info.subtype not in ('PCM_16', 'PCM_24') or info.channels != 1:
        raise ValueError('Input must be mono PCM16/PCM24 WAV; convert an explicit COPY if needed.')
    x, sr = sf.read(str(path), dtype='float64')
    if len(x) != info.frames:
        raise ValueError('Truncated input')
    return x, sr, {'file': str(Path(path).resolve()), 'subtype': info.subtype}


def intervals(mask, hop, sr, window, offset_samples=0, total_samples=None):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        start = max(0, int(a) * hop - offset_samples)
        end = int(b - 1) * hop + window - offset_samples
        if total_samples is not None:
            end = min(total_samples, end)
        if end > start:
            result.append({'start_seconds': round(start / sr, 4), 'end_seconds': round(end / sr, 4)})
    return result


def analyze(x, sr):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not 16000 <= sr <= 96000 or len(x) < sr * 3 or not np.all(np.isfinite(x)):
        raise ValueError('Need finite mono audio, >=3 seconds, 16–96 kHz.')
    n, hop = round(sr * .02), round(sr * .01)
    # Include the same partial edge frames whether the input already has
    # digital silence around it or not. Padding exists only for analysis.
    pad = 2 * hop
    framed_audio = np.pad(x, (pad, pad))
    rms = librosa.feature.rms(y=framed_audio, frame_length=n, hop_length=hop, center=False)[0]
    levels = 20 * np.log10(np.maximum(rms, 1e-12))
    # Exact digital silence must not change the percentile used to estimate
    # a recording's operating level. Nonzero low-level frames are retained
    # here; no assumption is made that they are speech or noise.
    nonzero = rms > 0
    if not np.any(nonzero):
        raise ValueError('Insufficient measurable active audio (digital silence).')
    gate = max(float(np.percentile(levels[nonzero], 95)) - 40, -75)
    active = levels > gate
    if np.sum(active) * hop / sr < 1 or np.max(np.abs(x)) < 1e-5:
        raise ValueError('Insufficient measurable active audio (need >=1 second).')
    # Only count frame hops centered within the actual nonzero sample span;
    # the synthetic edge padding must not add time to active_seconds.
    nonzero_samples = np.flatnonzero(x)
    hop_centers = np.arange(len(active)) * hop - pad + hop / 2
    in_source_span = (hop_centers >= nonzero_samples[0]) & (hop_centers <= nonzero_samples[-1])
    active_seconds = min(float(np.sum(active & in_source_span) * hop / sr), len(x) / sr)
    frames = librosa.util.frame(framed_audio, frame_length=n, hop_length=hop)
    peaks = 20 * np.log10(np.maximum(np.max(np.abs(frames), axis=0), 1e-12))
    p = {f'p{k}': float(np.percentile(levels[active], k)) for k in (10, 50, 90, 95, 99)}
    # Hann-window STFT on the same unpadded time grid as RMS.
    power = np.abs(librosa.stft(framed_audio, n_fft=n, hop_length=hop, center=False)) ** 2
    hz = np.fft.rfftfreq(n, 1 / sr)
    def energy(lo, hi):
        return np.sum(power[(hz >= lo) & (hz < hi)], axis=0)
    total = np.maximum(energy(80, min(10000, sr / 2)), 1e-20)
    bands = {}
    for label, lo, hi in [('body', 80, 500), ('mid', 500, 3000), ('presence', 3000, 5000), ('high', 5000, 8000)]:
        ratio = energy(lo, hi) / total
        bands[label] = {'range_hz': [lo, hi], 'median_energy_fraction': float(np.median(ratio[active]))}
    hf = energy(5000, 8000)
    mid = energy(500, 4000)
    contrast = 10 * np.log10(np.maximum(hf, 1e-20) / np.maximum(mid, 1e-20))
    # Coincident high-band dominance + change, only a candidate locator.
    rise = np.r_[0, np.maximum(0, 10 * np.log10(np.maximum(hf[1:], 1e-20) / np.maximum(hf[:-1], 1e-20)))]
    candidate = active & (contrast > -3) & ((rise > 3) | (contrast > 3))
    loud = active & (levels >= p['p90'])
    def original_intervals(flags):
        return intervals(flags, hop, sr, n, offset_samples=pad, total_samples=len(x))
    loud_events = original_intervals(loud)
    return {
        'schema_version': 'f1.features.v0.1', 'sample_rate': sr, 'samples': len(x),
        'duration_seconds': len(x) / sr, 'frame_ms': 20, 'hop_ms': 10,
        'activity_gate_dbfs': gate,
        'activity_method': 'relative_rms_gate_excludes_exact_zero_frames_not_vocal_detection',
        'active_seconds': active_seconds,
        'active_intervals': original_intervals(active),
        'rms_dbfs': p, 'dynamic_range_db': p['p95'] - p['p10'],
        'frame_peak_dbfs_p90': float(np.percentile(peaks[active], 90)),
        'sample_peak_dbfs': float(20 * np.log10(max(np.max(np.abs(x)), 1e-12))),
        'full_scale_samples': int(np.sum(np.abs(x) >= 32767 / 32768)),
        'longest_loud_event_seconds': max((p['end_seconds'] - p['start_seconds'] for p in loud_events), default=0),
        'band_energy': bands,
        'tonal_quality': {'status': 'unknown', 'reason': 'No perceptual calibration; spectral balance may be natural timbre.'},
        'sibilance': {'status': 'unknown', 'candidate_intervals': original_intervals(candidate),
                      'candidate_frame_fraction': float(np.mean(candidate[active])),
                      'reason': 'High-frequency bursts can be consonants, breath or natural timbre; not confirmed defects.'},
    }
