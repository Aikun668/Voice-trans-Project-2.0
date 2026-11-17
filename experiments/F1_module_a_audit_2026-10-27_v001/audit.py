"""Read-only comparison of A on three recordings; never exports processed audio."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import scipy.signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_adaptive_v0.analyzer import analyze, read_wav

OUT = Path(__file__).parent / 'results_v001'
SOURCES = [
    ('person_normal', ROOT / 'test_dataset/my_singing/shige_normal.wav'),
    ('person_off_pitch', ROOT / 'test_dataset/my_singing/shige_medium.wav'),
    ('network_normal', ROOT / 'test_dataset/web_singing/gtsinger_alto_control/告白气球_normal.wav'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def mask(intervals, sr, duration, offset=0):
    # 10 ms grid; compare the original region after compensating prefix padding.
    hop = round(sr * .01)
    times = np.arange(int(duration * sr) // hop) * hop / sr + offset
    answer = np.zeros(len(times), dtype=bool)
    for section in intervals:
        answer |= (times >= section['start_seconds']) & (times < section['end_seconds'])
    return answer


def compare(base, other, sr, offset, gain_db):
    result = {
        'expected_level_shift_db': gain_db,
        'actual_p50_level_shift_db': other['rms_dbfs']['p50'] - base['rms_dbfs']['p50'],
        'dynamic_range_delta_db': other['dynamic_range_db'] - base['dynamic_range_db'],
        'active_seconds_delta': other['active_seconds'] - base['active_seconds'],
        'gate_delta_db': other['activity_gate_dbfs'] - base['activity_gate_dbfs'],
        'max_band_fraction_delta': max(abs(other['band_energy'][k]['median_energy_fraction'] - v['median_energy_fraction']) for k, v in base['band_energy'].items()),
    }
    for field, a, b in [('activity_time_mismatch', base['active_intervals'], other['active_intervals']),
                        ('high_candidate_time_mismatch', base['sibilance']['candidate_intervals'], other['sibilance']['candidate_intervals'])]:
        left = mask(a, sr, base['duration_seconds'])
        right = mask(b, sr, base['duration_seconds'], offset)
        result[field] = float(np.mean(left != right))
    # Diagnostic tolerances, not a clinical or perceptual standard.
    result['diagnostic_within_tolerance'] = bool(
        abs(result['actual_p50_level_shift_db'] - gain_db) <= .1 and
        abs(result['dynamic_range_delta_db']) <= .1 and
        abs(result['active_seconds_delta']) <= .1 and
        result['max_band_fraction_delta'] <= .01 and
        result['activity_time_mismatch'] <= .01 and result['high_candidate_time_mismatch'] <= .01)
    return result


def independent_reference(x, sr, features):
    n, hop = round(sr * .02), round(sr * .01)
    # Independent NumPy framing and FFT; same stated window, no librosa feature functions.
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    active = db > max(float(np.percentile(db, 95)) - 40, -75)
    hann = scipy.signal.windows.hann(n, sym=False)
    power = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2
    hz = np.fft.rfftfreq(n, 1 / sr)
    def energy(lo, hi):
        return power[:, (hz >= lo) & (hz < hi)].sum(axis=1)
    total = np.maximum(energy(80, min(10000, sr / 2)), 1e-20)
    specs = [('body', 80, 500), ('mid', 500, 3000), ('presence', 3000, 5000), ('high', 5000, 8000)]
    diffs = {key: float(np.median((energy(lo, hi) / total)[active]) - features['band_energy'][key]['median_energy_fraction']) for key, lo, hi in specs}
    active_idx = np.flatnonzero(active)
    # These are 20 ms measurement windows, not verified lyric or musical passages.
    low = int(active_idx[np.argmin(rms[active_idx])])
    high = int(active_idx[np.argmax(rms[active_idx])])
    segments = []
    for label, i in [('quietest_active_frame', low), ('loudest_active_frame', high)]:
        segments.append({'type': label, 'start_seconds': i * hop / sr, 'end_seconds': (i * hop + n) / sr,
                         'rms_dbfs_independent': float(db[i])})
    return {'p50_rms_abs_error_db': float(abs(np.percentile(db[active], 50) - features['rms_dbfs']['p50'])),
            'p95_p10_dynamic_abs_error_db': float(abs((np.percentile(db[active], 95) - np.percentile(db[active], 10)) - features['dynamic_range_db'])),
            'band_fraction_deltas': diffs, 'measurement_frames': segments,
            'method': 'NumPy direct RMS/FFT with SciPy Hann; cross-checks numerical implementation, not perceptual ground truth.'}


def main():
    # Exclusive creation: preserve any earlier attempt or user files.
    OUT.mkdir(exist_ok=False)
    report = {'scope': 'module A analysis only; no DSP and no listening judgment',
              'assumptions': 'normal/off-pitch labels identify input files only; no pitch correctness claim',
              'tolerances': {'level_db': .1, 'dynamic_db': .1, 'active_seconds': .1,
                             'band_fraction': .01, 'time_mismatch_fraction': .01}, 'recordings': []}
    for label, path in SOURCES:
        before = sha(path)
        x, sr, info = read_wav(path)
        f = analyze(x, sr)
        save(OUT / f'{label}_features.json', f)
        observations = {}
        for seconds in (1, 5):
            padded = np.pad(x, (seconds * sr, seconds * sr))
            observations[f'pad_{seconds}s_each_side'] = compare(f, analyze(padded, sr), sr, seconds, 0)
        for db in (-12, -6):
            observations[f'gain_{db}dB'] = compare(f, analyze(x * 10 ** (db / 20), sr), sr, 0, db)
        reference = independent_reference(x, sr, f)
        after = sha(path)
        if before != after:
            raise RuntimeError('Source changed during audit')
        sfinfo = sf.info(str(path))
        report['recordings'].append({'id': label, 'path': str(path), 'source_sha256': before,
            'format': {'sample_rate': sr, 'subtype': info['subtype'], 'samples': len(x), 'seconds': len(x) / sr},
            'features_file': f'{label}_features.json', 'activity_gate_dbfs': f['activity_gate_dbfs'],
            'active_seconds': f['active_seconds'], 'dynamic_range_db': f['dynamic_range_db'],
            'candidate_count': len(f['sibilance']['candidate_intervals']),
            'candidate_preview': f['sibilance']['candidate_intervals'][:5],
            'independent_calculation': reference, 'invariance': observations})
    save(OUT / 'audit_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
