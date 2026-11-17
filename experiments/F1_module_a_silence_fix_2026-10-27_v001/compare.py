"""Compare frozen pre-fix and current analyzer; source WAVs are read-only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_adaptive_v0.analyzer import analyze, read_wav

HERE = Path(__file__).parent
OUTPUT = HERE / 'comparison_v002.json'
spec = importlib.util.spec_from_file_location('pre_fix_analyzer', HERE / 'analyzer_before.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
SOURCES = [('person_normal', ROOT / 'test_dataset/my_singing/shige_normal.wav'),
           ('person_off_pitch', ROOT / 'test_dataset/my_singing/shige_medium.wav'),
           ('network_normal', ROOT / 'test_dataset/web_singing/gtsinger_alto_control/告白气球_normal.wav')]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metric(f):
    return {'gate_dbfs': f['activity_gate_dbfs'], 'dynamic_db': f['dynamic_range_db'],
            'active_seconds': f['active_seconds'], 'p50_dbfs': f['rms_dbfs']['p50'],
            'candidate_count_not_events': len(f['sibilance']['candidate_intervals']),
            'band_fractions': {k: v['median_energy_fraction'] for k, v in f['band_energy'].items()}}

def time_mask(intervals, times):
    mask = np.zeros(len(times), dtype=bool)
    for item in intervals:
        mask |= (times >= item['start_seconds']) & (times < item['end_seconds'])
    return mask

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    out = {'scope': 'A only, digital silence fix; no processed WAV or listening evaluation',
           'old_code_sha256': sha(HERE / 'analyzer_before.py'),
           'new_code_sha256': sha(ROOT / 'f1_adaptive_v0/analyzer.py'),
           'tolerance_db': .1, 'cases': []}
    for name, path in SOURCES:
        source_hash = sha(path)
        x, sr, _ = read_wav(path)
        base_old, base_new = before.analyze(x, sr), analyze(x, sr)
        result = {'name': name, 'source_path': str(path), 'source_sha256': source_hash,
                  'old_baseline': metric(base_old), 'new_baseline': metric(base_new), 'variations': {}}
        for sec in (1, 5, 10):
            padded = np.pad(x, (sec * sr, sec * sr))
            old, new = before.analyze(padded, sr), analyze(padded, sr)
            times = np.arange(0, len(x) / sr, .01)
            result['variations'][f'digital_silence_{sec}s_each_side'] = {
                'old_dynamic_delta_db': old['dynamic_range_db'] - base_old['dynamic_range_db'],
                'new_dynamic_delta_db': new['dynamic_range_db'] - base_new['dynamic_range_db'],
                'new_gate_delta_db': new['activity_gate_dbfs'] - base_new['activity_gate_dbfs'],
                'new_activity_delta_seconds': new['active_seconds'] - base_new['active_seconds'],
                'activity_time_mismatch_fraction': float(np.mean(time_mask(base_new['active_intervals'], times) != time_mask(new['active_intervals'], times + sec))),
                'candidate_time_mismatch_fraction': float(np.mean(time_mask(base_new['sibilance']['candidate_intervals'], times) != time_mask(new['sibilance']['candidate_intervals'], times + sec))),
                'within_tolerance': bool(abs(new['dynamic_range_db'] - base_new['dynamic_range_db']) <= .1)}
        for db in (-6, -12):
            changed = analyze(x * 10 ** (db / 20), sr)
            result['variations'][f'gain_{db}dB'] = {
                'level_shift_error_db': changed['rms_dbfs']['p50'] - base_new['rms_dbfs']['p50'] - db,
                'dynamic_delta_db': changed['dynamic_range_db'] - base_new['dynamic_range_db'],
                'active_delta_seconds': changed['active_seconds'] - base_new['active_seconds']}
        # Low-level nonzero room tone is intentionally not classified as silence.
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 10 ** (-65 / 20), sr * 5)
        noisy = analyze(np.r_[noise, x, noise], sr)
        result['variations']['nonzero_noise_minus65dbfs_5s_each_side'] = {
            'new_dynamic_delta_db': noisy['dynamic_range_db'] - base_new['dynamic_range_db'],
            'new_gate_delta_db': noisy['activity_gate_dbfs'] - base_new['activity_gate_dbfs'],
            'new_active_delta_seconds': noisy['active_seconds'] - base_new['active_seconds'],
            'interpretation': 'Exploratory artificial nonzero noise, not guaranteed ambient-noise behavior.'}
        if sha(path) != source_hash:
            raise RuntimeError('Source hash changed')
        out['cases'].append(result)
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
