"""A-only low-level audit; all variations stay in memory, no WAV export."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from f1_adaptive_v0.analyzer import analyze, read_wav

spec = importlib.util.spec_from_file_location('old_analyzer', HERE / 'analyzer_before.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    out = HERE / 'results_v001'
    out.mkdir(exist_ok=False)
    report = {'scope': 'A only; no processed audio or subjective judgment',
              'tolerance_db': .1, 'source_integrity': [],
              'old_code_sha256': sha(HERE / 'analyzer_before.py'),
              'new_code_sha256': sha(ROOT / 'f1_adaptive_v0/analyzer.py'), 'recordings': []}
    files = [('person_normal', ROOT / 'test_dataset/my_singing/shige_normal.wav'),
             ('person_off_pitch', ROOT / 'test_dataset/my_singing/shige_medium.wav'),
             ('network_normal', ROOT / 'test_dataset/web_singing/gtsinger_alto_control/告白气球_normal.wav')]
    for name, path in files:
        hash_before = sha(path)
        x, sr, _ = read_wav(path)
        baseline = analyze(x, sr)
        baseline_old = old.analyze(x, sr)
        save(out / f'{name}_features.json', baseline)
        save(out / f'{name}_previous_features.json', baseline_old)
        item = {'id': name, 'source': str(path), 'sha256': hash_before,
                'old_dynamic_db': baseline_old['dynamic_range_db'], 'new_dynamic_db': baseline['dynamic_range_db'],
                'old_gate_dbfs': baseline_old['activity_gate_dbfs'], 'new_gate_dbfs': baseline['activity_gate_dbfs'],
                'low_level_unknown_seconds': baseline['low_level']['seconds'],
                'reference_warnings': baseline['activity_assessment']['warnings'], 'variations': []}
        for seconds in (1, 5):
            for noise_db in (-65, -55, -45):
                rng = np.random.default_rng(42)
                noise = rng.normal(0, 10 ** (noise_db / 20), round(sr * seconds))
                z = np.r_[noise, x, noise]
                old_f, new_f = old.analyze(z, sr), analyze(z, sr)
                delta = new_f['dynamic_range_db'] - baseline['dynamic_range_db']
                item['variations'].append({'kind': 'nonzero_pause', 'seconds_each_side': seconds,
                    'noise_rms_dbfs': noise_db, 'below_new_gate_by_db': baseline['activity_gate_dbfs'] - noise_db,
                    'old_gate_delta_db': old_f['activity_gate_dbfs'] - baseline_old['activity_gate_dbfs'],
                    'new_gate_delta_db': new_f['activity_gate_dbfs'] - baseline['activity_gate_dbfs'],
                    'old_dynamic_delta_db': old_f['dynamic_range_db'] - baseline_old['dynamic_range_db'],
                    'new_dynamic_delta_db': delta,
                    'dynamic_within_diagnostic_tolerance': bool(abs(delta) <= .1),
                    'unknown_seconds': new_f['low_level']['seconds'],
                    'note': 'Noise above/near gate may legitimately enter level statistics; no promise of noise separation.'})
        for seconds in (1, 5, 10):
            f = analyze(np.pad(x, (sr * seconds, sr * seconds)), sr)
            item['variations'].append({'kind': 'digital_silence', 'seconds_each_side': seconds,
                'new_gate_delta_db': f['activity_gate_dbfs'] - baseline['activity_gate_dbfs'],
                'new_dynamic_delta_db': f['dynamic_range_db'] - baseline['dynamic_range_db']})
        for gain_db in (-6, -12):
            f = analyze(x * 10 ** (gain_db / 20), sr)
            item['variations'].append({'kind': 'gain', 'input_gain_db': gain_db,
                'new_gate_shift_error_db': f['activity_gate_dbfs'] - baseline['activity_gate_dbfs'] - gain_db,
                'new_dynamic_delta_db': f['dynamic_range_db'] - baseline['dynamic_range_db']})
        unchanged = sha(path) == hash_before
        if not unchanged:
            raise RuntimeError('Source changed')
        report['source_integrity'].append({'id': name, 'unchanged': unchanged})
        report['recordings'].append(item)
    save(out / 'comparison.json', report)
    for item in report['recordings']:
        low = [v for v in item['variations'] if v['kind'] == 'nonzero_pause' and v['noise_rms_dbfs'] == -65 and v['seconds_each_side'] == 5][0]
        print(item['id'], 'dynamic', round(item['old_dynamic_db'], 3), '->', round(item['new_dynamic_db'], 3),
              '-65dB_pause_delta', round(low['old_dynamic_delta_db'], 4), '->', round(low['new_dynamic_delta_db'], 4),
              'unknown_seconds', round(item['low_level_unknown_seconds'], 3))


if __name__ == '__main__':
    main()
