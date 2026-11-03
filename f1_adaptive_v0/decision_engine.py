"""A pure features -> parameters interface, replaceable independently of DSP."""
from pathlib import Path
import numpy as np
import yaml


def load_rules(path=None):
    return yaml.safe_load(Path(path or Path(__file__).parent / 'config/rules.yaml').read_text(encoding='utf-8'))


def validate_rules(rules):
    c = rules['compressor']
    bounds = {'min_dynamic_range_db': (12, 30), 'min_active_seconds': (1, 10),
              'min_loud_event_seconds': (.02, 1), 'min_ratio': (1, 2), 'max_ratio': (1, 2),
              'max_static_reduction_db': (.1, 4), 'attack_ms': (15, 60), 'release_ms': (80, 400)}
    for key, (lo, hi) in bounds.items():
        v = c[key]
        if not np.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'Unsafe rule {key}: {v}, allowed [{lo}, {hi}]')
    if c['min_ratio'] > c['max_ratio']:
        raise ValueError('min_ratio exceeds max_ratio')
    if c['min_ratio'] <= 1:
        raise ValueError('Adaptive compressor ratio must be >1; use enabled:false for bypass.')
    if rules['eq']['enabled'] or rules['deesser']['enabled']:
        raise ValueError('Automatic EQ/de-essing is not calibrated in V0; cannot enable via config.')


def decide(features, rules):
    validate_rules(rules)
    c = rules['compressor']
    reasons = []
    checks = [('enabled', bool(c['enabled'])),
              ('dynamic_range', features['dynamic_range_db'] >= c['min_dynamic_range_db']),
              ('active_duration', features['active_seconds'] >= c['min_active_seconds']),
              ('sustained_loud_event', features['longest_loud_event_seconds'] >= c['min_loud_event_seconds']),
              ('unclipped_input', features['full_scale_samples'] == 0)]
    enabled = all(value for _, value in checks)
    reasons = [f'{name}: {"met" if passed else "not met"}' for name, passed in checks]
    result = {'schema_version': 'f1.decision.v0', 'rules_version': rules['version'],
              'mode': 'process' if enabled else 'bypass', 'confidence': 'heuristic_candidate',
              'compressor': {'enabled': enabled, 'reason': reasons},
              'eq': {'enabled': False, 'status': 'unknown', 'reason': rules['eq']['reason']},
              'deesser': {'enabled': False, 'status': 'unknown', 'reason': rules['deesser']['reason']},
              'listening_acceptance': 'pending_jade',
              'interpretation': 'Dynamic variation may be intentional; triggering is not evidence of a perceptual defect.'}
    if enabled:
        ratio = float(np.clip(1.3 + .06 * (features['dynamic_range_db'] - c['min_dynamic_range_db']), c['min_ratio'], c['max_ratio']))
        factor = 1 - 1 / ratio
        # Frame RMS is a better operating-level reference than frame maxima:
        # maxima set an ineffectively high threshold on peaky real vocals.
        # Retain the conservative sample-peak static-reduction upper bound.
        threshold = max(features['rms_dbfs']['p90'] - 3,
                        features['sample_peak_dbfs'] - c['max_static_reduction_db'] / factor)
        threshold = float(np.clip(threshold, -60, 0))
        result['compressor'].update(threshold_db=threshold, ratio=ratio,
                                    attack_ms=c['attack_ms'], release_ms=c['release_ms'],
                                    predicted_static_peak_reduction_db=max(0, features['sample_peak_dbfs'] - threshold) * factor)
    return result


def fixed_baseline():
    return {'schema_version': 'f1.decision.v0', 'mode': 'process',
            'compressor': {'enabled': True, 'threshold_db': -18., 'ratio': 1.5,
                           'attack_ms': 25., 'release_ms': 180.,
                           'reason': 'Predeclared fixed experimental comparator; not an accepted beauty preset.'},
            'eq': {'enabled': False}, 'deesser': {'enabled': False}}
