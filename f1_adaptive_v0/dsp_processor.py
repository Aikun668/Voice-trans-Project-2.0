"""Thin wrapper around existing Pedalboard processors; no voice resynthesis."""
import numpy as np
from pedalboard import Pedalboard, Compressor, PeakFilter


def process(x, sr, decision, disabled=()):
    if set(disabled) - {'eq', 'compressor', 'deesser'}:
        raise ValueError('Unknown disabled effect')
    effects = []
    eq = decision['eq']
    if eq['enabled'] and 'eq' not in disabled:
        for p in eq.get('bands', []):
            if not (80 <= p['frequency_hz'] <= min(8000, sr * .45) and
                    -2 <= p['gain_db'] <= 2 and .3 <= p['q'] <= 2):
                raise ValueError('EQ parameters outside V0 safety bounds')
            effects.append(PeakFilter(cutoff_frequency_hz=p['frequency_hz'], gain_db=p['gain_db'], q=p['q']))
    c = decision['compressor']
    if c['enabled'] and 'compressor' not in disabled:
        if not (-60 <= c['threshold_db'] <= 0 and 1 <= c['ratio'] <= 2 and
                15 <= c['attack_ms'] <= 60 and 80 <= c['release_ms'] <= 400):
            raise ValueError('Compressor parameters outside V0 safety bounds')
        effects.append(Compressor(**{k: c[k] for k in ('threshold_db', 'ratio', 'attack_ms', 'release_ms')}))
    if decision['deesser']['enabled'] and 'deesser' not in disabled:
        raise ValueError('De-esser is not implemented/validated; no high-cut substitute.')
    if not effects:
        return np.array(x, dtype=float, copy=True)
    out = np.asarray(Pedalboard(effects)(np.asarray(x, dtype=np.float32), sr), dtype=float).reshape(-1)
    if len(out) != len(x) or not np.all(np.isfinite(out)):
        raise ValueError('DSP changed length or produced nonfinite audio')
    return out
