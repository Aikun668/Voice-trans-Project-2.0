import copy
import numpy as np
import pytest
from f1_adaptive_v0.analyzer import analyze
from f1_adaptive_v0.decision_engine import decide, load_rules


def sample(dynamic=False, gain=1):
    sr = 24000
    t = np.arange(sr * 5) / sr
    amp = np.where(t < 3, .025, .5) if dynamic else .2
    return analyze(gain * amp * np.sin(2 * np.pi * 200 * t), sr)


def test_normal_audio_bypasses_and_uncertain_spectrum_does_not_trigger_eq():
    d = decide(sample(), load_rules())
    assert d['mode'] == 'bypass'
    assert all(not d[k]['enabled'] for k in ('eq', 'compressor', 'deesser'))


def test_dynamic_audio_triggers_bounded_gain_invariant_compression():
    a, b = [decide(sample(True, g), load_rules()) for g in (1, .25)]
    for d in (a, b):
        c = d['compressor']
        assert c['enabled']
        assert 1 <= c['ratio'] <= 2
        assert -60 <= c['threshold_db'] <= 0
        assert c['predicted_static_peak_reduction_db'] <= 4.0001
    assert a['compressor']['ratio'] == pytest.approx(b['compressor']['ratio'], abs=.01)
    assert a['compressor']['threshold_db'] - b['compressor']['threshold_db'] == pytest.approx(12.0412, abs=.1)


def test_disabled_effect_and_invalid_rules():
    rules = load_rules()
    rules['compressor']['enabled'] = False
    assert decide(sample(True), rules)['mode'] == 'bypass'
    rules = copy.deepcopy(rules)
    rules['compressor']['max_ratio'] = 99
    with pytest.raises(ValueError):
        decide(sample(True), rules)


def test_unity_ratio_rules_rejected_cleanly():
    rules = load_rules()
    rules['compressor'].update(min_ratio=1., max_ratio=1.)
    with pytest.raises(ValueError, match='ratio'):
        decide(sample(True), rules)
