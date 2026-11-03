import numpy as np
import pytest
from pathlib import Path

from f1_adaptive_v0.analyzer import analyze


def tone(seconds=4, sr=24000):
    t = np.arange(round(seconds * sr)) / sr
    return .2 * np.sin(2 * np.pi * 200 * t), sr


def test_known_rms_and_silence_does_not_create_dynamics():
    x, sr = tone()
    f = analyze(x, sr)
    padded = analyze(np.r_[np.zeros(sr), x, np.zeros(sr)], sr)
    assert f['rms_dbfs']['p50'] == pytest.approx(20 * np.log10(.2 / np.sqrt(2)), abs=.1)
    assert padded['dynamic_range_db'] == pytest.approx(f['dynamic_range_db'], abs=.5)
    assert padded['active_seconds'] == pytest.approx(4, abs=.1)


def test_local_high_frequency_burst_is_candidate_not_diagnosis():
    x, sr = tone()
    t = np.arange(len(x)) / sr
    mask = (t >= 1) & (t < 1.15)
    x[mask] += .15 * np.sin(2 * np.pi * 6500 * t[mask])
    f = analyze(x, sr)
    assert f['sibilance']['status'] == 'unknown'
    assert any(.9 <= p['start_seconds'] <= 1.1 for p in f['sibilance']['candidate_intervals'])


def test_invalid_signal_is_rejected():
    for x in [np.zeros(96000), np.full(96000, np.nan), np.ones((96000, 2)), np.ones(20)]:
        with pytest.raises(ValueError):
            analyze(x, 24000)


def test_real_recording_stats_stay_stable_with_digital_silence():
    from f1_adaptive_v0.analyzer import read_wav
    path = Path(__file__).resolve().parents[2] / 'test_dataset/my_singing/shige_normal.wav'
    if not path.exists():
        pytest.skip('Personal recording unavailable')
    x, sr, _ = read_wav(path)
    baseline = analyze(x, sr)
    for seconds in (1, 5, 10):
        padded = analyze(np.pad(x, (seconds * sr, seconds * sr)), sr)
        assert padded['dynamic_range_db'] == pytest.approx(baseline['dynamic_range_db'], abs=.1)
        assert padded['activity_gate_dbfs'] == pytest.approx(baseline['activity_gate_dbfs'], abs=.1)


def test_soft_voice_and_fading_tail_remain_in_activity_with_padding():
    sr = 24000
    t = np.arange(sr * 5) / sr
    # Quiet tail reaches about 39 dB below the loud section; existing -40 dB
    # gate cannot promise to retain arbitrarily quiet audio below that range.
    envelope = np.where(t < 2.5, .4, np.where(t < 4, .012, .012 * np.exp(-(t - 4))))
    x = envelope * np.sin(2 * np.pi * 200 * t)
    x[(t >= 2.0) & (t < 2.3)] = 0  # internal breath pause, not activity
    base = analyze(x, sr)
    padded = analyze(np.pad(x, (5 * sr, 5 * sr)), sr)
    for time in (3.0, 4.5, 4.9):
        assert any(p['start_seconds'] <= time < p['end_seconds'] for p in base['active_intervals'])
        assert any(p['start_seconds'] <= time + 5 < p['end_seconds'] for p in padded['active_intervals'])
    assert not any(p['start_seconds'] <= 2.15 < p['end_seconds'] for p in base['active_intervals'])
    assert padded['dynamic_range_db'] == pytest.approx(base['dynamic_range_db'], abs=.1)


def test_activity_duration_never_exceeds_input_due_to_edge_padding():
    x, sr = tone()
    f = analyze(x, sr)
    padded = analyze(np.pad(x, (5 * sr, 5 * sr)), sr)
    assert f['active_seconds'] <= f['duration_seconds']
    assert padded['active_seconds'] == pytest.approx(f['active_seconds'], abs=.01)


def test_low_level_nonzero_pause_does_not_move_recording_gate():
    from f1_adaptive_v0.analyzer import read_wav
    path = Path(__file__).resolve().parents[2] / 'test_dataset/my_singing/shige_normal.wav'
    if not path.exists():
        pytest.skip('Personal recording unavailable')
    x, sr, _ = read_wav(path)
    baseline = analyze(x, sr)
    rng = np.random.default_rng(42)
    for seconds in (1, 5):
        noise = rng.normal(0, 10 ** (-65 / 20), sr * seconds)
        prefixed = analyze(np.r_[noise, x, noise], sr)
        assert prefixed['activity_gate_dbfs'] == pytest.approx(baseline['activity_gate_dbfs'], abs=.1)
        assert prefixed['dynamic_range_db'] == pytest.approx(baseline['dynamic_range_db'], abs=.1)


def test_very_soft_tail_retained_as_unknown_not_declared_silence():
    sr = 24000
    t = np.arange(sr * 5) / sr
    amp = np.where(t < 3, .4, .002)  # ~46dB down, below activity gate
    x = amp * np.sin(2 * np.pi * 200 * t)
    f = analyze(x, sr)
    assert not any(p['start_seconds'] <= 4 < p['end_seconds'] for p in f['active_intervals'])
    low = f['low_level']
    assert low['status'] == 'unknown'
    assert any(p['start_seconds'] <= 4 < p['end_seconds'] for p in low['candidate_intervals'])
    assert f['activity_assessment']['is_voice_detection'] is False


def test_isolated_loud_burst_warns_about_reference_risk():
    x, sr = tone()
    rng = np.random.default_rng(7)
    x[sr:sr + round(sr * .01)] = rng.uniform(-.95, .95, round(sr * .01))
    f = analyze(x, sr)
    assert 'sparse_loud_reference' in f['activity_assessment']['warnings']
