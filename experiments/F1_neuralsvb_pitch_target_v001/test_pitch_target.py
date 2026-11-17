import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_target import TargetConfig, make_target, with_smoothing  # noqa: E402


def steady(n=60, hz=220.0):
    return np.full(n, hz)


def test_clean_input_is_unchanged():
    f0 = np.concatenate([steady(40, 220), np.zeros(10), steady(40, 330)])
    out, rep = make_target(f0)
    assert np.allclose(out, f0)
    assert rep["octave_fixed_frames"] == rep["spike_frames"] == rep["gap_filled_frames"] == 0


def test_length_and_no_mutation():
    f0 = np.concatenate([steady(20), np.zeros(3), steady(20)])
    before = f0.copy()
    out, _ = make_target(f0)
    assert len(out) == len(f0)
    assert np.array_equal(f0, before)


def test_octave_error_is_fixed():
    f0 = steady(60, 220.0)
    f0[30:34] = 110.0
    out, rep = make_target(f0)
    assert np.allclose(out, 220.0, rtol=1e-6)
    assert rep["octave_fixed_frames"] == 4


def test_real_note_change_is_not_an_octave_error():
    f0 = np.concatenate([steady(30, 220.0), steady(30, 330.0)])  # a fifth up, a real melody step
    out, _ = make_target(f0)
    assert np.allclose(out, f0)


def test_short_dropout_is_filled_by_interpolation():
    f0 = steady(60, 220.0)
    f0[30:34] = 0.0
    out, rep = make_target(f0)
    assert rep["gap_filled_frames"] == 4
    assert np.allclose(out, 220.0)


def test_gap_not_filled_when_hint_says_unvoiced():
    f0 = steady(60, 220.0)
    f0[30:34] = 0.0
    hint = np.ones(60, dtype=bool)
    hint[30:34] = False
    out, rep = make_target(f0, voiced_hint=hint)
    assert rep["gap_filled_frames"] == 0
    assert np.all(out[30:34] == 0.0)


def test_gap_filled_when_hint_says_voiced():
    f0 = steady(60, 220.0)
    f0[30:34] = 0.0
    out, rep = make_target(f0, voiced_hint=np.ones(60, dtype=bool))
    assert rep["gap_filled_frames"] == 4 and np.all(out > 0)


def test_long_gap_and_edges_stay_unvoiced():
    f0 = np.concatenate([np.zeros(5), steady(30), np.zeros(20), steady(30), np.zeros(5)])
    out, rep = make_target(f0)
    assert rep["gap_filled_frames"] == 0
    assert np.array_equal(out > 0, f0 > 0)


def test_gap_across_a_big_pitch_jump_is_not_bridged():
    f0 = np.concatenate([steady(30, 220.0), np.zeros(4), steady(30, 440.0)])
    out, rep = make_target(f0)
    assert rep["gap_filled_frames"] == 0


def test_isolated_spike_is_interpolated():
    f0 = steady(60, 220.0)
    f0[30:32] = 330.0
    out, rep = make_target(f0)
    assert rep["spike_frames"] == 2
    assert np.allclose(out, 220.0)


def test_hint_length_mismatch_raises():
    with pytest.raises(ValueError):
        make_target(steady(10), voiced_hint=np.ones(9, dtype=bool))


def test_smoothing_is_off_by_default_and_optional():
    rng = np.random.default_rng(0)
    f0 = 220.0 * 2.0 ** (rng.normal(0, 10, 80) / 1200.0)
    same, _ = make_target(f0)
    assert np.allclose(same, f0)
    smooth, _ = make_target(f0, config=with_smoothing(TargetConfig(), 5))
    assert np.std(np.log2(smooth)) < np.std(np.log2(f0))


def test_deterministic():
    f0 = np.concatenate([steady(30), np.zeros(3), steady(30)])
    a, _ = make_target(f0)
    b, _ = make_target(f0)
    assert np.array_equal(a, b)
