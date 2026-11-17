import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from octave_evidence import OctaveConfig, comb_score, fix_octaves  # noqa: E402

SR = 22050
HOP = 128


def harmonic_tone(f0, seconds=1.0, n_harm=10):
    t = np.arange(int(SR * seconds)) / SR
    return sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, n_harm + 1) if k * f0 < SR / 2)


def tracks(n_frames, hz):
    return np.full(n_frames, float(hz))


def frames_for(seconds=1.0):
    return int(SR * seconds) // HOP + 1


def test_comb_score_prefers_the_true_fundamental():
    w = harmonic_tone(300.0)
    assert comb_score(w, SR, 300.0, 0.5) > comb_score(w, SR, 150.0, 0.5) + 0.3
    assert comb_score(w, SR, 300.0, 0.5) > comb_score(w, SR, 600.0, 0.5)


def test_tracker_one_octave_low_is_fixed():
    w = harmonic_tone(300.0)
    n = frames_for()
    f0, ref = tracks(n, 300.0), tracks(n, 300.0)
    f0[40:60] = 150.0                       # tracker dropped an octave for 20 frames
    out, rep = fix_octaves(f0, ref, w, SR, HOP)
    assert np.allclose(out, 300.0)
    assert rep["frames_changed"] == 20 and rep["stretches"][0]["applied"]


def test_tracker_one_octave_high_is_fixed():
    w = harmonic_tone(150.0)
    n = frames_for()
    f0, ref = tracks(n, 150.0), tracks(n, 150.0)
    f0[40:60] = 300.0
    out, rep = fix_octaves(f0, ref, w, SR, HOP)
    assert np.allclose(out, 150.0) and rep["frames_changed"] == 20


def test_tracker_is_kept_when_the_waveform_says_it_is_right():
    w = harmonic_tone(150.0)                 # true pitch 150; the *reference* tracker is the one that is wrong
    n = frames_for()
    f0, ref = tracks(n, 150.0), tracks(n, 150.0)
    ref[40:60] = 300.0
    out, rep = fix_octaves(f0, ref, w, SR, HOP)
    assert np.array_equal(out, f0) and rep["frames_changed"] == 0
    assert rep["stretches"] and not rep["stretches"][0]["applied"]


def test_short_disagreement_is_ignored():
    w = harmonic_tone(300.0)
    n = frames_for()
    f0, ref = tracks(n, 300.0), tracks(n, 300.0)
    f0[40:42] = 150.0
    out, rep = fix_octaves(f0, ref, w, SR, HOP)
    assert np.array_equal(out, f0) and rep["stretches"] == []


def test_non_octave_disagreement_is_ignored():
    w = harmonic_tone(300.0)
    n = frames_for()
    f0, ref = tracks(n, 300.0), tracks(n, 300.0)
    ref[40:60] = 450.0                       # a fifth, not an octave
    out, rep = fix_octaves(f0, ref, w, SR, HOP)
    assert np.array_equal(out, f0) and rep["stretches"] == []


def test_small_advantage_does_not_overrule_the_tracker():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(SR)              # no harmonic structure at all
    n = frames_for()
    f0, ref = tracks(n, 200.0), tracks(n, 200.0)
    f0[40:60] = 100.0
    out, rep = fix_octaves(f0, ref, w, SR, HOP, OctaveConfig(margin=5.0))
    assert np.array_equal(out, f0) and rep["frames_changed"] == 0


def test_unvoiced_frames_stay_unvoiced_and_no_mutation():
    w = harmonic_tone(300.0)
    n = frames_for()
    f0, ref = tracks(n, 300.0), tracks(n, 300.0)
    f0[40:60] = 150.0
    f0[10:15] = 0.0
    ref[10:15] = 0.0
    before = f0.copy()
    out, _ = fix_octaves(f0, ref, w, SR, HOP)
    assert np.all(out[10:15] == 0.0) and np.array_equal(f0, before) and len(out) == len(f0)


def test_correction_extends_into_adjacent_wrong_frames_only():
    w = harmonic_tone(300.0)
    n = frames_for()
    f0, ref = tracks(n, 300.0), tracks(n, 300.0)
    f0[38:62] = 150.0                        # wrong run is longer than the flagged part
    ref[38:62] = 0.0
    ref[44:56] = 300.0                       # reference only voiced in the middle
    out, _ = fix_octaves(f0, ref, w, SR, HOP)
    assert np.allclose(out[38:62], 300.0)    # extension reaches the whole wrong run
    assert np.allclose(out[:38], 300.0) and np.allclose(out[62:], 300.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fix_octaves(np.zeros(10), np.zeros(9), np.zeros(SR), SR, HOP)
