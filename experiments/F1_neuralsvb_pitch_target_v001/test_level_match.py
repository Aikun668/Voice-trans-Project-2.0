import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dip_repair import HOP, frame_db  # noqa: E402
from level_match import LevelConfig, level_match  # noqa: E402

SR = 22050


def tone_like(seconds=4.0, seed=0, level=0.1):
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal(int(SR * seconds))


def lvl(x):
    return 10 * np.log10(np.mean(np.asarray(x) ** 2) + 1e-12)


def test_identical_signals_are_unchanged():
    x = tone_like()
    y, rep = level_match(x, x.copy(), SR)
    assert np.allclose(y, x) and rep["max_boost_db"] == 0 and rep["max_cut_db"] == 0


def test_constant_gain_is_not_corrected():
    x = tone_like()
    out = x * 10 ** (-7 / 20)
    y, rep = level_match(x, out, SR)
    assert np.allclose(y, out, atol=1e-9)
    assert rep["baseline_offset_db"] == pytest.approx(7.0, abs=0.2)


def test_collapse_is_restored_to_the_inputs_contour():
    x = tone_like(seed=1)
    out = x * 10 ** (-4 / 20)
    lo, hi = int(1.5 * SR), int(1.9 * SR)
    out[lo:hi] *= 10 ** (-18 / 20)
    y, _ = level_match(x, out, SR)
    # the 2 dB dead zone is left to the model on purpose, so allow dead zone + smoothing slack
    assert abs(lvl(y[lo + 800:hi - 800]) - lvl(out[int(0.5 * SR):int(0.9 * SR)])) < 3.5
    assert lvl(y[lo + 800:hi - 800]) - lvl(out[lo + 800:hi - 800]) > 12.0   # most of the 18 dB collapse is undone


def test_boost_and_cut_are_limited():
    x = tone_like(seed=2)
    out = x.copy()
    out[int(1 * SR):int(1.5 * SR)] *= 10 ** (-40 / 20)
    out[int(2 * SR):int(2.5 * SR)] *= 10 ** (30 / 20)
    _, rep = level_match(x, out, SR, LevelConfig(max_boost_db=12.0, max_cut_db=8.0))
    assert rep["max_boost_db"] <= 12.0 + 1e-6 and rep["max_cut_db"] <= 8.0 + 1e-6


def test_small_differences_stay_inside_the_dead_zone():
    x = tone_like(seed=3)
    out = x.copy()
    out[int(1 * SR):int(2 * SR)] *= 10 ** (1.5 / 20)
    y, rep = level_match(x, out, SR)
    assert np.allclose(y, out, atol=1e-3)


def test_quiet_passage_is_left_alone():
    x = np.concatenate([tone_like(2.0, 4, 0.1), tone_like(1.0, 5, 0.0005)])
    out = x.copy()
    out[2 * SR:] *= 10 ** (-20 / 20)
    y, _ = level_match(x, out, SR)
    assert np.allclose(y[2 * SR + 1000:], out[2 * SR + 1000:], atol=1e-6)


def test_length_mismatch_and_no_mutation():
    x, o = tone_like(), tone_like(seed=9)
    before = o.copy()
    level_match(x, o, SR)
    assert np.array_equal(o, before)
    with pytest.raises(ValueError):
        level_match(x, o[:-1], SR)


def test_correction_curve_is_smooth():
    x = tone_like(seed=6)
    out = x.copy()
    out[int(1.5 * SR):int(1.9 * SR)] *= 10 ** (-18 / 20)
    y, _ = level_match(x, out, SR)
    ratio_db = 20 * np.log10((np.abs(y) + 1e-12) / (np.abs(out) + 1e-12))
    g = np.convolve(ratio_db, np.ones(256) / 256, mode="valid")
    assert np.max(np.abs(np.diff(g[::64]))) < 3.0    # no step larger than 3 dB between ~3 ms neighbours
