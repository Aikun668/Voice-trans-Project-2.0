"""Test the adapter's audible mechanism and level handling, not upstream internals."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def function():
    from f1_density import density
    return density


def tone():
    t = np.arange(44100 * 3) / 44100
    return 0.3 * np.sin(2 * np.pi * 180 * t)


def test_zero_mix_is_a_true_bypass():
    x = tone()
    y, meta = function()(x, 44100, drive_db=18, mix=0)
    assert np.array_equal(x, y)
    assert meta["bypass"]


def test_silence_has_no_non_finite_gain_or_output():
    y, _ = function()(np.zeros(44100), 44100, drive_db=18, mix=.65)
    assert np.array_equal(y, np.zeros(44100))


def test_input_level_adaptation_preserves_relative_processing_strength():
    x = tone()
    y, _ = function()(x, 44100, drive_db=18, mix=.65)
    quiet, _ = function()(x * .1, 44100, drive_db=18, mix=.65)
    assert np.allclose(quiet, y * .1, atol=1e-7)
    assert len(y) == len(x)


def test_processing_adds_harmonics_without_shifting_the_fundamental():
    x = tone()
    mild, _ = function()(x, 44100, drive_db=12, mix=.35)
    strong, _ = function()(x, 44100, drive_db=18, mix=.65)
    f = np.fft.rfftfreq(len(x), 1/44100)
    index = lambda hz: int(np.argmin(abs(f-hz)))
    spectra = [abs(np.fft.rfft(data)) for data in (x, mild, strong)]
    assert f[np.argmax(spectra[2])] == 180
    ratios = [s[index(540)] / s[index(180)] for s in spectra]
    assert ratios[1] > .001
    assert ratios[2] > ratios[1] * 2
    assert np.max(np.abs(strong)) < 1


@pytest.mark.parametrize("data,mix", [(np.array([np.nan, 0]), .5), (np.zeros((2, 1000)), .5), (tone(), 1.5)])
def test_invalid_audio_or_mix_is_rejected(data, mix):
    with pytest.raises(ValueError):
        function()(data, 44100, drive_db=18, mix=mix)
