"""Regression tests for using the approved pair and preserving the song split."""
import hashlib
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f2"))
from train_confirmed import frame_rows, read_clip, validate_split, word_bounds

def test_read_clip_uses_exact_approved_frames(tmp_path):
    samples = np.arange(600, dtype="<i2")
    wav = tmp_path / "source.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    entry = {"path": str(wav), "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
             "seconds": [.01, .035], "rate": 8000, "frames": [80, 280]}
    actual, rate = read_clip(entry)
    assert rate == 8000
    assert len(actual) == 200
    assert actual[0] == 80 / 32768
    assert actual[-1] == 279 / 32768
    with pytest.raises(ValueError, match="bounds"):
        read_clip({**entry, "frames": [81, 280]})

def test_word_probes_cannot_expand_approved_window():
    a = {"start": .79, "end": 1.04}
    b = {"start": .78, "end": 1.05}
    assert word_bounds(a, b, [.80, 1.00]) == (.80, 1.00)
    assert word_bounds(a, {"start": .9, "end": .9}, [.80, 1.00]) is None

def test_short_confirmed_glide_keeps_signed_time_varying_corrections():
    pair = {"id": 3, "song": "心墙", "split": "train", "self": {"seconds": [.785, 1.216]}}
    times = np.array([.91, .95, .99, 1.03])
    pitches = np.array([-700., -600., -500., -400.])
    rows = frame_rows(pair, "学", times, pitches, -550, -1000, .2)
    assert [row["t"] for row in rows] == [.91, .95, .99, 1.03]
    assert [row["target_shift"] for row in rows] == [-330., -410., -490., -570.]

def test_same_song_cannot_appear_in_training_and_holdout():
    with pytest.raises(ValueError, match="song"):
        validate_split([{"song": "心墙", "split": "train"}, {"song": "心墙", "split": "holdout"}])


def test_confirmed_last_edge_overrides_unstable_qwen_end():
    exact = {"start": 120.461, "end": 120.621}
    padded = {"start": 120.473, "end": 120.973}
    assert word_bounds(exact, padded, [120.061, 120.960], last=True) == (120.473, 120.960)
