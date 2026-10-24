"""Exercise real audio processing and deliberately broken deliveries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_wav(path, samples, rate=44100):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes((np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())


@pytest.fixture
def recording(tmp_path):
    t = np.arange(44100 * 5) / 44100
    envelope = 0.35 + 0.25 * np.sin(2 * np.pi * 0.7 * t)
    samples = envelope * (np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 360 * t))
    path = tmp_path / "voice.wav"
    write_wav(path, samples)
    return path


def run_cli(source, destination):
    return subprocess.run(
        [sys.executable, str(ROOT / "f1_poc.py"), str(source), "--out", str(destination)],
        capture_output=True, text=True, encoding="utf-8", timeout=90,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_real_pipeline_keeps_pitch_time_source_and_matches_loudness(recording, tmp_path):
    # Catches clipping, trimming, pitch shifting, gain-only or loudness-biased output.
    before_hash = hashlib.sha256(recording.read_bytes()).hexdigest()
    out = tmp_path / "delivery"
    result = run_cli(recording, out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(recording.read_bytes()).hexdigest() == before_hash
    assert (out / "00_original.wav").read_bytes() == recording.read_bytes()
    report = json.loads((out / "qa_report.json").read_text(encoding="utf-8"))
    assert report["technical_pass"] is True
    assert report["listening_acceptance"] == "pending_jade"
    assert len(report["outputs"]) == 3
    loudness = [item["loudness"]["integrated_lufs"] for item in report["outputs"]]
    assert max(loudness) - min(loudness) <= 0.3
    for item in report["outputs"]:
        with wave.open(str(out / item["file"]), "rb") as stream:
            assert stream.getparams()[:4] == (1, 2, 44100, 220500)
            data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float)
        frequencies = np.fft.rfftfreq(len(data), 1 / 44100)
        peak = frequencies[np.argmax(np.abs(np.fft.rfft(data)))]
        assert abs(peak - 180) < 1
        assert item["loudness"]["true_peak_dbtp"] <= -2.9
        assert item["samples"]["clipped_samples"] == 0


def test_existing_delivery_is_not_overwritten(recording, tmp_path):
    # Catches accidental destruction when rerunning a previous experiment.
    out = tmp_path / "keep"
    out.mkdir()
    sentinel = out / "notes.txt"
    sentinel.write_text("keep my listening notes", encoding="utf-8")
    result = run_cli(recording, out)
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep my listening notes"
    assert list(out.iterdir()) == [sentinel]


def test_silent_input_is_rejected(tmp_path):
    # Catches invalid loudness math and delivery of an empty 'successful' result.
    source = tmp_path / "silent.wav"
    write_wav(source, np.zeros(44100 * 5))
    out = tmp_path / "delivery"
    result = run_cli(source, out)
    assert result.returncode != 0
    assert not out.exists()


def test_quality_gate_rejects_damaged_audio(recording):
    # Catches guards that report success regardless of actual damaged samples.
    from f1_poc import compare_audio, read_pcm16
    samples, rate = read_pcm16(recording)
    same = compare_audio(samples, samples.copy(), rate)
    assert same["passed"]
    damaged = [samples[:-1000], np.zeros_like(samples), np.clip(samples * 10, -1, 1)]
    shifted = np.concatenate([np.zeros(4410), samples[:-4410]])
    damaged.append(shifted)
    t = np.arange(len(samples)) / rate
    damaged.append(0.2 * np.sin(2 * np.pi * 220 * t))
    for candidate in damaged:
        assert not compare_audio(samples, candidate, rate)["passed"]


def test_gain_match_quantization_is_not_mistaken_for_high_frequency_damage(recording):
    # A near-empty frequency band changes in dB when PCM16 is rounded again.
    # This must not reject a correctly attenuated original.
    from f1_poc import compare_audio, read_pcm16
    samples, rate = read_pcm16(recording)
    rounded = np.round(samples * 0.49 * 32768) / 32768
    check = compare_audio(samples, rounded, rate)
    assert check["passed"], check


def test_v2_real_singing_has_tonal_separation_at_matched_loudness(tmp_path):
    # Regression for Jade's first-round feedback: valid but almost identical tone.
    # These measurements ensure deliberate tonal change, not perceived improvement.
    from f1_poc import experiment
    source = ROOT / "test_dataset/my_singing/molihua_normal.wav"
    if not source.exists():
        pytest.skip("Local singing fixture is unavailable")
    report = experiment(source, tmp_path / "v2", profile="v2")
    assert report["technical_pass"], report["errors"]
    before, light, strong = report["outputs"]
    mid = "mid_500_3000"
    light_change = light["comparison"]["relative_band_change_db"][mid]
    strong_change = strong["comparison"]["relative_band_change_db"][mid]
    assert light_change < -1
    assert strong_change < light_change - 1
    assert len({item["sha256"] for item in report["outputs"]}) == 3
    assert max(item["loudness"]["integrated_lufs"] for item in report["outputs"]) - min(
        item["loudness"]["integrated_lufs"] for item in report["outputs"]) <= 0.3
    assert (tmp_path / "v2/04_same_phrase_ABC.wav").exists()
