import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from f1_adaptive_v0.analyzer import read_wav
from f1_adaptive_v0.decision_engine import fixed_baseline
from f1_adaptive_v0.dsp_processor import process


def test_dsp_bypass_bounds_and_frequency_preservation():
    sr = 24000
    t = np.arange(sr * 4) / sr
    x = .6 * np.sin(2 * np.pi * 200 * t)
    d = fixed_baseline()
    assert np.array_equal(process(x, sr, d, ['compressor']), x)
    y = process(x, sr, d)
    assert len(y) == len(x)
    hz = np.fft.rfftfreq(len(y), 1 / sr)
    assert hz[np.argmax(abs(np.fft.rfft(y))) ] == 200
    unsafe = copy.deepcopy(d)
    unsafe['compressor']['ratio'] = 10
    with pytest.raises(ValueError):
        process(x, sr, unsafe)
    unsafe = copy.deepcopy(d)
    unsafe['deesser']['enabled'] = True
    with pytest.raises(ValueError):
        process(x, sr, unsafe)


def test_read_contract(tmp_path):
    for subtype in ('PCM_16', 'PCM_24'):
        p = tmp_path / f'{subtype}.wav'
        sf.write(p, np.zeros(96000), 24000, subtype=subtype)
        x, sr, _ = read_wav(p)
        assert len(x) == 96000 and sr == 24000
    p = tmp_path / 'stereo.wav'
    sf.write(p, np.zeros((96000, 2)), 24000)
    with pytest.raises(ValueError):
        read_wav(p)


def test_adaptive_threshold_actually_reduces_sustained_loud_audio():
    from f1_adaptive_v0.analyzer import analyze
    from f1_adaptive_v0.decision_engine import decide, load_rules
    sr = 24000
    t = np.arange(sr * 5) / sr
    x = np.where(t < 3, .025, .5) * np.sin(2 * np.pi * 200 * t)
    d = decide(analyze(x, sr), load_rules())
    y = process(x, sr, d)
    # A trigger must yield measurable sustained reduction, not only float rounding.
    region = t > 4
    reduction = 20 * np.log10(np.sqrt(np.mean(x[region] ** 2) / np.mean(y[region] ** 2)))
    assert .3 < reduction <= 4.1


def test_real_trigger_is_not_a_noop():
    from f1_adaptive_v0.decision_engine import decide, load_rules
    root = Path(__file__).resolve().parents[2]
    path = root / 'test_dataset/my_singing/molihua_normal.wav'
    snapshot = root / 'f1_adaptive_v0/outputs/2026-10-27-molihua-v002/features.json'
    if not path.exists() or not snapshot.exists():
        pytest.skip('Historical personal recording/features unavailable')
    x, sr, _ = read_wav(path)
    # Reproduce the original DSP threshold defect using its frozen features.
    # Improved A may legitimately bypass this recording; do not force a trigger.
    d = decide(json.loads(snapshot.read_text(encoding='utf-8')), load_rules())
    y = process(x, sr, d)
    assert d['compressor']['enabled']
    assert 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(y)) > .05


def test_real_export_loudness_hash_and_overwrite(tmp_path):
    from f1_adaptive_v0.run import run
    sr = 24000
    t = np.arange(sr * 4) / sr
    x = .2 * np.sin(2 * np.pi * 200 * t)
    source = tmp_path / 'source.wav'
    sf.write(source, x, sr, subtype='PCM_24')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / 'delivery'
    report = run(source, out)
    assert report['technical_pass'], report
    assert report['listening_acceptance'] == 'pending_jade'
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    files = [sf.read(out / p)[0] for p in ('original_matched.wav', 'fixed_baseline.wav', 'adaptive.wav')]
    assert all(len(y) == len(x) for y in files)
    assert np.array_equal(files[0], files[2])  # honest adaptive bypass
    loudness = [v['loudness']['integrated_lufs'] for v in report['outputs']]
    assert max(loudness) - min(loudness) <= .3
    assert all(v['loudness']['true_peak_dbtp'] <= -2.9 for v in report['outputs'])
    with pytest.raises(FileExistsError):
        run(source, out)
    assert json.loads((out / 'decision.json').read_text())['mode'] == 'bypass'


def test_silence_does_not_leave_a_delivery(tmp_path):
    from f1_adaptive_v0.run import run
    source = tmp_path / 'silent.wav'
    sf.write(source, np.zeros(96000), 24000)
    with pytest.raises(ValueError):
        run(source, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
