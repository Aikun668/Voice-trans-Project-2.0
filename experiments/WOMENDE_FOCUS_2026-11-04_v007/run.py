"""F1 listening calibration. Run with .venv-f1-adaptive on Windows."""
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.append(str(ROOT / ".venv-f2/Lib/site-packages"))


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    stage, audio = HERE / "work_output", HERE / "audio"
    report_path = HERE / "report.json"
    if any(path.exists() for path in (stage, audio, report_path)):
        raise SystemExit("输出已存在，拒绝覆盖。")
    stage.mkdir()
    meter = load("calibration_meter", "experiments/F1_demo_chain_2026-11-02_v001/run.py")
    pitch = load("calibration_pitch", "experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py")
    recordings = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
    thin = recordings / "我们的歌-薄音版.wav"
    v1 = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v001"
    tilt = load("v001_tilt", "experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py")
    full = json.loads((v1 / "report.json").read_text(encoding="utf-8"))["c2"]
    zero = v1 / "work/thin_keyup.wav"
    source_hash = digest(zero)
    x, sr = sf.read(zero, dtype="float32")
    half_slope = full["slope_db_per_oct"] / 2
    tilt.MAX_GAIN_DB = full["max_gain_db"] / 2
    half = tilt.process(x, sr, *tilt.load_f0("thin_keyup"), half_slope)
    half_input = stage / "half_raw.wav"
    sf.write(half_input, half, sr, subtype="FLOAT")
    inputs = {
        "original": zero,
        "shift_only": half_input,
        "normal": v1 / "work/c2_keyup_tilt_raw.wav",
    }
    hashes = {name: digest(path) for name, path in inputs.items()}
    levels = {name: meter.ebur(path) for name, path in inputs.items()}
    # Lower the common target if needed, keeping 0.3 dB of true-peak headroom.
    target = min(-18.5, *(lufs - peak - 1.3 for lufs, peak in levels.values()))
    rows = {}
    for name, source in inputs.items():
        original, rate = sf.read(source)
        gain_db = target - levels[name][0]
        out = stage / f"{name}.wav"
        sf.write(out, original * 10 ** (gain_db / 20), rate, subtype="PCM_16")
        readback, readback_rate = sf.read(out)
        lufs, peak = meter.ebur(out)
        pitch_from_input = pitch.pitch_delta(zero, out, 0)
        same_shape = original.shape == readback.shape and rate == readback_rate
        finite = bool(np.isfinite(readback).all())
        clipped = int(np.count_nonzero(np.abs(readback) >= 0.9999))
        cents = pitch_from_input["median_cents"]
        passed = (
            same_shape and finite and not clipped and abs(lufs - target) <= 0.3
            and peak <= -1.0 and cents is not None and abs(cents) < 5
            and pitch_from_input["voiced_overlap"] >= 0.95
        )
        rows[name] = {
            "input": ((HERE / "half_raw.wav") if source == half_input else source).relative_to(ROOT).as_posix(),
            "input_sha256": hashes[name],
            "input_hash_unchanged": digest(source) == hashes[name],
            "output_sha256": digest(out),
            "fixed_gain_db": round(gain_db, 2),
            "samples": len(readback),
            "sample_rate": rate,
            "duration_s": round(len(readback) / rate, 3),
            "sample_shape_unchanged": same_shape,
            "finite": finite,
            "clipped_samples": clipped,
            "lufs": lufs,
            "true_peak_dbtp": peak,
            "pitch_vs_shifted_base": pitch_from_input,
            "technical_pass": passed,
        }
    shift_check = pitch.pitch_delta(thin, stage / "original.wav", 310)
    shift_cents = shift_check["median_cents"]
    shift_pass = shift_cents is not None and abs(shift_cents - 310) < 5 and shift_check["voiced_overlap"] >= 0.95
    loudness = [row["lufs"] for row in rows.values()]
    spread = round(max(loudness) - min(loudness), 2)
    passed = digest(zero) == source_hash and spread <= 0.3 and shift_pass and all(
        row["technical_pass"] and row["input_hash_unchanged"] for row in rows.values()
    )
    report = {
        "purpose": "same-pitch harmonic compensation dose",
        "half_slope_db_per_oct": half_slope,
        "half_gain_cap_db": tilt.MAX_GAIN_DB,
        "full": full,
        "technical_pass": passed,
        "listening": "pending_jade",
        "common_target_lufs": round(target, 2),
        "measured_loudness_spread_lu": spread,
        "processing": "fixed gain only, no limiter",
        "shift_only_vs_thin": shift_check,
        "versions": rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("工程检查未通过，音频留在 work_output。")
    half_input.rename(HERE / "half_raw.wav")
    stage.rename(audio)


if __name__ == "__main__":
    main()
