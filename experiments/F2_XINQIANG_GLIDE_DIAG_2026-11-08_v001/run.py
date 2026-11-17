"""Compare glide retention and shift strength with the same WORLD adapter."""

import json
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory
import wave

import numpy as np


DEST = Path(__file__).resolve().parent
BASELINE = DEST.parent / "F2_XINQIANG_RESYNTH_DIAG_2026-11-08_v001"
sys.path.insert(0, str(BASELINE))
import probe


old = probe.old
ORDER = ("original.wav", "world.wav", "retain_glide.wav", "half_shift.wav")


def report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def main():
    audio = DEST / "audio"
    result_path = DEST / "report.json"
    if audio.exists() or result_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")

    baseline = report(BASELINE)
    prior_curve = report(probe.approved_splice.PREVIOUS)
    approved = report(old.APPROVED)
    target = prior_curve["target_cents"]
    inputs = baseline["inputs"] | {
        str(BASELINE / "audio" / name): baseline["levels"][name]["sha256"] for name in ORDER[:2]
    }
    if any(old.digest(Path(path)) != expected for path, expected in inputs.items()):
        raise RuntimeError("源文件哈希不符")

    source, sr = old.read_wav_pcm16(old.f2.seg_path(old.SEGMENT))
    anchor, _ = old.read_wav_pcm16(BASELINE / "audio" / ORDER[0])
    cut = old.span_slice(approved["take_preview"]["source_seconds"], sr)
    lead = round(approved["lead_silence_seconds"] * sr)
    tail = len(anchor) - lead - (cut.stop - cut.start)
    offset = (cut.start - lead) / sr
    t, original_pitch = old.pitch(anchor, sr, offset)
    core = (t >= old.SOURCE_CORE[0]) & (t < old.SOURCE_CORE[1])
    original_center = float(np.nanmedian(original_pitch[core]))
    full_shift = target - original_center
    grid, prior_command, source_track = probe.approved_curve(source, sr, prior_curve)
    # The baseline command factors into the target displacement and a raised-cosine gate.
    source_shape = np.interp(grid, *source_track)
    baseline_displacement = target + prior_curve["retained_fraction_of_glide"] * (
        source_shape - prior_curve["source_median_cents"]) - source_shape
    gate = np.divide(prior_command, baseline_displacement, out=np.zeros_like(grid),
                     where=baseline_displacement != 0)
    span = old.span_slice((probe.LEFT_FADE[0], probe.RIGHT_FADE[1]), sr)
    edge = np.ones(cut.stop - cut.start)
    fade = round(sr * 0.005)
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    target_lufs = np.mean([baseline["levels"][name]["lufs"] for name in ORDER[:2]])
    versions = {}
    levels = {}
    readback = {}
    commands = {ORDER[1]: prior_command, ORDER[2]: gate * full_shift, ORDER[3]: gate * full_shift / 2}
    centers = {ORDER[1]: target, ORDER[2]: target, ORDER[3]: original_center + full_shift / 2}
    calibration = baseline["versions"]["world.wav"]["engine_calibration_cents"]

    with TemporaryDirectory(prefix="qa_", dir=DEST) as directory:
        staging = Path(directory)
        for name in ORDER[:2]:
            copyfile(BASELINE / "audio" / name, staging / name)
        for name in ORDER[2:]:
            guide = (grid, commands[name], source_track)
            reconstructed = probe.synthesize_world(source, sr, guide, calibration_cents=calibration)
            y, core_gain = probe.stitch(source, reconstructed, sr)
            if not (np.array_equal(y[:span.start], source[:span.start])
                    and np.array_equal(y[span.stop:], source[span.stop:])):
                raise RuntimeError(f"改动超出拼接区：{name}")
            preview = np.pad(y[cut] * edge, (lead, tail))
            base_gain = approved["preview_gain_db"]["take.wav"]
            preview *= 10 ** (base_gain / 20)
            extra_gain = float(target_lufs - old.measure_level(preview, sr))
            preview *= 10 ** (extra_gain / 20)
            if not np.isfinite(preview).all() or np.max(np.abs(preview)) >= 1:
                raise RuntimeError(f"试听音频样本异常：{name}")
            old.write_wav_pcm16(staging / name, preview, sr)
            versions[name] = {"commanded_body_shift_cents": float(np.min(commands[name])),
                              "core_gain_db": core_gain, "fixed_preview_gain_db": base_gain + extra_gain,
                              "outside_splice_identical_before_preview_gain": True}

        with wave.open(str(staging / "compare.wav"), "wb") as joined:
            joined.setparams((1, 2, sr, 0, "NONE", "not compressed"))
            for name in ORDER:
                with wave.open(str(staging / name), "rb") as piece:
                    joined.writeframes(piece.readframes(len(anchor)))

        for name in (*ORDER, "compare.wav"):
            path = staging / name
            y, rate = old.read_wav_pcm16(path)
            lufs, dbtp = old.f2.ebur128(path)
            expected_count = len(anchor) * (len(ORDER) if name == "compare.wav" else 1)
            if (rate != sr or len(y) != expected_count or not np.isfinite(y).all()
                    or np.max(np.abs(y)) >= 32767 / 32768 or not np.isfinite([lufs, dbtp]).all() or dbtp > -1):
                raise RuntimeError(f"音频回读检查失败：{name}")
            readback[name] = y
            levels[name] = {"sample_rate": rate, "samples": len(y), "lufs": lufs, "dbtp": dbtp,
                            "sha256": old.digest(path)}

        original_range = np.nanpercentile(original_pitch[core], [10, 90])
        versions[ORDER[0]] = {"core_p10_p90_span_cents": float(np.ptp(original_range)),
                              "core_p10_p90_hz": (440 * 2 ** (original_range / 1200)).tolist()}
        for name in ORDER[1:]:
            _, actual = old.pitch(readback[name], sr, offset)
            before, after = np.isfinite(original_pitch), np.isfinite(actual)
            expected = np.interp(t, grid, commands[name])
            pitch_range = np.nanpercentile(actual[core], [10, 90])
            versions.setdefault(name, {}).update(
                target_center_cents=centers[name], target_center_error_cents=float(np.nanmedian(actual[core]) - centers[name]),
                median_command_error_cents=float(np.nanmedian((actual - original_pitch - expected)[core])),
                voiced_overlap=float(np.count_nonzero(before & after) / np.count_nonzero(before | after)),
                core_p10_p90_span_cents=float(np.ptp(pitch_range)),
                core_p10_p90_hz=(440 * 2 ** (pitch_range / 1200)).tolist(),
            )

        checks = {
            "source_hashes_unchanged": all(old.digest(Path(p)) == h for p, h in inputs.items()),
            "baseline_audio_bit_exact": all(levels[n]["sha256"] == baseline["levels"][n]["sha256"] for n in ORDER[:2]),
            "compare_pcm_exact": np.array_equal(readback["compare.wav"], np.concatenate([readback[n] for n in ORDER])),
            "sample_count_rate_and_true_peak_verified": True,
            "outside_splice_identical_before_preview_gain": True,
            "level_difference_at_most_0_5_lu": np.ptp([levels[n]["lufs"] for n in ORDER]) <= 0.5,
            "pitch_errors_under_5_cents": all(abs(versions[n][key]) < 5 for n in ORDER[1:]
                                              for key in ("median_command_error_cents", "target_center_error_cents")),
            "voiced_overlap_at_least_0_95": all(versions[n]["voiced_overlap"] >= 0.95 for n in ORDER[1:]),
            "new_versions_preserve_90_percent_of_original_glide": all(
                versions[n]["core_p10_p90_span_cents"] >= 0.9 * versions[ORDER[0]]["core_p10_p90_span_cents"]
                for n in ORDER[2:]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"工程检查未通过：{checks}")

        payload = {"source": old.SEGMENT, "word": "学", "core_seconds": old.SOURCE_CORE,
                   "engine": baseline["engines"]["world"], "engine_calibration_cents": calibration,
                   "left_fade_seconds": probe.LEFT_FADE, "right_fade_seconds": probe.RIGHT_FADE,
                   "compare_order": ORDER, "compare_split_seconds": len(anchor) / sr,
                   "levels": levels, "versions": versions, "inputs": inputs,
                   "checks": {k: bool(v) for k, v in checks.items()},
                   "engineering": "technical_pass", "listening": "pending_jade"}
        audio.mkdir()
        for path in staging.iterdir():
            path.replace(audio / path.name)
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"versions": versions, "checks": payload["checks"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
