"""Render the matched Xue comparisons and measure the actual PCM16 outputs."""

import json
from pathlib import Path
from shutil import copyfile
import wave

import numpy as np

import probe


old = probe.old
prior = probe.approved_splice
VERSION_ORDER = ("original.wav", "previous.wav", "world.wav", "lpc.wav")


def read_report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def main():
    audio = probe.DEST / "audio"
    report_path = probe.DEST / "report.json"
    if audio.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")

    curve_report = read_report(prior.PREVIOUS)
    splice_report = read_report(probe.V002)
    cut_report = read_report(old.APPROVED)
    inputs = dict(cut_report["inputs"])
    original_path = prior.PREVIOUS / "audio" / "original.wav"
    previous_path = probe.V002 / "audio" / "corrected.wav"
    inputs[str(original_path)] = curve_report["levels"]["original.wav"]["sha256"]
    inputs[str(previous_path)] = splice_report["levels"]["corrected.wav"]["sha256"]
    if any(old.digest(Path(p)) != h for p, h in inputs.items()):
        raise RuntimeError("输入文件哈希不符")

    source, sr = old.read_wav_pcm16(old.f2.seg_path(old.SEGMENT))
    guide = probe.approved_curve(source, sr, curve_report)
    grid, correction, _ = guide
    target = curve_report["target_cents"]
    span = old.span_slice((probe.LEFT_FADE[0], probe.RIGHT_FADE[1]), sr)
    source_core = old.span_slice(old.SOURCE_CORE, sr)
    cut = old.span_slice(cut_report["take_preview"]["source_seconds"], sr)
    lead = round(cut_report["lead_silence_seconds"] * sr)
    length = round(cut_report["compare_split_seconds"] * sr)
    tail = length - lead - (cut.stop - cut.start)
    offset = (cut.start - lead) / sr
    edge = np.ones(cut.stop - cut.start)
    fade = round(sr * 0.005)
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    anchor_gain = cut_report["preview_gain_db"]["take.wav"]
    target_lufs = np.mean([curve_report["levels"]["original.wav"]["lufs"],
                           splice_report["levels"]["corrected.wav"]["lufs"]])

    versions = {}
    previews = {}
    for name, render, calibration in (("world.wav", probe.synthesize_world, -5.0),
                                       ("lpc.wav", probe.synthesize_lpc, -4.0)):
        synthetic = render(source, sr, guide, calibration_cents=calibration)
        y, core_gain = probe.stitch(source, synthetic, sr)
        if not (np.array_equal(source[:span.start], y[:span.start])
                and np.array_equal(source[span.stop:], y[span.stop:])):
            raise RuntimeError(f"拼接范围外的样本被改变：{name}")
        clip = np.pad(y[cut] * edge, (lead, tail)) * 10 ** (anchor_gain / 20)
        extra_gain = float(target_lufs - old.measure_level(clip, sr))
        preview = clip * 10 ** (extra_gain / 20)
        if len(preview) != length or not np.isfinite(preview).all() or np.max(np.abs(preview)) >= 1:
            raise RuntimeError(f"试听音频长度或峰值异常：{name}")
        previews[name] = preview
        versions[name] = {"engine_calibration_cents": calibration, "core_gain_db": core_gain,
                          "fixed_preview_gain_db": anchor_gain + extra_gain,
                          "outside_splice_identical": True}

    zero = old.shift_by_curve(source, sr, np.zeros_like(correction))
    shifted = old.shift_by_curve(source, sr, correction)
    core_rms = prior.rms_db(source[source_core])
    r3 = {name: probe.score(source, probe.stitch(source, probe.synthesize_rubberband(
              source, sr, guide, short_window=short), sr)[0], sr, guide, target)
          for name, short in (("standard", False), ("short_window", True))}

    audio.mkdir()
    copyfile(original_path, audio / VERSION_ORDER[0])
    copyfile(previous_path, audio / VERSION_ORDER[1])
    for name, clip in previews.items():
        old.write_wav_pcm16(audio / name, clip, sr)
    with wave.open(str(audio / "compare.wav"), "wb") as joined:
        joined.setparams((1, 2, sr, 0, "NONE", "not compressed"))
        for name in VERSION_ORDER:
            with wave.open(str(audio / name), "rb") as piece:
                joined.writeframes(piece.readframes(length))

    levels, readback = {}, {}
    for name in (*VERSION_ORDER, "compare.wav"):
        path = audio / name
        y, rate = old.read_wav_pcm16(path)
        lufs, dbtp = old.f2.ebur128(path)
        levels[name] = {"samples": len(y), "sample_rate": rate, "lufs": lufs, "dbtp": dbtp,
                        "sha256": old.digest(path)}
        if (rate != sr or len(y) != length * (len(VERSION_ORDER) if name == "compare.wav" else 1)
                or not np.isfinite(y).all() or np.max(np.abs(y)) >= 32767 / 32768 or dbtp > -1):
            raise RuntimeError(f"回读格式或真峰值不合格：{name}")
        readback[name] = y

    times, before = old.pitch(readback[VERSION_ORDER[0]], sr, offset)
    core = (times >= old.SOURCE_CORE[0]) & (times < old.SOURCE_CORE[1])
    voiced_before = np.isfinite(before)
    expected = np.interp(times, grid, correction)
    for name in VERSION_ORDER[1:]:
        _, after = old.pitch(readback[name], sr, offset)
        voiced_after = np.isfinite(after)
        delta = (after - before - expected)[core]
        overlap = np.count_nonzero(voiced_before & voiced_after) / np.count_nonzero(voiced_before | voiced_after)
        values = versions.setdefault(name, {})
        values.update(median_command_error_cents=float(np.nanmedian(delta)),
                      p90_absolute_command_error_cents=float(np.nanpercentile(np.abs(delta), 90)),
                      target_center_error_cents=float(np.nanmedian(after[core]) - target),
                      voiced_overlap=overlap)

    checks = {
        "inputs_unchanged": all(old.digest(Path(p)) == h for p, h in inputs.items()),
        "anchors_bit_exact": levels["original.wav"]["sha256"] == inputs[str(original_path)]
                             and levels["previous.wav"]["sha256"] == inputs[str(previous_path)],
        "comparison_order_exact": bool(np.array_equal(readback["compare.wav"], np.concatenate(
            [readback[name] for name in VERSION_ORDER]))),
        "pitch_errors_under_5_cents": all(abs(v["target_center_error_cents"]) < 5
                                          and abs(v["median_command_error_cents"]) < 5 for v in versions.values()),
        "voiced_overlap_at_least_0_95": all(v["voiced_overlap"] >= 0.95 for v in versions.values()),
        "matched_loudness_within_0_5_lu": np.ptp([levels[name]["lufs"] for name in VERSION_ORDER]) <= 0.5,
        "outside_splice_identical_before_preview_gain": all(versions[name]["outside_splice_identical"]
                                                             for name in previews),
        "sample_count_rate_and_true_peak_verified": True,
    }
    report = {
        "source": old.SEGMENT, "word": "学", "model_weights_used": False,
        "target_note": curve_report["target_note"], "target_cents": target,
        "pitch_curve_source": str(prior.PREVIOUS / "report.json"),
        "left_fade_seconds": probe.LEFT_FADE, "right_fade_seconds": probe.RIGHT_FADE,
        "core_seconds": old.SOURCE_CORE,
        "peak_shift_command_cents": float(np.max(np.abs(correction))),
        "psola_core_rms_change_before_splice_db": {
            "zero_shift": prior.rms_db(zero[source_core]) - core_rms,
            "approved_shift": prior.rms_db(shifted[source_core]) - core_rms,
        },
        "rubberband_r3_rejected_full_take_measurements": r3,
        "engines": {
            "world": {"adapter": "pyworld 0.3.5", "license": "MIT", "frame_period_ms": 2.5,
                      "source": "https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder",
                      "paper": "https://www.jstage.jst.go.jp/article/transinf/E99.D/7/E99.D_2015EDP7457/_article/-char/en"},
            "lpc": {"adapter": "Praat via Parselmouth", "version": old.f2.parselmouth.__version__,
                    "praat_version": old.f2.parselmouth.PRAAT_VERSION, "license": "GPL-3.0-or-later",
                    "source": "https://praat.org/manual/Manipulation.html"},
            "rubberband": {"adapter": "R3 CLI 4.0.0", "license": "GPL-2.0-or-later",
                           "source": "https://breakfastquay.com/rubberband/index.html",
                           "vendor_zip_sha256": "f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644"},
        },
        "compare_order": VERSION_ORDER, "compare_split_seconds": length / sr,
        "versions": versions, "levels": levels, "inputs": inputs,
        "checks": {name: bool(value) for name, value in checks.items()},
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if not all(checks.values()):
        raise SystemExit(f"工程检查未通过：{checks}")
    print(json.dumps({"versions": versions, "checks": report["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
