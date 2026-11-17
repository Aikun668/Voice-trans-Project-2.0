"""Keep v001's resynthesis and repair only the two waveform joins."""

import importlib.util
import json
from pathlib import Path

import numpy as np


DEST = Path(__file__).resolve().parent
PREVIOUS = DEST.parent / "F2_XINQIANG_XUE_CONTOUR_2026-11-08_v001"
LEFT = (0.855, 0.875)
RIGHT = (1.086, 1.087)
LEFT_PROBE = (0.862, 0.867)
OLD_RIGHT = (1.066, 1.086)
PCM16_STEP = 1 / 32768

spec = importlib.util.spec_from_file_location("previous_xue", PREVIOUS / "run.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def rms_db(signal):
    return float(20 * np.log10(np.sqrt(np.mean(signal ** 2))))


def crossfade(source, synthetic, sr):
    slices = [previous.span_slice(span, sr) for span in (LEFT, RIGHT)]
    for s in slices:
        if np.max(np.abs(source[s] - synthetic[s])) > PCM16_STEP:
            raise RuntimeError("接缝波形相差超过一个 PCM16 量化步长")
    left, right = slices
    output = source.copy()
    output[left.stop:right.start] = synthetic[left.stop:right.start]
    for s, direction in zip(slices, (1, -1), strict=True):
        weight = np.sin(np.linspace(0, 0.5 * np.pi, s.stop - s.start)) ** 2
        weight = weight[::direction]
        output[s] += weight * (synthetic[s] - source[s])
    return output


def main():
    audio = DEST / "audio"
    report_path = DEST / "report.json"
    if audio.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    old_report = json.loads((PREVIOUS / "report.json").read_text(encoding="utf-8"))
    approved = json.loads((previous.APPROVED / "report.json").read_text(encoding="utf-8"))
    source_path = previous.f2.seg_path(previous.SEGMENT)
    source, sr = previous.read_wav_pcm16(source_path)
    hashes = {path: previous.digest(Path(path)) for path in old_report["inputs"]}
    if hashes != old_report["inputs"] or hashes != approved["inputs"]:
        raise RuntimeError("源文件哈希与 v001 不一致")

    times, cents = previous.pitch(source, sr)
    body = (times >= previous.SOURCE_BODY[0]) & (times < previous.SOURCE_BODY[1]) & np.isfinite(cents)
    curve_times = (np.arange(int(np.ceil(len(source) / sr / previous.HOP))) + 0.5) * previous.HOP
    shape = np.interp(curve_times, times[body], cents[body])
    enter = np.clip((curve_times - 0.81) / (0.879 - 0.81), 0, 1)
    leave = np.clip((1.13 - curve_times) / (1.13 - 1.09), 0, 1)
    envelope = np.sin(0.5 * np.pi * np.minimum(enter, leave)) ** 2
    target = old_report["target_cents"]
    desired = target + old_report["retained_fraction_of_glide"] * (shape - old_report["source_median_cents"])
    curve = (desired - shape) * envelope
    synth = previous.shift_by_curve(source, sr, curve)
    old, _ = previous.splice(source, synth, sr, [(*previous.SPLICE_CORE, 0.0)])
    corrected = crossfade(source, synth, sr)

    cut = previous.span_slice(approved["take_preview"]["source_seconds"], sr)
    lead = round(approved["lead_silence_seconds"] * sr)
    tail = round(approved["compare_split_seconds"] * sr) - lead - (cut.stop - cut.start)
    edge = np.ones(cut.stop - cut.start)
    fade = round(sr * 0.005)
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    gain = 10 ** (approved["preview_gain_db"]["take.wav"] / 20)
    version_gain = 10 ** (old_report["corrected_preview_fixed_gain_db"] / 20)
    original, old_preview, new_preview = [np.pad(signal[cut] * edge, (lead, tail)) * gain
                                          for signal in (source, old, corrected)]
    old_preview *= version_gain
    new_preview *= version_gain
    files = {
        "previous.wav": old_preview,
        "corrected.wav": new_preview,
        "compare.wav": np.concatenate((old_preview, new_preview)),
        "original_to_new.wav": np.concatenate((original, new_preview)),
    }
    if any(not np.isfinite(y).all() or np.max(np.abs(y)) >= 1 for y in files.values()):
        raise RuntimeError("试听输出非有限或削波")
    audio.mkdir()
    levels, readback = {}, {}
    for name, samples in files.items():
        path = audio / name
        previous.write_wav_pcm16(path, samples, sr)
        y, rate = previous.read_wav_pcm16(path)
        if rate != sr or len(y) != len(samples) or np.max(np.abs(y - samples)) > 2 * PCM16_STEP:
            raise RuntimeError(f"回读与预期不符：{name}")
        lufs, dbtp = previous.f2.ebur128(path)
        levels[name] = {"samples": len(y), "sample_rate": rate, "lufs": lufs, "dbtp": dbtp, "sha256": previous.digest(path)}
        readback[name] = y

    original_pcm, _ = previous.read_wav_pcm16(PREVIOUS / "audio" / "original.wav")
    offset = cut.start / sr - lead / sr
    before = previous.pitch(original_pcm, sr, offset)
    old_pitch = previous.pitch(readback["previous.wav"], sr, offset)
    after = previous.pitch(readback["corrected.wav"], sr, offset)
    core = previous.SOURCE_CORE
    before_core, old_core, after_core = [previous.values(p, core) for p in (before, old_pitch, after)]
    core_times = before[0][(before[0] >= core[0]) & (before[0] < core[1])]
    commanded = np.interp(core_times, curve_times, curve)
    command_error = float(np.nanmedian(after_core - before_core - commanded))
    target_error = float(np.nanmedian(after_core) - target)
    va, vb = np.isfinite(before[1]), np.isfinite(after[1])
    overlap = float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb))
    sample_bounds = previous.span_slice((LEFT[0], RIGHT[1]), sr)
    core_samples = previous.span_slice(core, sr)
    left_probe = previous.span_slice(LEFT_PROBE, sr)
    old_right = previous.span_slice(OLD_RIGHT, sr)
    old_bump = rms_db(old[left_probe]) - rms_db(source[left_probe])
    new_bump = rms_db(corrected[left_probe]) - rms_db(source[left_probe])
    split = len(readback["previous.wav"])
    checks = {
        "inputs_unchanged": all(previous.digest(Path(p)) == h for p, h in hashes.items()),
        "v001_reproduced_exactly": levels["previous.wav"]["sha256"] == old_report["levels"]["corrected.wav"]["sha256"],
        "comparison_order_exact": bool(
            np.array_equal(readback["compare.wav"], np.concatenate((readback["previous.wav"], readback["corrected.wav"])))
            and np.array_equal(readback["original_to_new.wav"][:split], original_pcm)
            and np.array_equal(readback["original_to_new.wav"][split:], readback["corrected.wav"])
        ),
        "central_waveform_matches_v001": bool(np.array_equal(old[core_samples], corrected[core_samples])),
        "old_right_overlap_replaced_by_synthesis_only": bool(np.array_equal(corrected[old_right], synth[old_right])),
        "outside_splice_identical": bool(
            np.array_equal(source[:sample_bounds.start], corrected[:sample_bounds.start])
            and np.array_equal(source[sample_bounds.stop:], corrected[sample_bounds.stop:])
        ),
        "left_bump_removed": old_bump > 2 and abs(new_bump) < 0.1,
        "matched_loudness": max(
            abs(levels["corrected.wav"]["lufs"] - lufs)
            for lufs in (old_report["levels"]["original.wav"]["lufs"], levels["previous.wav"]["lufs"])
        ) <= 0.5,
        "true_peak_at_most_minus_1_dbtp": all(item["dbtp"] <= -1 for item in levels.values()),
        "pitch_target_and_median_command_error_within_5_cents": abs(target_error) < 5 and abs(command_error) < 5,
        "voiced_overlap_at_least_0_95": overlap >= 0.95,
    }
    report = {
        "source": previous.SEGMENT, "word": "学", "previous": str(PREVIOUS / "report.json"),
        "target_note": old_report["target_note"], "target_cents": target,
        "resynthesis": "Same Praat/Parselmouth and same time-varying correction curve as v001",
        "splice": "Amplitude-complementary raised-cosine fades, with the right fade moved past the mismatch",
        "left_fade_seconds": LEFT, "right_fade_seconds": RIGHT, "old_right_overlap_seconds": OLD_RIGHT,
        "left_rms_bump_db": {"span_seconds": LEFT_PROBE, "previous": old_bump, "revised": new_bump},
        "old_right_waveform_correlation": float(np.corrcoef(source[old_right], synth[old_right])[0, 1]),
        "new_right_max_sample_mismatch": float(np.max(np.abs((source - synth)[previous.span_slice(RIGHT, sr)]))),
        "central_synthesis_rms_difference_vs_source_db": rms_db(synth[core_samples]) - rms_db(source[core_samples]),
        "median_frame_command_error_cents": command_error, "target_error_cents": target_error,
        "core_median_difference_from_v001_cents": float(np.nanmedian(after_core - old_core)),
        "voiced_overlap": overlap, "compare_split_seconds": split / sr,
        "levels": levels, "source_hashes": hashes, "checks": checks,
        "research": {
            "crossfade": "https://www.dafx.de/paper-archive/details/gUO-U6xEDk0Z_m4S-PivJQ",
            "praat": "https://praat.org/manual/overlap-add.html",
        },
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
