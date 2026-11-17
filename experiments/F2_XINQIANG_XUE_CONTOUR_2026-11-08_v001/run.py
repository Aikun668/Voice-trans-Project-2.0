"""Use the matched reference to reduce the excess glide in “学”."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

DEST = Path(__file__).resolve().parent
APPROVED = DEST.parent / "F2_XINQIANG_WORD_CUT_2026-11-08_v001"
KEY_CHECK = DEST.parent / "F2_XINQIANG_ONE_NOTE_TUNE_2026-11-08_v001" / "report.json"
sys.path.insert(0, str(DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase2 import pitch_series
from poc_phase4b import splice
from poc_phase4c import HOP, shift_by_curve
from self_ref_align import note_name


SEGMENT = "心墙01-对照微微跑调"
SOURCE_BODY = (0.889, 1.082)
SOURCE_CORE = (0.900, 1.0615)
REFERENCE_CORE = (114.034, 114.144)
SPLICE_CORE = (0.915, 1.026)
AUTHORIZED_AUDIO = (0.854, 1.087)
RETAIN_GLIDE = 0.20
MAX_TRIAL_SHIFT = 700.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span_slice(span, sr):
    return slice(round(span[0] * sr), round(span[1] * sr))


def pitch(signal, sr, offset=0):
    times, hz = pitch_series(signal, sr)
    return times + offset, 1200 * np.log2(hz / 440.0)


def values(track, span):
    time, cents = track
    return cents[(time >= span[0]) & (time < span[1])]


def median(track, span):
    frames = values(track, span)
    if np.count_nonzero(np.isfinite(frames)) < 5:
        raise RuntimeError(f"缺少有效音高帧：{span}")
    return float(np.nanmedian(frames))


def grid(cents):
    return round(cents / 100) * 100


def peak_step(track, span):
    time, cents = track
    valid = np.isfinite(cents)
    inside = (time[:-1] >= span[0]) & (time[1:] < span[1]) & valid[:-1] & valid[1:]
    return float(np.max(np.abs(np.diff(cents)[inside])))


def measure_level(audio, sr):
    args = [
        "ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sr),
        "-ac", "1", "-i", "pipe:0", "-af", "ebur128", "-f", "null", "-",
    ]
    run = subprocess.run(args, input=audio.astype("<f4").tobytes(), capture_output=True, check=True)
    matches = re.findall(rb"I:\s*(-?[\d.]+)\s+LUFS", run.stderr)
    if not matches:
        raise RuntimeError("无法测量试听片段的响度")
    return float(matches[-1])


def main():
    out = DEST / "audio"
    report_path = DEST / "report.json"
    if out.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")

    approved = json.loads((APPROVED / "report.json").read_text(encoding="utf-8"))
    source_path, ref_path = f2.seg_path(SEGMENT), f2.ref_path("心墙")
    hashes = {str(path): digest(path) for path in (source_path, ref_path)}
    key_check = json.loads(KEY_CHECK.read_text(encoding="utf-8"))
    if approved["inputs"] != hashes or key_check["inputs"] != hashes:
        raise RuntimeError("原始素材与已确认切口的记录不一致")
    source, sr = read_wav_pcm16(source_path)
    reference, ref_sr = read_wav_pcm16(ref_path)
    source_track = pitch(source, sr)
    ref_window = (113.55, 117.25)
    ref_track = pitch(reference[span_slice(ref_window, ref_sr)], ref_sr, ref_window[0])

    key = key_check["majority_semitone_offset_cents"]
    tuning = key_check["local_tuning_cents"]
    ref_cents = median(ref_track, REFERENCE_CORE)
    ref_frames = values(ref_track, REFERENCE_CORE)
    if np.nanpercentile(ref_frames, 90) - np.nanpercentile(ref_frames, 10) > 50:
        raise RuntimeError("原唱“学”的稳定部分音高离散过大")
    before_cents = median(source_track, SOURCE_CORE)
    target = grid(ref_cents) + key + tuning
    times, cents = source_track
    voiced_body = (times >= SOURCE_BODY[0]) & (times < SOURCE_BODY[1]) & np.isfinite(cents)
    curve_times = (np.arange(int(np.ceil(len(source) / sr / HOP))) + 0.5) * HOP
    source_shape = np.interp(curve_times, times[voiced_body], cents[voiced_body])
    enter = np.clip((curve_times - 0.81) / (0.879 - 0.81), 0, 1)
    leave = np.clip((1.13 - curve_times) / (1.13 - 1.09), 0, 1)
    envelope = np.sin(0.5 * np.pi * np.minimum(enter, leave)) ** 2
    desired_shape = target + RETAIN_GLIDE * (source_shape - before_cents)
    curve = (desired_shape - source_shape) * envelope
    max_shift = float(np.max(np.abs(curve)))
    if max_shift > MAX_TRIAL_SHIFT:
        raise RuntimeError(f"“学”的峰值修正量超出单字试验范围：{max_shift}")
    corrected, mask = splice(source, shift_by_curve(source, sr, curve), sr, [(*SPLICE_CORE, 0.0)])
    if corrected.shape != source.shape or not np.isfinite(corrected).all():
        raise RuntimeError("重合成改变了样本数或产生非有限值")
    allowed = span_slice(AUTHORIZED_AUDIO, sr)
    changed = np.flatnonzero(mask > 0)
    if not len(changed) or changed[0] < allowed.start or changed[-1] >= allowed.stop:
        raise RuntimeError("实际拼接越过“学”的范围")

    preview = approved["take_preview"]["source_seconds"]
    cut = span_slice(preview, sr)
    duration = approved["compare_split_seconds"]
    lead = round(approved["lead_silence_seconds"] * sr)
    tail = round(duration * sr) - lead - (cut.stop - cut.start)
    edge = np.ones(cut.stop - cut.start)
    fade = round(sr * 0.005)
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    gain = 10 ** (approved["preview_gain_db"]["take.wav"] / 20)
    original, fixed = [np.pad(signal[cut] * edge, (lead, tail)) * gain for signal in (source, corrected)]
    corrected_gain_db = measure_level(original, sr) - measure_level(fixed, sr)
    corrected_gain = 10 ** (corrected_gain_db / 20)
    fixed *= corrected_gain
    variants = {"original.wav": original, "corrected.wav": fixed, "compare.wav": np.concatenate((original, fixed))}
    if any(not np.isfinite(x).all() or np.max(np.abs(x)) >= 1 for x in variants.values()):
        raise RuntimeError("试听音频非有限或削波")

    out.mkdir()
    readback, levels = {}, {}
    for name, samples in variants.items():
        path = out / name
        write_wav_pcm16(path, samples, sr)
        signal, rate = read_wav_pcm16(path)
        if len(signal) != len(samples) or rate != sr or np.max(np.abs(signal - samples)) > 2 / 32768:
            raise RuntimeError(f"WAV 回读与预期不符：{name}")
        lufs, dbtp = f2.ebur128(path)
        readback[name] = signal
        levels[name] = {"samples": len(signal), "sample_rate": rate, "lufs": lufs, "dbtp": dbtp, "sha256": digest(path)}

    a, b, ab = (readback[name] for name in variants)
    offset = preview[0] - lead / sr
    before, after = [pitch(signal, sr, offset) for signal in (a, b)]
    before_core, after_core = [values(track, SOURCE_CORE) for track in (before, after)]
    core_times = before[0][(before[0] >= SOURCE_CORE[0]) & (before[0] < SOURCE_CORE[1])]
    commanded = np.interp(core_times, curve_times, curve)
    measured = after_core - before_core
    median_error = float(np.nanmedian(measured - commanded))
    readback_pitch = median(after, SOURCE_CORE)
    va, vb = [np.isfinite(track[1]) for track in (before, after)]
    overlap = float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb))
    peak_before, peak_after = [peak_step(track, AUTHORIZED_AUDIO) for track in (before, after)]
    ranges_hz = [np.nanpercentile(440 * 2 ** (core / 1200), [10, 90]).tolist() for core in (before_core, after_core)]
    spread_before, spread_after = [float(np.ptp(np.nanpercentile(core, [10, 90]))) for core in (before_core, after_core)]
    checks = {
        "inputs_unchanged": all(digest(Path(p)) == h for p, h in hashes.items()),
        "anchor_matches_approved_cut": levels["original.wav"]["sha256"] == approved["levels"]["take.wav"]["sha256"],
        "exact_original_then_corrected": bool(np.array_equal(ab, np.concatenate((a, b)))),
        "outside_splice_identical_before_preview_gain": bool(np.array_equal(corrected[mask == 0], source[mask == 0])),
        "matched_loudness": abs(levels["original.wav"]["lufs"] - levels["corrected.wav"]["lufs"]) <= 0.5,
        "true_peak_at_most_minus_1_dbtp": all(item["dbtp"] <= -1 for item in levels.values()),
        "pitch_target_and_median_command_error_within_5_cents": abs(readback_pitch - target) < 5 and abs(median_error) < 5,
        "voiced_glide_reduced": spread_after <= 0.35 * spread_before,
        "voiced_overlap_at_least_0_95": overlap >= 0.95,
        "largest_word_pitch_step_not_increased": peak_after <= peak_before + 5,
    }
    report = {
        "source": SEGMENT, "word": "学", "approved_cut": str(APPROVED / "report.json"),
        "key_check": str(KEY_CHECK), "majority_semitone_offset_cents": key, "local_tuning_cents": tuning,
        "reference_core_seconds": REFERENCE_CORE, "reference_note": note_name(ref_cents),
        "source_core_seconds": SOURCE_CORE, "source_median_cents": before_cents,
        "target_note": note_name(target), "target_cents": target,
        "planned_core_median_shift_cents": float(np.median(commanded)),
        "measured_core_median_shift_cents": float(np.nanmedian(measured)),
        "median_frame_command_error_cents": median_error, "readback_core_cents": readback_pitch,
        "peak_commanded_shift_cents": max_shift, "local_trial_shift_limit_cents": MAX_TRIAL_SHIFT,
        "original_core_hz_p10_p90": ranges_hz[0], "corrected_core_hz_p10_p90": ranges_hz[1],
        "retained_fraction_of_glide": RETAIN_GLIDE,
        "affected_audio_seconds": [changed[0] / sr, (changed[-1] + 1) / sr],
        "processing": "One pass with poc_phase4c.shift_by_curve (Praat), with local crossfade via poc_phase4b.splice",
        "corrected_preview_fixed_gain_db": corrected_gain_db,
        "trained_f2_model_used": False,
        "peak_10ms_step_cents_in_word": {"original": peak_before, "corrected": peak_after},
        "voiced_overlap": overlap, "preview_source_seconds": preview, "compare_split_seconds": duration,
        "levels": levels, "inputs": hashes, "checks": checks,
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
