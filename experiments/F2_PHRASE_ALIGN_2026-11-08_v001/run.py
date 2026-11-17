"""Prepare a two-second take/reference alignment comparison without retuning."""

import hashlib
import json
from math import gcd
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.signal import resample_poly

DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase2 import analyze_notes


SEGMENT = "莫问归期02-对照微微跑调"
PHRASE = "沉默有时"
REFERENCE_OCCURRENCE = "original_late"
ANCHOR_POSITION = 1.68
DURATION = 2.0
TARGET_LUFS = -18.5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sustained_note(audio, sr, probes):
    words = [probe["words"] for probe in probes]
    if any("".join(w["text"] for w in row[:4]) != PHRASE for row in words):
        raise RuntimeError("待对齐片段不对应标注歌词")
    starts = [row[3]["start"] for row in words]
    ends = [row[3]["end"] for row in words]
    if max(np.ptp(starts), np.ptp(ends)) > 0.08:
        raise RuntimeError("“时”的两次定位差异大于 80 毫秒")
    start, end = max(starts), min(ends)
    region = probes[0]["search_window_seconds"]
    offset, stop = (round(t * sr) for t in region)
    candidates = []
    for t0, t1, hz in analyze_notes(audio[offset:stop], sr):
        t0, t1 = t0 + offset / sr, t1 + offset / sr
        overlap = min(end, t1) - max(start, t0)
        cents = 1200 * np.log2(hz / 440.0)
        spread = float(np.ptp(np.percentile(cents, [10, 90])))
        if overlap >= 0.10 and t1 - t0 >= 0.15 and spread <= 120:
            candidates.append((overlap, t0, t1, float(np.median(cents)), spread))
    if not candidates:
        raise RuntimeError("“时”未找到稳定的持续唱音")
    _, t0, t1, cents, spread = max(candidates)
    return {
        "word_probe_seconds": list(zip(starts, ends, strict=True)),
        "sustained_seconds": [t0, t1],
        "anchor_seconds": (t0 + t1) / 2,
        "pitch_cents": cents,
        "pitch_spread_cents": spread,
    }


def crop_for_preview(audio, sr, out_sr, note, probes):
    start = round((note["anchor_seconds"] - ANCHOR_POSITION) * sr)
    end = start + round(DURATION * sr)
    first_word = min(row["words"][0]["start"] for row in probes)
    next_word = min(row["words"][4]["start"] for row in probes)
    if not (0 <= start < end <= len(audio) and start / sr <= first_word
            and note["sustained_seconds"][1] <= end / sr <= next_word):
        raise RuntimeError("试听截取未完整覆盖本句，或包含下一句")
    clip = audio[start:end]
    if sr != out_sr:
        divisor = gcd(sr, out_sr)
        clip = resample_poly(clip, out_sr // divisor, sr // divisor)
    if len(clip) != round(DURATION * out_sr):
        raise RuntimeError("试听片段长度不符")
    bounds = {"source_seconds": [start / sr, end / sr], "relative_anchor_seconds": note["anchor_seconds"] - start / sr}
    return clip, bounds


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
    alignment = json.loads((DEST / "alignment.json").read_text(encoding="utf-8"))
    take_path, ref_path = f2.seg_path(SEGMENT), f2.ref_path("莫问归期")
    inputs = {str(path): sha256(path) for path in (take_path, ref_path)}
    if inputs != alignment["inputs"]:
        raise RuntimeError("素材与逐字对齐时的哈希不一致")
    take, sr = read_wav_pcm16(take_path)
    reference, ref_sr = read_wav_pcm16(ref_path)
    probes = alignment["results"]
    notes = {
        name: sustained_note(take if name == "take" else reference, sr if name == "take" else ref_sr, rows)
        for name, rows in probes.items()
    }
    early_late_delta = abs(f2.circ(notes["original_late"]["pitch_cents"] - notes["original_early"]["pitch_cents"]))
    if early_late_delta > 20:
        raise RuntimeError("原唱两处对应音的音高不一致")

    take_clip, take_bounds = crop_for_preview(take, sr, sr, notes["take"], probes["take"])
    ref_clip, ref_bounds = crop_for_preview(
        reference, ref_sr, sr, notes[REFERENCE_OCCURRENCE], probes[REFERENCE_OCCURRENCE]
    )
    baselines = {"take.wav": take_clip, "reference.wav": ref_clip}
    fade = round(sr * 0.008)
    edge = np.ones(len(take_clip))
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    gained = {}
    gains_db = {}
    for name, clip in baselines.items():
        preview = clip * edge
        gain_db = TARGET_LUFS - measure_level(preview, sr)
        gains_db[name] = gain_db
        gained[name] = preview * 10 ** (gain_db / 20)
    headroom = min(1.0, 10 ** (-2 / 20) / max(np.max(np.abs(x)) for x in gained.values()))
    versions = {name: x * headroom for name, x in gained.items()}
    versions["compare.wav"] = np.concatenate((versions["take.wav"], versions["reference.wav"]))
    if any(not np.isfinite(x).all() or np.max(np.abs(x)) >= 1 for x in versions.values()):
        raise RuntimeError("试听文件有非有限值或削波")

    out.mkdir()
    levels = {}
    readback = {}
    for name, expected in versions.items():
        path = out / name
        write_wav_pcm16(path, expected, sr)
        signal, rate = read_wav_pcm16(path)
        if rate != sr or len(signal) != len(expected) or np.max(np.abs(signal - expected)) > 2 / 32768:
            raise RuntimeError(f"试听文件回读与预期不符：{name}")
        lufs, dbtp = f2.ebur128(path)
        levels[name] = {"samples": len(signal), "sample_rate": rate, "lufs": lufs, "dbtp": dbtp, "sha256": sha256(path)}
        readback[name] = signal

    pitch_checks = {}
    for name, baseline in baselines.items():
        original = f2.cents_of(baseline, sr)
        preview = f2.cents_of(readback[name], sr)
        va, vb = np.isfinite(original), np.isfinite(preview)
        pitch_checks[name] = {
            "median_shift_cents": float(np.median((preview - original)[va & vb])),
            "voiced_overlap": float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb)),
        }
    checks = {
        "inputs_unchanged": all(sha256(Path(p)) == h for p, h in inputs.items()),
        "anchor_match_within_input_sample": abs(take_bounds["relative_anchor_seconds"]
                                               - ref_bounds["relative_anchor_seconds"]) <= 1 / min(sr, ref_sr),
        "exact_take_then_reference": bool(np.array_equal(
            readback["compare.wav"], np.concatenate((readback["take.wav"], readback["reference.wav"]))
        )),
        "same_loudness_within_half_lu": abs(levels["take.wav"]["lufs"] - levels["reference.wav"]["lufs"]) <= 0.5,
        "true_peak_at_most_minus_1_dbtp": all(item["dbtp"] <= -1 for item in levels.values()),
        "pitch_preserved": all(
            abs(item["median_shift_cents"]) < 5 and item["voiced_overlap"] >= 0.95 for item in pitch_checks.values()
        ),
    }
    report = {
        "take": SEGMENT, "phrase": PHRASE, "word": "时", "reference_occurrence": REFERENCE_OCCURRENCE,
        "reference_occurrence_reason": "Both repeats have the same sustained note; retain the previous reference occurrence.",
        "anchors": notes, "reference_repeat_pitch_difference_cents": early_late_delta,
        "take_preview": take_bounds, "reference_preview": ref_bounds,
        "compare_split_seconds": DURATION,
        "audio_operations": "trim, 8 ms edge fades, fixed gain; reference preview resampled from 44.1 kHz to 48 kHz; concatenate",
        "preview_gain_db": {name: gain + 20 * np.log10(headroom) for name, gain in gains_db.items()},
        "levels": levels, "pitch_readback": pitch_checks, "inputs": inputs, "checks": checks,
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
