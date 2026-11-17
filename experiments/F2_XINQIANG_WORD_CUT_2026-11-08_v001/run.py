"""Cut the two confirmed words for a short take/reference audition."""

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


SEGMENT = "心墙01-对照微微跑调"
PHRASE = "学着"
MARKED_SPAN = (0.785, 1.216)
LEAD_SILENCE = 0.02
DURATION = 0.8
TARGET_LUFS = -18.5
# Refined from the confirmed Qwen windows: start at the high-frequency onset
# of “学”, then end at the level valley before the next word.
TAKE_CUT = (0.755, 1.415)
REFERENCE_CUT = (113.905, 114.455)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_positions(probes):
    words = [probe["words"][1:3] for probe in probes]
    if any("".join(w["text"] for w in row) != PHRASE for row in words):
        raise RuntimeError("待对齐片段不对应标注歌词")
    starts = [row[0]["start"] for row in words]
    ends = [row[-1]["end"] for row in words]
    if np.ptp(starts) > 0.12 or np.ptp(ends) > 0.12 or min(ends) - max(starts) < 0.15:
        raise RuntimeError(f"“学着”的定位不稳定：{list(zip(starts, ends, strict=True))}")
    return {
        "word_probes": words,
        "phrase_seconds": [min(starts), max(ends)],
        "anchor_seconds": float(np.median(starts)),
    }


def crop_for_preview(audio, sr, out_sr, note, cut_seconds):
    start, end = (round(t * sr) for t in cut_seconds)
    phrase_start, phrase_end = note["phrase_seconds"]
    if not (0 <= start < end <= len(audio) and abs(start / sr - phrase_start) <= 0.08
            and abs(end / sr - phrase_end) <= 0.08):
        raise RuntimeError("精切边界偏离逐字定位过大")
    clip = audio[start:end]
    if sr != out_sr:
        divisor = gcd(sr, out_sr)
        clip = resample_poly(clip, out_sr // divisor, sr // divisor)
    bounds = {"source_seconds": [start / sr, end / sr], "clip_seconds_before_padding": len(clip) / out_sr}
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
    alignment_path = DEST.parent / "F2_XINQIANG_ALIGN_2026-11-08_v001" / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    take_path, ref_path = f2.seg_path(SEGMENT), f2.ref_path("心墙")
    inputs = {str(path): sha256(path) for path in (take_path, ref_path)}
    if inputs != alignment["inputs"]:
        raise RuntimeError("素材与逐字对齐时的哈希不一致")
    take, sr = read_wav_pcm16(take_path)
    reference, ref_sr = read_wav_pcm16(ref_path)
    probes = alignment["results"]
    notes = {name: word_positions(rows) for name, rows in probes.items()}
    if not TAKE_CUT[0] <= MARKED_SPAN[0] < MARKED_SPAN[1] <= TAKE_CUT[1]:
        raise RuntimeError("精切没有覆盖原标注")
    take_clip, take_bounds = crop_for_preview(take, sr, sr, notes["take"], TAKE_CUT)
    ref_clip, ref_bounds = crop_for_preview(reference, ref_sr, sr, notes["reference"], REFERENCE_CUT)
    clips = {"take.wav": take_clip, "reference.wav": ref_clip}
    fade = round(sr * 0.005)
    lead, slot = round(sr * LEAD_SILENCE), round(sr * DURATION)
    baselines = {}
    gained = {}
    gains_db = {}
    for name, clip in clips.items():
        padding = (lead, slot - lead - len(clip))
        if padding[1] < 0:
            raise RuntimeError("试听片段超过播放槽长度")
        edge = np.ones(len(clip))
        edge[:fade] = np.linspace(0, 1, fade)
        edge[-fade:] = edge[:fade][::-1]
        baselines[name] = np.pad(clip, padding)
        preview = np.pad(clip * edge, padding)
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
        "clip_starts_at_same_slot_offset": all(not np.any(readback[name][:lead]) for name in clips),
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
        "take": SEGMENT, "phrase": PHRASE,
        "mark_source": "F2_R1_MARK_2026-11-07_v001/jade_fragments_原话.md, #3",
        "marked_span_seconds": MARKED_SPAN,
        "alignment_source": str(alignment_path),
        "boundary_basis": "Confirmed Qwen word windows, then high-frequency fricative onset and low-energy valley before next syllable",
        "anchors": notes,
        "take_preview": take_bounds, "reference_preview": ref_bounds,
        "compare_split_seconds": DURATION,
        "lead_silence_seconds": LEAD_SILENCE,
        "audio_operations": "trim, 5 ms edge fades, fixed gain, pad with silence; reference preview resampled from 44.1 kHz to 48 kHz; concatenate",
        "preview_gain_db": {name: gain + 20 * np.log10(headroom) for name, gain in gains_db.items()},
        "levels": levels, "pitch_readback": pitch_checks, "inputs": inputs, "checks": checks,
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
