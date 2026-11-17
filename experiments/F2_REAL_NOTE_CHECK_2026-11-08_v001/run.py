"""Render one reference-guided edit and compare the saved model's decision."""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

DEST = Path(__file__).resolve().parent
TRAIN = DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"
sys.path.insert(0, str(TRAIN))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase2 import pitch_series


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(signal, sr):
    time, hz = pitch_series(signal, sr)
    return time, 1200 * np.log2(hz / 440.0)


def window(pitch, span):
    time, cents = pitch
    return cents[(time >= span[0]) & (time < span[1])]


def stable_pitch(pitch, span):
    frames = window(pitch, span)
    voiced = frames[np.isfinite(frames)]
    if len(voiced) < 8 or len(voiced) / len(frames) < 0.8:
        raise RuntimeError(f"稳定音高帧不足：{span}")
    return float(np.median(voiced)), float(np.ptp(np.percentile(voiced, [10, 90])))


def cut(span, sr):
    return slice(round(span[0] * sr), round(span[1] * sr))


def main():
    audio = DEST / "audio"
    report_path = DEST / "report.json"
    if audio.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    case = json.loads((DEST / "case.json").read_text(encoding="utf-8"))
    segment = case["segment"]
    src_path = f2.seg_path(segment)
    ref_path = f2.ref_path(f2.seg_song_take(segment)[0])
    paired_path = f2.seg_path(segment.replace("轻度跑调", "对照微微跑调"))
    weights = TRAIN / "results" / "model.pkl"
    inputs = {path: digest(path) for path in (src_path, ref_path, paired_path, weights)}
    source, sr = read_wav_pcm16(src_path)
    reference, ref_rate = read_wav_pcm16(ref_path)
    paired, pair_rate = read_wav_pcm16(paired_path)
    preview, edit = case["preview"], case["edit"]
    source_pitch = track(source, sr)
    ref_start = case["reference_analysis"][0]
    ref_clip = reference[cut(case["reference_preview"], ref_rate)]
    ref_time, ref_cents = track(reference[cut(case["reference_analysis"], ref_rate)], ref_rate)
    ref_pitch = (ref_time + ref_start, ref_cents)
    pair_pitch = track(paired, pair_rate)
    anchor_offsets = [
        f2.circ(stable_pitch(source_pitch, own)[0] - stable_pitch(ref_pitch, ref)[0])
        for own, ref in case["reference_key_anchors"]
    ]
    pair_offsets = [
        f2.circ(stable_pitch(source_pitch, own)[0] - stable_pitch(pair_pitch, other)[0])
        for own, other in case["paired_key_anchors"]
    ]
    key = float(np.median(anchor_offsets))
    pair_key = float(np.median(pair_offsets))
    sung, sung_spread = stable_pitch(source_pitch, case["source_core"])
    ref, ref_spread = stable_pitch(ref_pitch, case["reference_core"])
    pair, pair_spread = stable_pitch(pair_pitch, case["paired_core"])
    shift = round(-f2.circ(sung - ref - key), 1)
    pair_shift = round(-f2.circ(sung - pair - pair_key), 1)
    if max(sung_spread, ref_spread, pair_spread) > 80 or abs(shift - pair_shift) > 50:
        raise RuntimeError(f"参考不够稳定或两个目标分歧：{shift}, {pair_shift}")
    if not 60 <= abs(shift) <= f2.FIX_MAX_ABS:
        raise RuntimeError(f"目标改变量超出本轮范围：{shift}")

    core_start, core_end = case["source_core"]
    note = next(n for n in f2.notes_of(str(src_path)) if n["t0"] <= core_start and n["t1"] >= core_end)
    features = f2.feature_row({
        **note, "dur": round(note["t1"] - note["t0"], 3), "ref_c_global": ref + key,
        "ref_spread": ref_spread, "ref_voiced": round(float(np.mean(np.isfinite(window(ref_pitch, case["reference_core"])))), 2),
    })
    with weights.open("rb") as handle:
        model = pickle.load(handle)
    if model["features"] != f2.FEATURES:
        raise RuntimeError("权重特征顺序与代码不符")
    model_shift = float(model["model"].predict([features])[0])
    if not np.isfinite(model_shift):
        raise RuntimeError("模型预测为非有限值")
    with (TRAIN / "results" / "apply_notes.csv").open(encoding="utf-8", newline="") as handle:
        auto = next(row for row in csv.DictReader(handle) if row["segment"] == segment
                    and abs(float(row["t0"]) - note["t0"]) < 0.005)

    rendered, mask = f2.corrected_audio(source, sr, [(*edit, shift)])
    if rendered.shape != source.shape:
        raise RuntimeError("处理改变了录音长度")
    for span in case["protected_source_spans"]:
        if np.any(mask[cut(span, sr)] != 0):
            raise RuntimeError(f"修正区碰到了保护范围：{span}")
    part = cut(preview, sr)
    original, corrected = source[part], rendered[part]
    gain = f2.gain_match(original, corrected)
    versions = {
        "original.wav": (original * gain, sr),
        "corrected.wav": (corrected * gain, sr),
        "reference.wav": (ref_clip * f2.gain_match(ref_clip), ref_rate),
    }
    if any(not np.isfinite(y).all() or np.max(np.abs(y)) >= 1 for y, _ in versions.values()):
        raise RuntimeError("输出有非有限值或削波")
    audio.mkdir()
    levels, readback = {}, {}
    for name, (y, rate) in versions.items():
        path = audio / name
        write_wav_pcm16(path, y, rate)
        x, actual_rate = read_wav_pcm16(path)
        if len(x) != len(y) or actual_rate != rate or np.max(np.abs(x)) >= 1:
            raise RuntimeError(f"WAV 回读失败：{name}")
        loudness, peak = f2.ebur128(path)
        levels[name] = {"samples": len(x), "sample_rate": rate, "lufs": loudness, "dbtp": peak, "sha256": digest(path)}
        readback[name] = x
    raw, fixed = (readback[name] for name in ("original.wav", "corrected.wav"))
    before, after = track(raw, sr), track(fixed, sr)
    core = (core_start - preview[0], core_end - preview[0])
    measured = float(np.nanmedian(window(after, core) - window(before, core)))
    va, vb = np.isfinite(before[1]), np.isfinite(after[1])
    voiced_overlap = float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb))
    outside = mask[part] == 0
    checks = {
        "input_hashes_unchanged": all(digest(path) == value for path, value in inputs.items()),
        "outside_splice_identical": bool(np.array_equal(raw[outside], fixed[outside])),
        "true_peak_at_most_minus_1_dbtp": all(item["dbtp"] <= -1 for item in levels.values()),
        "loudness_matched": abs(levels["original.wav"]["lufs"] - levels["corrected.wav"]["lufs"]) <= 0.5,
        "voiced_overlap_at_least_0_95": voiced_overlap >= 0.95,
        "measured_shift_within_5_cents": abs(measured - shift) < 5,
    }
    report = {
        "case": case, "input_sha256": {str(p): h for p, h in inputs.items()},
        "key_offsets_cents": anchor_offsets, "paired_key_offsets_cents": pair_offsets,
        "reference_based_shift_cents": shift, "paired_take_crosscheck_shift_cents": pair_shift,
        "aligned_raw_model_shift_cents": round(model_shift, 3), "prior_automatic_decision_for_note": auto,
        "source_in_first_round_training": segment in f2.TRAIN_SEGS,
        "one_pass_measured_shift_cents": round(measured, 3), "voiced_overlap": voiced_overlap,
        "file_levels": levels, "checks": checks,
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail", "listening": "pending_jade",
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
