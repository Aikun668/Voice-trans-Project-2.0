"""Build a one-pass preview for one existing sung note, using verified lyric anchors."""

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


SONG = "离开我的依赖"
SEGMENT = SONG + "01-轻度跑调"
SOURCE_WORD = (8.40, 9.12)
REFERENCE_WORD = (62.68, 62.92)
SOURCE_CORE = (8.64, 9.02)
REFERENCE_CORE = (62.80, 62.92)
EDIT = (8.49, 9.20)
PREVIEW = (7.05, 11.05)
REFERENCE_PREVIEW = (61.55, 65.55)
ANCHORS = (
    ("你（前一句）", (6.24, 6.70), (61.13, 61.39)),
    ("身", (7.14, 7.50), (61.80, 62.20)),
    ("旁", (7.62, 8.15), (62.32, 62.56)),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pitch(signal, sr):
    times, hz = pitch_series(signal, sr)
    return times, 1200 * np.log2(hz / 440.0)


def select(track, interval):
    times, cents = track
    start, end = interval
    return cents[(times >= start) & (times < end)]


def median(track, interval):
    values = select(track, interval)
    if np.count_nonzero(np.isfinite(values)) < 6:
        raise RuntimeError(f"缺少可靠音高帧：{interval}")
    return float(np.nanmedian(values))


def sample_slice(interval, sr):
    return slice(*(round(t * sr) for t in interval))


def main():
    audio = DEST / "audio"
    report_path = DEST / "report.json"
    if audio.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    source_path, ref_path = f2.seg_path(SEGMENT), f2.ref_path(SONG)
    weights_path = TRAIN / "results" / "model.pkl"
    hashes = {path: digest(path) for path in (source_path, ref_path, weights_path)}
    source, sr = read_wav_pcm16(source_path)
    reference, ref_sr = read_wav_pcm16(ref_path)
    own_pitch = pitch(source, sr)
    ref_pitch = pitch(reference, ref_sr)
    anchor_rows = [
        {
            "word": word, "source_seconds": own, "reference_seconds": ref,
            "offset_cents": round(f2.circ(median(own_pitch, own) - median(ref_pitch, ref)), 3),
        }
        for word, own, ref in ANCHORS
    ]
    key = float(np.median([row["offset_cents"] for row in anchor_rows]))
    source_cents, ref_cents = median(own_pitch, SOURCE_CORE), median(ref_pitch, REFERENCE_CORE)
    shift = round(-f2.circ(source_cents - ref_cents - key), 1)
    if not 100 <= abs(shift) <= f2.FIX_MAX_ABS:
        raise RuntimeError(f"候选修正量不在本次试验范围：{shift}")

    note = min(f2.notes_of(str(source_path)), key=lambda n: abs(n["t0"] - SOURCE_CORE[0]))
    if not note["t0"] <= SOURCE_CORE[0] < SOURCE_CORE[1] <= note["t1"]:
        raise RuntimeError("模型音符与试听核心区不对应")
    ref_track = f2.cents_track(str(ref_path))
    first, last = (round(t / f2.HOP) for t in REFERENCE_CORE)
    ref_frames = ref_track[first:last + 1]
    voiced = ref_frames[np.isfinite(ref_frames)]
    if len(voiced) < 3:
        raise RuntimeError("模型参考特征缺少音高帧")
    p10, p90 = np.percentile(voiced, [10, 90])
    features = f2.feature_row({
        **note,
        "dur": round(note["t1"] - note["t0"], 3),
        "ref_c_global": ref_cents + key,
        "ref_spread": float(p90 - p10),
        "ref_voiced": round(len(voiced) / len(ref_frames), 2),
    })
    with weights_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle["features"] != f2.FEATURES:
        raise RuntimeError("模型特征顺序不匹配")
    model_shift = float(bundle["model"].predict([features])[0])
    if not np.isfinite(model_shift):
        raise RuntimeError("模型预测不是有限数值")

    processed, mask = f2.corrected_audio(source, sr, [(*EDIT, shift)])
    if processed.shape != source.shape:
        raise RuntimeError("修正改变了源音频长度")
    clip = sample_slice(PREVIEW, sr)
    original, corrected = source[clip], processed[clip]
    ref_clip = reference[sample_slice(REFERENCE_PREVIEW, ref_sr)]
    gain = f2.gain_match(original, corrected)
    variants = {
        "original.wav": (original * gain, sr),
        "corrected.wav": (corrected * gain, sr),
        "reference.wav": (ref_clip * f2.gain_match(ref_clip), ref_sr),
    }
    if any(not np.isfinite(x).all() or np.max(np.abs(x)) >= 1 for x, _ in variants.values()):
        raise RuntimeError("存在非有限值或削波")
    audio.mkdir()
    for name, (signal, rate) in variants.items():
        write_wav_pcm16(audio / name, signal, rate)

    actual = {name: read_wav_pcm16(audio / name) for name in variants}
    levels = {}
    for name, (signal, rate) in actual.items():
        lufs, dbtp = f2.ebur128(audio / name)
        levels[name] = {"samples": len(signal), "sample_rate": rate, "lufs": lufs, "dbtp": dbtp}

    original_readback = actual["original.wav"][0]
    corrected_readback = actual["corrected.wav"][0]
    comparison = [pitch(signal, sr) for signal in (original_readback, corrected_readback)]
    local_core = tuple(t - PREVIEW[0] for t in SOURCE_CORE)
    first_pitch, second_pitch = (select(track, local_core) for track in comparison)
    measured = float(np.nanmedian(second_pitch - first_pitch))
    first_voiced, second_voiced = (np.isfinite(track[1]) for track in comparison)
    overlap = float(np.count_nonzero(first_voiced & second_voiced) / np.count_nonzero(first_voiced | second_voiced))
    checks = {
        "input_hashes_unchanged": all(digest(path) == expected for path, expected in hashes.items()),
        "output_length_and_rate": all(
            y.shape == variants[name][0].shape and rate == variants[name][1]
            for name, (y, rate) in actual.items()
        ),
        "outside_splice_identical": bool(
            np.array_equal(original_readback[mask[clip] == 0], corrected_readback[mask[clip] == 0])
        ),
        "no_clipping_and_true_peak": all(
            np.max(np.abs(actual[name][0])) < 1 and values["dbtp"] <= -1
            for name, values in levels.items()
        ),
        "matched_loudness": abs(levels["original.wav"]["lufs"] - levels["corrected.wav"]["lufs"]) <= 0.5,
        "voiced_overlap_at_least_0_95": overlap >= 0.95,
        "measured_shift_within_5_cents": abs(measured - shift) < 5,
    }
    report = {
        "source": SEGMENT,
        "source_in_first_round_training": SEGMENT in f2.TRAIN_SEGS,
        "source_and_reference_sha256": {str(path): digest for path, digest in hashes.items()},
        "alignment": {
            "method": "Qwen3-ForcedAligner-0.6B + Jade's prior '当你' label + stable voiced windows",
            "lyrics_source": "https://www.youtube.com/watch?v=p13tH7Etlzc",
            "reference_search_window": [53, 70],
            "rejected_reference_window": [23, 39],
            "source_word_seconds": SOURCE_WORD,
            "reference_word_seconds": REFERENCE_WORD,
            "source_pitch_core_seconds": SOURCE_CORE,
            "reference_pitch_core_seconds": REFERENCE_CORE,
            "key_offset_cents": key,
            "anchors": anchor_rows,
        },
        "preview_seconds_from_source": PREVIEW,
        "optional_reference_preview_seconds": REFERENCE_PREVIEW,
        "edit_seconds_in_source": EDIT,
        "passes": 1,
        "target_is_candidate_pending_listening": True,
        "direct_correction_cents": shift,
        "raw_model_correction_cents": round(model_shift, 3),
        "model_features": dict(zip(f2.FEATURES, features, strict=True)),
        "measured_correction_cents": round(measured, 3),
        "voiced_overlap": overlap,
        "levels": levels,
        "checks": checks,
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
