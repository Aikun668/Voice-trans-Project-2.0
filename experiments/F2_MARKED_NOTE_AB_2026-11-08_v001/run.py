"""Render one previously marked deliberate pitch error and a continuous A/B WAV."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase2 import pitch_series


SONG = "莫问归期"
SEGMENT = SONG + "02-对照微微跑调"
MARKED = (0.0, 2.177)
EDIT = (1.94, 2.41)
PREVIEW = (0.50, 2.50)
SOURCE_CORE = (2.05, 2.31)
REFERENCE_CORE = (135.32, 135.49)
ANCHORS = (
    ((9.16, 9.40), (142.73, 142.98)),
    ((9.64, 9.90), (143.17, 143.40)),
    ((11.85, 12.03), (145.42, 145.54)),
    ((12.39, 12.66), (145.82, 146.00)),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def part(span, sr):
    return slice(round(span[0] * sr), round(span[1] * sr))


def pitch(signal, sr, offset=0.0):
    times, hz = pitch_series(signal, sr)
    return times + offset, 1200 * np.log2(hz / 440)


def values(track, span):
    times, cents = track
    return cents[(times >= span[0]) & (times < span[1])]


def median(track, span):
    frames = values(track, span)
    if np.count_nonzero(np.isfinite(frames)) < 8:
        raise RuntimeError(f"音高帧不足：{span}")
    return float(np.nanmedian(frames))


def main():
    audio = DEST / "audio"
    report = DEST / "report.json"
    if audio.exists() or report.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    source_path, reference_path = f2.seg_path(SEGMENT), f2.ref_path(SONG)
    hashes = {path: sha256(path) for path in (source_path, reference_path)}
    source, sr = read_wav_pcm16(source_path)
    reference, ref_sr = read_wav_pcm16(reference_path)
    own = pitch(source, sr)
    ref = pitch(reference[part((133, 147), ref_sr)], ref_sr, 133)
    offsets = [f2.circ(median(own, a) - median(ref, b)) for a, b in ANCHORS]
    key = float(np.median(offsets))
    shift = round(-f2.circ(median(own, SOURCE_CORE) - median(ref, REFERENCE_CORE) - key), 1)
    if not -f2.FIX_MAX_ABS <= shift <= -100:
        raise RuntimeError(f"目标音偏差超出诊断范围：{shift}")
    corrected, mask = f2.corrected_audio(source, sr, [(*EDIT, shift)])
    if source.shape != corrected.shape:
        raise RuntimeError("处理改变了源音频长度")

    clip = part(PREVIEW, sr)
    original_clip, corrected_clip = source[clip], corrected[clip]
    gain = f2.gain_match(original_clip, corrected_clip)
    # Match both clip edges so the A/B join itself does not add a click.
    edge = np.ones(len(original_clip))
    fade = round(0.008 * sr)
    edge[:fade] = np.linspace(0, 1, fade)
    edge[-fade:] = edge[:fade][::-1]
    original_clip = original_clip * gain * edge
    corrected_clip = corrected_clip * gain * edge
    variants = {
        "original.wav": original_clip,
        "corrected.wav": corrected_clip,
        "compare.wav": np.concatenate((original_clip, corrected_clip)),
    }
    if any(not np.isfinite(y).all() or np.max(np.abs(y)) >= 1 for y in variants.values()):
        raise RuntimeError("输出存在非有限值或削波")
    audio.mkdir()
    readback, levels = {}, {}
    for name, samples in variants.items():
        path = audio / name
        write_wav_pcm16(path, samples, sr)
        y, sample_rate = read_wav_pcm16(path)
        if len(y) != len(samples) or sample_rate != sr or np.max(np.abs(y)) >= 1:
            raise RuntimeError(f"WAV 回读失败：{name}")
        lufs, dbtp = f2.ebur128(path)
        readback[name] = y
        levels[name] = {"samples": len(y), "sample_rate": sr, "lufs": lufs, "dbtp": dbtp, "sha256": sha256(path)}

    a, b, ab = (readback[name] for name in variants)
    before, after = (pitch(y, sr, PREVIEW[0]) for y in (a, b))
    measured = float(np.nanmedian(values(after, SOURCE_CORE) - values(before, SOURCE_CORE)))
    va, vb = np.isfinite(before[1]), np.isfinite(after[1])
    voiced_overlap = float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb))
    outside = mask[clip] == 0
    checks = {
        "inputs_unchanged": all(sha256(path) == expected for path, expected in hashes.items()),
        "exact_original_then_corrected": bool(np.array_equal(ab, np.concatenate((a, b)))),
        "outside_splice_identical": bool(np.array_equal(a[outside], b[outside])),
        "loudness_difference_at_most_half_lu": abs(levels["original.wav"]["lufs"] - levels["corrected.wav"]["lufs"]) <= 0.5,
        "true_peak_at_most_minus_1_dbtp": all(info["dbtp"] <= -1 for info in levels.values()),
        "measured_shift_within_5_cents": abs(measured - shift) < 5,
        "voiced_overlap_at_least_0_95": voiced_overlap >= 0.95,
    }
    result = {
        "source": SEGMENT,
        "mark_source": "F2_R1_MARK_2026-11-07_v001/jade_fragments.csv, #17 in the original export",
        "marked_phrase": "沉默有时",
        "marked_seconds": MARKED,
        "edited_word": "时",
        "alignment": "Qwen3-ForcedAligner-0.6B + stable voiced windows; use complete sustained syllable",
        "aligned_word_seconds": {"source": [1.76, 2.40], "reference": [135.16, 135.56]},
        "edited_source_seconds": EDIT,
        "preview_source_seconds": PREVIEW,
        "reference_core_seconds": REFERENCE_CORE,
        "source_core_seconds": SOURCE_CORE,
        "key_anchor_windows": ANCHORS,
        "key_anchor_offsets_cents": offsets,
        "target_shift_cents": shift,
        "measured_shift_cents": round(measured, 3),
        "voiced_overlap": voiced_overlap,
        "compare_split_seconds": len(a) / sr,
        "preview_edge_fade_ms": 8,
        "levels": levels,
        "source_hashes": {str(path): expected for path, expected in hashes.items()},
        "checks": checks,
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
