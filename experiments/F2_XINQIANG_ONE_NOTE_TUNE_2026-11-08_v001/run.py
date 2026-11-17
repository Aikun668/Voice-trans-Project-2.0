"""Tune one sustained vowel in the confirmed take, then render an exact A/B."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEST = Path(__file__).resolve().parent
APPROVED = DEST.parent / "F2_XINQIANG_WORD_CUT_2026-11-08_v001"
sys.path.insert(0, str(DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase2 import pitch_series
from poc_phase4b import FIX_THRESHOLD, MAX_SHIFT, smooth_shift, splice
from self_ref_align import note_name


SEGMENT = "心墙01-对照微微跑调"
SOURCE_CORE = (1.205, 1.315)
REFERENCE_CORE = (114.278, 114.390)
PITCH_CURVE = (1.100, 1.443)
SPLICE_CORE = (1.160, 1.350)
AUTHORIZED_AUDIO = (1.100, 1.415)
ANCHORS = (
    ("不", (1.515, 1.595), (114.520, 114.610)),
    ("去", (1.745, 1.825), (114.770, 114.820)),
    ("心", (2.235, 2.365), (115.540, 115.620)),
    ("得", (2.435, 2.545), (115.790, 115.945)),
    ("太", (2.685, 2.875), (116.345, 116.445)),
    ("远", (3.090, 3.600), (116.515, 116.730)),
)


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


def main():
    out = DEST / "audio"
    report_path = DEST / "report.json"
    if out.exists() or report_path.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")

    approved = json.loads((APPROVED / "report.json").read_text(encoding="utf-8"))
    source_path, ref_path = f2.seg_path(SEGMENT), f2.ref_path("心墙")
    hashes = {str(path): digest(path) for path in (source_path, ref_path)}
    if approved["inputs"] != hashes:
        raise RuntimeError("原始素材与已确认切口的记录不一致")
    source, sr = read_wav_pcm16(source_path)
    reference, ref_sr = read_wav_pcm16(ref_path)
    source_track = pitch(source, sr)
    ref_window = (113.55, 117.25)
    ref_track = pitch(reference[span_slice(ref_window, ref_sr)], ref_sr, ref_window[0])

    anchor_rows = []
    for word, own, ref in ANCHORS:
        own_cents, ref_cents = median(source_track, own), median(ref_track, ref)
        anchor_rows.append({
            "word": word, "source_seconds": own, "reference_seconds": ref,
            "source_cents": own_cents, "reference_cents": ref_cents,
            "note_offset_cents": int(f2.circ(grid(own_cents) - grid(ref_cents))),
        })
    offsets, counts = np.unique([row["note_offset_cents"] for row in anchor_rows], return_counts=True)
    best = int(np.argmax(counts))
    if counts[best] <= len(anchor_rows) / 2:
        raise RuntimeError("邻近音符无法形成一致的调差")
    key = int(offsets[best])
    tuning = float(np.median([
        row["source_cents"] - grid(row["source_cents"])
        for row in anchor_rows if row["note_offset_cents"] == key
    ]))
    ref_cents = median(ref_track, REFERENCE_CORE)
    ref_frames = values(ref_track, REFERENCE_CORE)
    if np.nanpercentile(ref_frames, 90) - np.nanpercentile(ref_frames, 10) > 50:
        raise RuntimeError("原唱“着”的稳定部分音高离散过大")
    before_cents = median(source_track, SOURCE_CORE)
    target = before_cents - f2.circ(before_cents - grid(ref_cents) - key - tuning)
    shift = round(target - before_cents, 1)
    if not FIX_THRESHOLD <= abs(shift) <= MAX_SHIFT:
        raise RuntimeError(f"“着”的修正量超出本轮范围：{shift}")

    smooth = smooth_shift(source, sr, [(*PITCH_CURVE, shift)])
    corrected, mask = splice(source, smooth, sr, [(*SPLICE_CORE, shift)])
    if corrected.shape != source.shape or not np.isfinite(corrected).all():
        raise RuntimeError("重合成改变了样本数或产生非有限值")
    allowed = span_slice(AUTHORIZED_AUDIO, sr)
    changed = np.flatnonzero(mask > 0)
    if not len(changed) or changed[0] < allowed.start or changed[-1] >= allowed.stop:
        raise RuntimeError("实际拼接越过“着”的范围")

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
    measured = float(np.nanmedian(after_core - before_core))
    readback_pitch = median(after, SOURCE_CORE)
    va, vb = [np.isfinite(track[1]) for track in (before, after)]
    overlap = float(np.count_nonzero(va & vb) / np.count_nonzero(va | vb))
    outside = np.pad(mask[cut] == 0, (lead, tail), constant_values=True)
    peak_before, peak_after = [peak_step(track, AUTHORIZED_AUDIO) for track in (before, after)]
    checks = {
        "inputs_unchanged": all(digest(Path(p)) == h for p, h in hashes.items()),
        "anchor_matches_approved_cut": levels["original.wav"]["sha256"] == approved["levels"]["take.wav"]["sha256"],
        "exact_original_then_corrected": bool(np.array_equal(ab, np.concatenate((a, b)))),
        "outside_splice_identical": bool(np.array_equal(a[outside], b[outside]) and np.array_equal(
            corrected[mask == 0], source[mask == 0]
        )),
        "matched_loudness": abs(levels["original.wav"]["lufs"] - levels["corrected.wav"]["lufs"]) <= 0.5,
        "true_peak_at_most_minus_1_dbtp": all(item["dbtp"] <= -1 for item in levels.values()),
        "pitch_target_within_5_cents": abs(readback_pitch - target) < 5 and abs(measured - shift) < 5,
        "voiced_overlap_at_least_0_95": overlap >= 0.95,
        "largest_word_pitch_step_not_increased": peak_after <= peak_before + 5,
    }
    report = {
        "source": SEGMENT, "word": "着", "approved_cut": str(APPROVED / "report.json"),
        "anchors": anchor_rows, "majority_semitone_offset_cents": key, "local_tuning_cents": tuning,
        "reference_core_seconds": REFERENCE_CORE, "reference_note": note_name(ref_cents),
        "source_core_seconds": SOURCE_CORE, "source_median_cents": before_cents,
        "target_note": note_name(target), "target_cents": target,
        "planned_shift_cents": shift, "measured_shift_cents": measured, "readback_core_cents": readback_pitch,
        "pitch_curve_seconds": PITCH_CURVE, "affected_audio_seconds": [changed[0] / sr, (changed[-1] + 1) / sr],
        "processing": "One pass with poc_phase4b.smooth_shift (Praat), with existing local crossfade via splice",
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
