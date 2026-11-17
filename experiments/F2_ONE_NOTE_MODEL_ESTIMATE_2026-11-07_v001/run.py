"""Probe the saved F2 model with a manually aligned note, without rendering audio."""

import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import sklearn

DEST = Path(__file__).resolve().parent
TRAIN = DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"
CONTROL = DEST.parent / "F2_ONE_NOTE_ROUND_TRIP_2026-11-07_v002"
sys.path.insert(0, str(TRAIN))
import r1_pipeline as f2


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = DEST / "report.json"
    if output.exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    control = json.loads((CONTROL / "report.json").read_text(encoding="utf-8"))
    audio = {name: CONTROL / "audio" / name for name in ("original.wav", "raised.wav")}
    audio_hashes = {name: sha256(path) for name, path in audio.items()}
    for name, digest in audio_hashes.items():
        if digest != control["file_levels"][name]["sha256"]:
            raise RuntimeError(f"试听素材与 v002 实验记录不一致: {name}")

    known_start, known_end = control["span_seconds"]
    core_start, core_end = control["pitch"]["core_seconds"]
    # Keep the normal vocal note features. The known interval selects the note,
    # and every frame is paired by its original timestamp instead of searching.
    candidates = [
        note for note in f2.notes_of(str(audio["raised.wav"]))
        if note["t0"] <= core_start and note["t1"] >= core_end
    ]
    if len(candidates) != 1:
        raise RuntimeError("未找到唯一覆盖目标长音中段的音符")
    note = candidates[0]
    reference_track = f2.cents_track(str(audio["original.wav"]))
    first = max(0, round(note["t0"] / f2.HOP))
    last = min(len(reference_track) - 1, round(note["t1"] / f2.HOP))
    reference_frames = reference_track[first:last + 1]
    voiced = reference_frames[np.isfinite(reference_frames)]
    if len(voiced) < 3 or len(voiced) / len(reference_frames) < f2.VOICED_MIN:
        raise RuntimeError("参考片段没有足够的有效音高帧")
    seed = np.median(voiced)
    folded = seed + (voiced - seed + 600.0) % 1200.0 - 600.0
    features = f2.feature_row({
        **note,
        "dur": round(note["t1"] - note["t0"], 3),
        "ref_c_global": float(np.median(folded)),  # Same take and key, offset is zero.
        "ref_spread": float(np.percentile(folded, 90) - np.percentile(folded, 10)),
        "ref_voiced": round(len(voiced) / len(reference_frames), 2),
    })

    weight = TRAIN / "results" / "model.pkl"
    weight_hash = sha256(weight)
    with weight.open("rb") as handle:
        bundle = pickle.load(handle)  # Existing local model created by this project.
    if bundle["features"] != f2.FEATURES:
        raise RuntimeError("模型的特征顺序与推理代码不一致")
    predicted = float(bundle["model"].predict([features])[0])
    if not np.isfinite(predicted):
        raise RuntimeError("模型预测不是有限数值")

    inputs = dict(zip(f2.FEATURES, features, strict=True))
    baseline = f2.circ(inputs["ref_c_global"] - inputs["sung_cents"])
    target = -control["injection"]["shift_cents"]
    if weight_hash != sha256(weight) or any(
        sha256(path) != audio_hashes[name] for name, path in audio.items()
    ):
        raise RuntimeError("实验期间输入文件发生变化")

    report = {
        "model": str(weight),
        "model_sha256": weight_hash,
        "sklearn_version": sklearn.__version__,
        "control": str(CONTROL / "report.json"),
        "audio_sha256": audio_hashes,
        "alignment": "same take, matched timestamps, zero key offset",
        "injected_span_seconds": [known_start, known_end],
        "detected_note_span_seconds": [round(note["t0"], 3), round(note["t1"], 3)],
        "reference_track_frames": [first, last],
        "model_features": {key: round(float(value), 6) for key, value in inputs.items()},
        "commanded_raise_cents": -target,
        "measured_raise_in_core_cents": control["pitch"]["core_raise_vs_original_cents"],
        "expected_correction_cents": target,
        "raw_model_correction_cents": round(predicted, 3),
        "model_error_signed_cents": round(predicted - target, 3),
        "aligned_pitch_difference_correction_cents": round(baseline, 3),
        "aligned_pitch_difference_error_signed_cents": round(baseline - target, 3),
        "raw_prediction": True,
        "guardrail_or_fallback_applied": False,
        "rendered_audio": False,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
