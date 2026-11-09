"""Train an experimental reference-pitch regressor from explicitly confirmed WAV ranges."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = Path(__file__).with_name("active_pitch_model.json")
SCHEMA = "f2-confirmed-frames-v1"
FEATURES = ("abs_center_delta", "oriented_local_delta", "shape_retained")
MAX_WORD_SHIFT = 300
MIN_CORRECTION = 35

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def save_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_clip(rec, verified=None):
    """Read exactly the human-confirmed frames and reject inconsistent source metadata."""
    path = Path(rec["path"])
    expected = rec["sha256"]
    if verified is None or verified.get(path) != expected:
        if sha(path) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        if verified is not None:
            verified[path] = expected
    start, end = rec["seconds"]
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        first, last = (round(t * rate) for t in (start, end))
        if [first, last] != rec["frames"] or not 0 <= first < last <= audio.getnframes():
            raise ValueError(f"Invalid approved bounds: {path} {rec['seconds']}")
        if rate != rec["rate"] or (audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (1, 2, "NONE"):
            raise ValueError(f"Unexpected WAV format: {path}")
        audio.setpos(first)
        data = np.frombuffer(audio.readframes(last - first), dtype="<i2")
    if len(data) != last - first:
        raise ValueError(f"Truncated audio: {path}")
    return data.astype(np.float64) / 32768.0, rate

def validate_split(pairs):
    groups = defaultdict(set)
    for row in pairs:
        groups[row["song"]].add(row["split"])
    if any(not value <= {"train", "holdout"} or len(value) != 1 for value in groups.values()):
        raise ValueError("Each song must belong to exactly one valid split.")

def word_bounds(first, second, interval, *, first_word=False, last=False):
    a, b = (float(first[k]) for k in ("start", "end"))
    c, d = (float(second[k]) for k in ("start", "end"))
    if first_word:
        a = c = interval[0]
    if last:
        b = d = interval[1]
    if not np.isfinite([a, b, c, d]).all() or b <= a or d <= c or max(abs(a-c), abs(b-d)) > .20:
        return None
    lo, hi = max(a, c, interval[0]), min(b, d, interval[1])
    return (lo, hi) if hi - lo >= .035 else None

def frame_rows(pair, word, times, cents, center, target, retain=1.0):
    lower, upper = pair["self"]["seconds"]
    if len(times) != len(cents) or not np.isfinite(cents).all() or np.any((times < lower) | (times >= upper)):
        raise ValueError(f"Frame outside confirmed source interval: {pair['id']}")
    delta = float(target - center)
    manual = retain != 1.0
    return [
        {"id": pair["id"], "song": pair["song"], "split": pair["split"], "word": word,
         "t": round(float(t), 6), "center_delta": delta, "local_delta": float(c - center),
         "shape_retained": retain,
         "target_shift": float(delta + (retain - 1) * (c - center)) if manual or abs(delta) >= MIN_CORRECTION else 0.0}
        for t, c in zip(times, cents)
    ]

def select_frames(track, interval):
    times, cents = track
    start, end = interval
    pad = min(.025, .14 * (end - start))
    use = np.isfinite(cents) & (times >= start + pad) & (times < end - pad)
    return times[use], cents[use]

def plateau(track, interval):
    t, v = select_frames(track, interval)
    if len(v) < 4:
        return None
    region = np.abs(v[:, None] - v[None, :]) <= 55
    keep = region[np.argmax(region.sum(axis=1))]
    if keep.sum() < 4 or keep.mean() < .65:
        return None
    return t[keep], v[keep], float(np.median(v[keep]))

def contextual_key(pair, words, marks, track):
    """Estimate absolute register offset from stable same-word context outside repair masks."""
    group = []
    for row in words:
        if (row["song_group"], row["self_version"]) != (pair["take"], pair["version"]):
            continue
        if (row["self_char"] != row["reference_lyric_candidate"]
            or row["self_machine_status"] != "OK候选" or row["reference_machine_status"] != "OK候选"):
            continue
        if (Path(row["self_source"]).name != Path(pair["self"]["path"]).name
            or Path(row["reference_source"]).name != Path(pair["reference"]["path"]).name):
            continue
        source = [float(row[k]) for k in ("self_start_s", "self_end_s")]
        ref = [float(row[k]) for k in ("reference_start_s", "reference_end_s")]
        group.append((row, source, ref))
    # Reference repeat must be the one actually selected in the review.
    windows = defaultdict(list)
    for row, s, r in group:
        windows[row["reference_window_start_s"]].append((row, s, r))
    mid = np.mean(pair["reference"]["seconds"])
    choices = [rows for rows in windows.values()
               if min(x[2][0] for x in rows) - .5 <= mid <= max(x[2][1] for x in rows) + .5]
    if len(choices) != 1:
        return None, []
    own = track(pair["self"]["path"])
    ref = track(pair["reference"]["path"])
    observed = []
    for row, s, r in choices[0]:
        if any(max(s[0], lo) < min(s[1], hi) for lo, hi in marks):
            continue
        if min(s[1] - s[0], r[1] - r[0]) < .08:
            continue
        a, b = plateau(own, s), plateau(ref, r)
        if a is not None and b is not None:
            observed.append({"word": row["self_char"], "index": row["index"], "self": s, "reference": r,
                             "delta": float(a[2] - b[2])})
    diffs = np.array([x["delta"] for x in observed])
    if len(diffs) < 3:
        return None, observed
    clusters = [np.flatnonzero((diffs >= low) & (diffs <= low + 100)) for low in diffs]
    best = max(clusters, key=lambda ids: (len(ids), -float(np.std(diffs[ids]))))
    estimate = float(np.median(diffs[best]))
    if len(best) < 3 or len(best) <= len(diffs) / 2 or abs(estimate) > 1500:
        return None, observed
    for i, row in enumerate(observed):
        row["used"] = i in best
    return estimate, observed

def prepare(args):
    dest = args.out
    if dest.exists():
        raise FileExistsError(dest)
    extra_site = os.environ.get("F2_PRAAT_SITE")
    if extra_site:
        sys.path.append(extra_site)
    from poc_phase1 import read_wav_pcm16
    from poc_phase2 import pitch_series

    pairs = read_json(args.pairs)
    validate_split(pairs)
    ids = [row["id"] for row in pairs]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Approved pair IDs must be unique.")
    probes = read_json(args.probes)["pairs"]
    context = read_csv(args.context)
    prior = read_json(args.manual)
    manual = next(pair for pair in pairs if pair["phrase"] == "学着" and pair["version"] == "对照微微跑调")
    prior_inputs = {Path(name).name: value for name, value in prior["inputs"].items()}
    for side in ("self", "reference"):
        rec = manual[side]
        if prior_inputs.get(Path(rec["path"]).name) != rec["sha256"]:
            raise ValueError("The confirmed 学 inputs differ from the approved manual reference.")
    if prior["trained_f2_model_used"] or prior["word"] != "学":
        raise ValueError("Expected the prior manual 学 curve.")

    def measured(data, rate, offset=0):
        times, hz = pitch_series(data, rate)
        return times + offset, 1200 * np.log2(hz / 440.0)

    context_cache = {}
    def full_track(path):
        if path not in context_cache:
            context_cache[path] = measured(*read_wav_pcm16(Path(path)))
        return context_cache[path]

    source_marks = defaultdict(list)
    for pair in pairs:
        source_marks[pair["self"]["path"]].append(pair["self"]["seconds"])
    verified = {}
    all_frames, audits = [], []
    for pair in pairs:
        if pair["action"] != "fix":
            raise ValueError(f"Unsupported annotation action on pair {pair['id']}")
        traces = {}
        for side in ("self", "reference"):
            rec = pair[side]
            data, rate = read_clip(rec, verified)
            traces[side] = measured(data, rate, rec["frames"][0] / rate)
        out = []
        audit = {"id": pair["id"], "phrase": pair["phrase"], "split": pair["split"], "words": []}
        if pair["id"] == manual["id"]:
            lo, hi = prior["source_core_seconds"]
            rlo, rhi = prior["reference_core_seconds"]
            if not pair["self"]["seconds"][0] <= lo < hi <= pair["self"]["seconds"][1]:
                raise ValueError("Manual glide lies outside the confirmed take.")
            if not pair["reference"]["seconds"][0] <= rlo < rhi <= pair["reference"]["seconds"][1]:
                raise ValueError("Manual reference lies outside the confirmed reference.")
            times, f0 = traces["self"]
            keep = np.isfinite(f0) & (times >= lo) & (times < hi)
            if keep.sum() < 5:
                raise ValueError("Insufficient tracked frames for the approved 学 glide.")
            out.extend(frame_rows(pair, "学", times[keep], f0[keep], prior["source_median_cents"],
                                  prior["target_cents"], prior["retained_fraction_of_glide"]))
            audit["words"].append({"word": "学", "status": "manual_curve", "frame_count": len(out)})
        else:
            key, evidence = contextual_key(pair, context, source_marks[pair["self"]["path"]], full_track)
            audit["key_cents"], audit["context_evidence"] = key, evidence
            for i, (own_word, ref_word) in enumerate(zip(pair["self_text"], pair["reference_text"], strict=True)):
                item = {"word": own_word, "reference_word": ref_word}
                audit["words"].append(item)
                if own_word != ref_word:
                    item["status"] = "different_syllable"
                    continue
                bounds = {}
                for side, expected in (("self", own_word), ("reference", ref_word)):
                    a, b = (probe[i] for probe in probes[str(pair["id"])][side])
                    if a["word"] != expected or b["word"] != expected:
                        raise ValueError(f"Timing probe token mismatch in pair {pair['id']}")
                    interval = pair[side]["seconds"]
                    bounds[side] = word_bounds(a, b, interval) or word_bounds(
                        a, b, interval, first_word=i == 0, last=i == len(pair[side + "_text"]) - 1)
                if None in bounds.values():
                    item["status"] = "timing_uncertain"
                    continue
                if key is None:
                    item["status"] = "key_uncertain"
                    continue
                own = plateau(traces["self"], bounds["self"])
                ref = plateau(traces["reference"], bounds["reference"])
                if own is None or ref is None:
                    item["status"] = "no_stable_voiced_core"
                    continue
                goal = float(ref[2] + key)
                if abs(goal - own[2]) > MAX_WORD_SHIFT:
                    item["status"] = "offset_exceeds_bound"
                    continue
                frames = frame_rows(pair, own_word, own[0], own[1], own[2], goal)
                item.update(status="correction" if frames[0]["target_shift"] else "near_target_control",
                            self_core=bounds["self"], reference_core=bounds["reference"],
                            frame_count=len(frames), shift=frames[0]["target_shift"])
                out.extend(frames)
        audit["frame_count"] = len(out)
        audit["nonzero_frames"] = sum(abs(x["target_shift"]) > 0 for x in out)
        audits.append(audit)
        all_frames.extend(out)
        print(f"#{pair['id']:02d} {pair['phrase']} {pair['split']} "
              f"{audit['frame_count']} frames, {audit['nonzero_frames']} corrections", flush=True)
    if not all_frames or any(sha(path) != before for path, before in verified.items()):
        raise ValueError("No usable frames or input files changed during preparation.")
    dest.mkdir()
    save_csv(dest / "samples.csv", all_frames)
    save_json(dest / "audit.json", {"schema": SCHEMA, "inputs": {str(p): sha(p) for p in
              (args.pairs, args.probes, args.context, args.manual)}, "source_hashes_verified": True,
              "samples_sha256": sha(dest / "samples.csv"), "manual_pair_id": manual["id"], "pairs": audits})
    print(f"Prepared {len(all_frames)} frames from {sum(x['frame_count'] > 0 for x in audits)}/{len(audits)} pairs.", flush=True)

def data_from(prepared):
    audit = read_json(prepared / "audit.json")
    if audit["schema"] != SCHEMA or sha(prepared / "samples.csv") != audit["samples_sha256"]:
        raise ValueError("Prepared samples failed integrity verification.")
    rows = read_csv(prepared / "samples.csv")
    validate_split(rows)
    return audit, rows

def relative_pitch(rows):
    return np.array([[float(row[k]) for k in ("center_delta", "local_delta", "shape_retained")]
                     for row in rows]).T

def features(delta, local, retain):
    return np.column_stack((np.abs(delta), local * np.sign(delta), retain))

def reference_targets(rows):
    return np.array([float(row["target_shift"]) for row in rows])

def estimate(model, rows):
    delta, local, retain = relative_pitch(rows)
    direction = np.sign(delta + (retain - 1) * local)
    direction[(retain == 1) & (np.abs(delta) < MIN_CORRECTION)] = 0
    return direction * np.maximum(0, model.predict(features(delta, local, retain)))

def metrics(rows, predictions, labels):
    by_pair = {}
    for number in dict.fromkeys(row["id"] for row in rows):
        use = np.array([row["id"] == number for row in rows])
        by_pair[number] = float(np.mean(np.abs(predictions[use] - labels[use])))
    nonzero = labels != 0
    return {
        "frames": len(rows), "pairs": len(by_pair), "mae_cents": float(np.mean(np.abs(predictions - labels))),
        "mean_pair_mae_cents": float(np.mean(list(by_pair.values()))), "pair_mae_cents": by_pair,
        "no_shift_mae_cents": float(np.mean(np.abs(labels))),
        "nonzero_frames": int(nonzero.sum()),
        "opposite_sign_frames": int(np.sum(nonzero & (predictions * labels < 0))),
    }

def train(args):
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn
    if args.out.exists():
        raise FileExistsError(args.out)
    audit, rows = data_from(args.prepared)
    train_rows = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    corrections = [row for row in train_rows if float(row["target_shift"])]
    if len({row["id"] for row in corrections}) < 3 or len({row["song"] for row in corrections}) < 2 or not holdout:
        raise ValueError("Insufficient distinct correction pairs and songs to fit and verify.")
    if not any(row["id"] == str(audit["manual_pair_id"]) and row["word"] == "学" for row in corrections):
        raise ValueError("The 学 glide is missing from the training set.")
    x = features(*relative_pitch(train_rows))
    y = reference_targets(train_rows)
    hy = reference_targets(holdout)
    counts = Counter(row["id"] for row in train_rows)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0)
    model.fit(x, np.abs(y), sample_weight=np.array([1 / counts[row["id"]] for row in train_rows]))
    p = estimate(model, holdout)
    if not np.isfinite(p).all():
        raise ValueError("Model returned nonfinite predictions.")
    scores = metrics(holdout, p, hy)
    args.out.mkdir()
    weight = args.out / "model.pkl"
    payload = {"schema": SCHEMA, "features": FEATURES, "model": model, "sklearn": sklearn.__version__,
               "samples_sha256": audit["samples_sha256"]}
    with weight.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    save_csv(args.out / "holdout_predictions.csv", [{**row, "predicted_shift": float(pred)}
                                                     for row, pred in zip(holdout, p, strict=True)])
    save_json(args.out / "train_report.json", {"schema": SCHEMA, "features": FEATURES, "sklearn": sklearn.__version__,
              "samples_sha256": audit["samples_sha256"], "weight_sha256": sha(weight),
              "training_songs": sorted({row["song"] for row in train_rows}),
              "holdout_songs": sorted({row["song"] for row in holdout}),
              "training": metrics(train_rows, estimate(model, train_rows), y),
              "holdout": scores, "direct_reference_target_mae_cents": 0.0,
              "label_source": "confirmed pairs plus checked reference pitch; 学 uses the manual curve",
              "direction_policy": "Reference sets direction and deadband; model learns magnitude.",
              "listening": "pending_jade"})
    print(json.dumps(scores, ensure_ascii=False, indent=2))

def load_model(weight, expected_sha):
    if sha(weight) != expected_sha:
        raise ValueError("Weight checksum mismatch.")
    with weight.open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle["schema"] != SCHEMA or tuple(bundle["features"]) != FEATURES:
        raise ValueError("Incompatible weight schema.")
    return bundle

def verify(args):
    audit, rows = data_from(args.prepared)
    report = read_json(args.results / "train_report.json")
    bundle = load_model(args.results / "model.pkl", report["weight_sha256"])
    if bundle["samples_sha256"] != audit["samples_sha256"]:
        raise ValueError("Weights and prepared samples differ.")
    holdout = [row for row in rows if row["split"] == "holdout"]
    old = read_csv(args.results / "holdout_predictions.csv")
    if len(old) != len(holdout) or any(any(r[k] != s[k] for k in s) for r, s in zip(old, holdout)):
        raise ValueError("Saved predictions refer to different samples.")
    hy = reference_targets(holdout)
    actual = estimate(bundle["model"], holdout)
    if not np.array_equal(actual, np.array([float(row["predicted_shift"]) for row in old])):
        raise ValueError("Reloaded weights failed exact prediction replay.")
    scores = metrics(holdout, actual, hy)
    if scores != report["holdout"]:
        raise ValueError("Holdout report did not reproduce.")
    print(f"Verified {len(rows)} exact prepared frames and {len(holdout)} held-out predictions after reload.")
    if args.control_report:
        controls = audio_control(bundle["model"], args.control_report)
        print(json.dumps(controls, ensure_ascii=False, indent=2))
        if args.audit_out:
            save_json(args.audit_out, {"weight_sha256": report["weight_sha256"],
                                      "heldout_prediction_replay": True, "control_audio": controls,
                                      "listening": "pending_jade"})

def audio_control(model, report_path):
    """Check both signs with the archived, measured half-step WAV pair."""
    extra_site = os.environ.get("F2_PRAAT_SITE")
    if extra_site:
        sys.path.append(extra_site)
    from poc_phase2 import pitch_series

    report = read_json(report_path)
    interval = report["pitch"]["core_seconds"]
    expected = float(report["injection"]["shift_cents"])
    f0 = {}
    for name in ("original.wav", "raised.wav"):
        path = report_path.parent / "audio" / name
        recorded = report["file_levels"][name]
        rate = recorded["sample_rate"]
        rec = {"path": str(path), "sha256": recorded["sha256"], "seconds": interval,
               "rate": rate, "frames": [round(t * rate) for t in interval]}
        data, sr = read_clip(rec)
        _, hz = pitch_series(data, sr)
        f0[name] = 1200 * np.log2(hz[np.isfinite(hz) & (hz > 0)] / 440)
        if len(f0[name]) < 8:
            raise ValueError("Control audio has insufficient voiced frames.")
    center = {name: float(np.median(values)) for name, values in f0.items()}
    measured = center["raised.wav"] - center["original.wav"]
    if abs(measured - expected) > 5:
        raise ValueError("Archived control no longer measures the known offset.")
    outcomes = []
    for own, ref, intended in (("original.wav", "raised.wav", expected),
                                ("raised.wav", "original.wav", -expected),
                                ("original.wav", "original.wav", 0)):
        rows = [{"center_delta": center[ref] - center[own], "local_delta": float(v - center[own]),
                 "shape_retained": 1.0} for v in f0[own]]
        predictions = estimate(model, rows)
        error = float(np.max(np.abs(predictions - intended)))
        if error > 50 or (intended and np.any(predictions * intended <= 0)) or (not intended and np.any(predictions)):
            raise ValueError(f"Known-shift control failed: target={intended} worst_error={error}")
        outcomes.append({"take": own, "reference": ref, "expected_cents": intended,
                         "median_prediction_cents": float(np.median(predictions)),
                         "max_absolute_error_cents": error, "frames": len(rows)})
    return {"source_report": str(report_path), "measured_raise_cents": measured, "cases": outcomes}

def predict(args):
    pointer = read_json(args.active)
    prepared = ROOT / pointer["prepared"]
    audit, rows = data_from(prepared)
    bundle = load_model(ROOT / pointer["weights"], pointer["sha256"])
    if audit["samples_sha256"] != bundle["samples_sha256"]:
        raise ValueError("The active model points to a different prepared set.")
    selected = [row for row in rows if row["id"] == str(args.id)]
    if not selected:
        raise ValueError(f"No verified target frames for approved pair {args.id}.")
    predicted = estimate(bundle["model"], selected)
    output = []
    for row, value in zip(selected, predicted, strict=True):
        direct = float(row["target_shift"])
        safe = value * direct > 0 and abs(value - direct) <= 60
        output.append({"id": args.id, "t": float(row["t"]), "word": row["word"],
                       "model_shift": float(value), "reference_shift": direct,
                       "proposed_shift": float(value) if safe and direct else 0.0,
                       "status": "model" if safe and direct else "keep" if direct == 0 else "review"})
    print(json.dumps(output, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    for name in ("pairs", "probes", "context", "manual", "out"):
        build.add_argument("--" + name, type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--prepared", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--prepared", type=Path, required=True)
    check.add_argument("--results", type=Path, required=True)
    check.add_argument("--control-report", type=Path)
    check.add_argument("--audit-out", type=Path)
    use = sub.add_parser("predict")
    use.add_argument("--id", type=int, required=True)
    use.add_argument("--active", type=Path, default=ACTIVE)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "verify": verify, "predict": predict}[args.command](args)

if __name__ == "__main__":
    main()

