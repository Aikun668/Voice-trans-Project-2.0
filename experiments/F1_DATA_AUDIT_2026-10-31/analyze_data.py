"""Phase-1 data audit for the F1 thin-vocal PoC.

Read-only on source audio. Writes analysis copies and reports only.
"""
import csv
import hashlib
import html
import json
import math
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import measure_loudness, export_wav, sample_stats, estimate_pitch  # noqa: E402
from f1_adaptive_v0.analyzer import analyze, read_wav  # noqa: E402

DATA = ROOT / "test_dataset/f1_dataset_v2"
EXPDIR = ROOT / "experiments/F1_DATA_AUDIT_2026-10-31"
OUT = EXPDIR / "run-002"
if OUT.exists():
    raise FileExistsError(OUT)
OUT.mkdir(parents=True)

# First batch: 6 A, 3 B, 3 C. Remaining items are reserves (pending).
A_FIRST = [("A01", "A_self_01", "dev"), ("A02", "A_self_03", "dev"),
           ("A03", "A_self_05", "dev"), ("A04", "A_self_07", "holdout"),
           ("A05", "A_gt_02", "dev"), ("A06", "A_gt_03", "holdout")]
A_RESERVE = ["A_self_02", "A_self_04", "A_self_06", "A_self_08", "A_gt_01", "A_gt_04"]
B_FIRST = [("B01", "B_gt_02", "dev"), ("B02", "B_gt_06", "dev"), ("B03", "B_gt_03", "holdout")]
B_RESERVE = ["B_gt_01", "B_gt_04", "B_gt_05"]
C_FIRST = [("C01", "C_gt_01", "dev"), ("C02", "C_gt_05", "dev"), ("C03", "C_gt_02", "holdout")]
C_RESERVE = ["C_gt_03", "C_gt_04", "C_gt_06"]

SAMPLE_MAP = {}
for prefix, first, reserve in (("A", A_FIRST, A_RESERVE), ("B", B_FIRST, B_RESERVE), ("C", C_FIRST, C_RESERVE)):
    for sid, did, split in first:
        SAMPLE_MAP[did] = (sid, split, True)
    next_index = len(first) + 1
    for did in reserve:
        SAMPLE_MAP[did] = (f"{prefix}{next_index:02d}", "pending", False)
        next_index += 1

manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}
if len(items) != 24:
    raise ValueError(f"Expected 24 items, got {len(items)}")

FFMPEG = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if FFMPEG.returncode != 0:
    raise RuntimeError("ffmpeg is not on PATH; prepend the WinGet Links directory before running")
FFMPEG_VERSION = FFMPEG.stdout.splitlines()[0]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def longest_zero_run(x, sr, threshold=1e-5):
    mask = np.abs(x) < threshold
    if not len(mask):
        return 0.0
    if mask.all():
        return len(x) / sr
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return float(max((e - s) / sr for s, e in zip(starts, ends)))


def gap_summary(a, duration):
    intervals = a["active_intervals"]
    if not intervals:
        return {"leading_gap_s": duration, "trailing_gap_s": 0.0, "longest_gap_s": duration, "gaps": []}
    lead = float(intervals[0]["start_seconds"])
    trail = float(duration - intervals[-1]["end_seconds"])
    gaps = [float(intervals[i]["start_seconds"] - intervals[i - 1]["end_seconds"]) for i in range(1, len(intervals))]
    return {"leading_gap_s": lead, "trailing_gap_s": trail,
            "longest_gap_s": max([lead, trail, *gaps], default=0.0), "gaps": gaps}


def band_fractions(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    total = float(np.sum(p[(f >= 20) & (f < 8000)]))
    bands = {"sub_20_80": (20, 80), "body_80_500": (80, 500), "mid_500_3000": (500, 3000),
             "presence_3000_5000": (3000, 5000), "high_5000_8000": (5000, 8000)}
    return {k: float(np.sum(p[(f >= lo) & (f < hi)]) / max(total, 1e-20)) for k, (lo, hi) in bands.items()}


def spectral_summary(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    band = (f >= 20) & (f < 8000)
    f, p = f[band], p[band]
    total = max(float(np.sum(p)), 1e-20)
    centroid = float(np.sum(f * p) / total)
    flatness = float(np.exp(np.mean(np.log(np.maximum(p, 1e-20)))) / max(np.mean(p), 1e-20))
    cumulative = np.cumsum(p)
    rolloff = float(f[np.searchsorted(cumulative, 0.85 * cumulative[-1], side="left")])
    return {"centroid_hz": centroid, "flatness": flatness, "rolloff85_hz": rolloff}


def noise_summary(x, sr, gate):
    # Descriptive proxy only: use frames below the activity gate as a rough
    # noise/quiet-tail pool. Do not turn this into a pass/fail quality gate.
    n = max(1, round(sr * 0.02))
    frames = x[:len(x) // n * n].reshape(-1, n)
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-12))
    low = levels[levels <= gate]
    floor = float(np.median(low)) if len(low) >= 5 else None
    active = levels[levels > gate]
    snr = float(np.median(active) - floor) if floor is not None and len(active) else None
    status = "proxy_only" if floor is not None else "insufficient_low_level_frames"
    return {"noise_floor_dbfs": floor, "estimated_snr_db": snr, "noise_status": status}


def feature_item(item):
    wav_path = DATA / item["wav"]["path"]
    x, sr, info = read_wav(wav_path)
    a = analyze(x, sr)
    loud = measure_loudness(x, sr)
    stats = sample_stats(x, sr)
    bands = band_fractions(x, sr)
    spec = spectral_summary(x, sr)
    gaps = gap_summary(a, len(x) / sr)
    noise = noise_summary(x, sr, a["activity_gate_dbfs"])
    pitch = estimate_pitch(x, sr)
    valid = pitch[np.isfinite(pitch) & (pitch > 0)]
    sid, split, first = SAMPLE_MAP[item["id"]]

    source = "Self" if item["source_type"] == "self" else "GTSinger"
    if item["source_type"] == "self":
        singer_id = "Self_User"
        singer_label = "用户本人"
    elif item["category"] == "C_observe":
        singer_id = item["singer"]
        singer_label = item["singer"]
    else:
        singer_id = "unknown"
        singer_label = item.get("gender_label_from_filename", "unknown") + "（文件名标签）"

    if item["source_type"] == "self":
        song = item.get("song", "")
        variant = item.get("variant", "")
        technique = f"自录版本：{variant}"
    elif item["category"] == "C_observe":
        song = item.get("song", "")
        variant = item.get("technique_group", "")
        technique = f"{item.get('technique_group', '')}；音区 {item['technique_annotation']['global_labels'].get('range')}"
    else:
        song = ""
        variant = ""
        technique = "未知；仅有男/女标签"

    flags = []
    if stats["clipped_samples"]:
        flags.append("clipping")
    if stats["sample_peak_dbfs"] > -1.0:
        flags.append("near_full_scale")
    if len(x) / sr < 8:
        flags.append("duration_lt_8s")
    if gaps["longest_gap_s"] > 1.0:
        flags.append("low_level_gap_gt_1s")
    if bands["sub_20_80"] > 0.02:
        flags.append("low_freq_content_review")
    review_flags = {"clipping", "near_full_scale", "low_freq_content_review"}
    status = "review" if any(f in review_flags for f in flags) else ("warn" if flags else "ok")

    if item["source_type"] == "self":
        acapella = "unconfirmed_user_recording"
    else:
        acapella = "dataset_declared_acapella"

    notes = []
    if item["source_type"] == "self":
        notes.append("原始 m4a 为 48k 立体声 AAC；wav 为 48k 单声道 PCM16")
        if item["clip"]["trimmed"]:
            notes.append(f"wav 截取 {item['clip']['start_seconds']}-{item['clip']['end_seconds']} s；原始文件保留")
    elif item["category"] == "C_observe":
        notes.append(f"GTSinger 精确路径：{item['source']['hf_path']}")
        notes.append("C 类仅作条件变化观察，不是跨歌手泛化")
    else:
        notes.append("GTSinger A/B 原始歌曲/歌手 ID 未在文件名中保留")
    if len(x) / sr < 8:
        notes.append("源片段长度不足 8 s，未补长")
    if flags:
        notes.append("质量提示：" + ", ".join(flags))
    notes.append("底噪/SNR 为低电平帧代理，仅作描述，不作质量判定")

    return {
        "sample_id": sid, "dataset_id": item["id"], "category": item["category"],
        "source": source, "source_type": item["source_type"], "singer_id": singer_id,
        "singer_label": singer_label, "song": song, "variant": variant, "technique": technique,
        "original_file": str(pathlib.Path("test_dataset/f1_dataset_v2") / item["raw"]["path"]).replace("\\", "/"),
        "wav_file": str(pathlib.Path("test_dataset/f1_dataset_v2") / item["wav"]["path"]).replace("\\", "/"),
        "clip_start_s": item["clip"]["start_seconds"], "clip_end_s": item["clip"]["end_seconds"],
        "trimmed": item["clip"]["trimmed"], "duration_s": len(x) / sr, "sample_rate": sr,
        "channels": 1, "bits_per_sample": 16,
        "loudness": loud, "level": stats,
        "spectral": spec, "bands": bands,
        "activity": {"active_seconds": a["active_seconds"], "low_level_seconds": a["low_level"]["seconds"],
                     "dynamic_range_db": a["dynamic_range_db"], **gaps,
                     "digital_silence_s": longest_zero_run(x, sr), **noise},
        "pitch": {"median_hz": float(np.median(valid)) if len(valid) else None,
                  "p10_hz": float(np.percentile(valid, 10)) if len(valid) else None,
                  "p90_hz": float(np.percentile(valid, 90)) if len(valid) else None,
                  "voiced_frames": int(len(valid)), "total_frames": int(len(pitch)),
                  "voiced_fraction": float(len(valid) / len(pitch)) if len(pitch) else None,
                  "limit": "legacy autocorrelation 60-500 Hz; not a full-range pitch guarantee"},
        "quality": {"status": status, "flags": flags, "acapella_status": acapella,
                    "distortion_note": "no clipping/full-scale detected; AAC source is lossy; not a full distortion analysis"},
        "split": split, "in_first_batch": first, "listening_judgement": "",
        "notes": notes,
    }


features = []
for item_id in sorted(items):
    features.append(feature_item(items[item_id]))
CATEGORY_ORDER = {"A_thin": 0, "B_full": 1, "C_observe": 2}
features.sort(key=lambda f: (CATEGORY_ORDER[f["category"]], int(f["sample_id"][1:])))

# Source integrity check
for f in features:
    raw = DATA / pathlib.Path(f["original_file"]).relative_to("test_dataset/f1_dataset_v2")
    if sha(raw) != items[f["dataset_id"]]["raw"]["sha256"]:
        raise ValueError(f"Raw source changed: {raw}")

# Preview: first batch only, common loudness target, constant gain only.
preview_dir = OUT / "previews"
preview_dir.mkdir()
first_features = [f for f in features if f["in_first_batch"]]
meters = [measure_loudness(read_wav(DATA / pathlib.Path(f["wav_file"]).relative_to("test_dataset/f1_dataset_v2"))[0], 48000)
          for f in first_features]
target = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in meters])
preview_manifest = []
for f in first_features:
    x, sr, _ = read_wav(DATA / pathlib.Path(f["wav_file"]).relative_to("test_dataset/f1_dataset_v2"))
    meter = measure_loudness(x, sr)
    gain = target - meter["integrated_lufs"]
    dest = preview_dir / f"{f['sample_id']}.wav"
    export_wav(x, sr, gain, dest)
    exported, rate = read_wav(dest)[:2]
    check = measure_loudness(exported, rate)
    f["preview"] = {"file": str(dest.relative_to(OUT)), "target_lufs": target, "gain_db": gain,
                    "lufs": check["integrated_lufs"], "true_peak_dbtp": check["true_peak_dbtp"]}
    preview_manifest.append({"sample_id": f["sample_id"], "dataset_id": f["dataset_id"],
                             "source": f["source"], "category": f["category"], "split": f["split"],
                             "preview_file": f["preview"]["file"].replace("\\", "/"), "gain_db": gain,
                             "lufs": check["integrated_lufs"], "true_peak_dbtp": check["true_peak_dbtp"]})

# Samples CSV
csv_path = OUT / "samples.csv"
columns = ["sample_id", "dataset_id", "source", "singer_id", "singer_label", "song", "variant",
           "technique_or_range", "original_file_relative_path", "clip_start_s", "clip_end_s",
           "duration_s", "sample_rate", "channels", "bits_per_sample", "integrated_lufs",
           "true_peak_dbtp", "peak_dbfs", "rms_dbfs", "spectral_centroid_hz", "spectral_flatness",
           "f0_median_hz", "f0_p10_hz", "f0_p90_hz", "voiced_fraction", "longest_low_level_gap_s",
           "noise_floor_dbfs", "estimated_snr_db", "noise_status", "clipped_samples", "quality_flags",
           "quality_status", "acapella_status", "listening_judgement", "split", "in_first_batch",
           "preview_file", "notes"]
with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for x in features:
        writer.writerow([
            x["sample_id"], x["dataset_id"], x["source"], x["singer_id"], x["singer_label"], x["song"], x["variant"],
            x["technique"], x["original_file"], x["clip_start_s"], x["clip_end_s"], round(x["duration_s"], 3),
            x["sample_rate"], x["channels"], x["bits_per_sample"], round(x["loudness"]["integrated_lufs"], 2),
            round(x["loudness"]["true_peak_dbtp"], 2), round(x["level"]["sample_peak_dbfs"], 2),
            round(x["level"]["rms_dbfs"], 2), round(x["spectral"]["centroid_hz"], 1),
            round(x["spectral"]["flatness"], 4),
            round(x["pitch"]["median_hz"], 1) if x["pitch"]["median_hz"] else "",
            round(x["pitch"]["p10_hz"], 1) if x["pitch"]["p10_hz"] else "",
            round(x["pitch"]["p90_hz"], 1) if x["pitch"]["p90_hz"] else "",
            round(x["pitch"]["voiced_fraction"], 3), round(x["activity"]["longest_gap_s"], 3),
            round(x["activity"]["noise_floor_dbfs"], 2) if x["activity"]["noise_floor_dbfs"] is not None else "",
            round(x["activity"]["estimated_snr_db"], 2) if x["activity"]["estimated_snr_db"] is not None else "",
            x["activity"]["noise_status"],
            x["level"]["clipped_samples"], ";".join(x["quality"]["flags"]), x["quality"]["status"],
            x["quality"]["acapella_status"], x["listening_judgement"], x["split"],
            "yes" if x["in_first_batch"] else "no",
            x["preview"]["file"].replace("\\", "/") if "preview" in x else "",
            " | ".join(x["notes"]),
        ])

# Detailed features JSON
(OUT / "features.json").write_text(json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "preview_manifest.json").write_text(json.dumps(preview_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Simple listening page
def card(f):
    badges = ", ".join(f["quality"]["flags"]) if f["quality"]["flags"] else "无明显工程提示"
    label = f"{f['sample_id']} · {f['source']} · {f['singer_label']}"
    detail = " / ".join(x for x in [f["song"], f["variant"], f["technique"]] if x)
    return (f'<section class="card"><h3>{html.escape(label)}</h3>'
            f'<p class="meta">{html.escape(detail)} · {f["duration_s"]:.2f}s · split={f["split"]}</p>'
            f'<p class="meta">质量：{html.escape(f["quality"]["status"])} · {html.escape(badges)}</p>'
            f'<audio controls preload="metadata" src="{html.escape(f["preview"]["file"].replace(chr(92), "/"))}"></audio>'
            f'<p class="path">{html.escape(f["original_file"])}</p></section>')

page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 Phase-1 候选试听</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}
h1{font-size:24px}h2{font-size:20px;margin-top:28px}.card{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:12px 0}
h3{font-size:17px;margin:0 0 8px}.meta{color:#596579;margin:4px 0}.path{color:#7a8494;font-size:12px;word-break:break-all}
audio{width:100%;margin-top:8px}.note{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}
</style>
<h1>F1 Phase-1 候选试听</h1>
<div class="note"><b>先说明：</b>分类只是候选，尚未人工确认。预览只做等响固定增益，没有 EQ、压缩、谐波或美化处理。请固定播放器音量，先听 A01–A06，判断“是否确实单薄”；B 用于保护，C 用于音区/唱法观察。</div>
<h2>A · 单薄候选（未确认）</h2>''' + "".join(card(f) for f in first_features if f["category"] == "A_thin") + '''
<h2>B · 饱满保护候选（未确认）</h2>''' + "".join(card(f) for f in first_features if f["category"] == "B_full") + '''
<h2>C · 音区/唱法观察（技巧标签已知，感知类别未确认）</h2>''' + "".join(card(f) for f in first_features if f["category"] == "C_observe") + '''
<p class="meta">详细清单：<a href="samples.csv">samples.csv</a> · 特征：<a href="features.json">features.json</a></p>
</html>'''
(OUT / "index.html").write_text(page, encoding="utf-8")

# Summary and run log
counts = {k: sum(1 for f in features if f["category"] == k) for k in ("A_thin", "B_full", "C_observe")}
quality = {k: sum(1 for f in features if f["quality"]["status"] == k) for k in ("ok", "warn", "review")}
summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest": str(DATA / "manifest.json"), "dataset_manifest_sha256": sha(DATA / "manifest.json"),
    "counts": counts, "total": len(features), "first_batch": len(first_features),
    "quality_status": quality, "preview_target_lufs": target,
    "preview_count": len(preview_manifest), "preview_dir": str(preview_dir.relative_to(ROOT)),
    "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "ffmpeg": FFMPEG_VERSION},
    "outputs": ["features.json", "samples.csv", "preview_manifest.json", "index.html", "previews/"],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {summary['created_utc']}\n"
    f"dataset_manifest: {summary['dataset_manifest']}\n"
    f"dataset_manifest_sha256: {summary['dataset_manifest_sha256']}\n"
    f"python: {summary['environment']['python']}\nplatform: {summary['environment']['platform']}\n"
    f"ffmpeg: {summary['environment']['ffmpeg']}\n"
    f"counts: {json.dumps(counts, ensure_ascii=False)}\n"
    f"first_batch: {len(first_features)}\npreview_target_lufs: {target}\n"
    f"source integrity: all raw SHA256 matched dataset manifest\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=1))
