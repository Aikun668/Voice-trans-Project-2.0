"""F1-EXP-013 / F1-DIAG-3: 响度-only 验证。

背景：EXP-009/012 合并后，薄音与正常版的结构性差异只剩"频谱幅度分布"（已证伪）
     和"整体响度"（4/4 一致，中位差 4dB）。本轮验证一个假设：

      假设：所谓"单薄"，很大成分可能只是"唱得轻/录得小"，而非可修的频谱缺陷。

方法：薄音版与正常版各自施加**纯常数增益**（无 EQ/谐波/压缩/饱和/限制器），
      统一到同一试听响度。若响度抹平后薄音版不再被听作"薄"，则假设成立。

只处理 A01 与 A02（同曲薄/正常对照）；Holdout 不动。
"""
import csv, hashlib, json, os, pathlib, platform, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import scipy

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats


def ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return
    links = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
        return
    raise SystemExit("ffmpeg.exe not found; add its directory to PATH.")


ensure_ffmpeg_on_path()
if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
    raise RuntimeError("ffmpeg not on PATH")

OUT = ROOT / "experiments/F1-EXP-013"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-013 outputs already exist")
for d in (OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

PAIR = [("thin", "A_self_01", "薄音版"), ("normal", "A_self_02", "正常版")]
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load(did):
    it = items[did]
    wav = DATA / it["wav"]["path"]
    raw = DATA / it["raw"]["path"]
    if sha(raw) != it["raw"]["sha256"] or sha(wav) != it["wav"]["sha256"]:
        raise ValueError(f"input changed: {did}")
    x, sr = read_pcm16(wav)
    if len(x) != it["wav"]["frames"]:
        raise ValueError(f"frames mismatch: {did}")
    return x, sr


# ---------------- 读入与原始电平 ----------------
data = {}
for role, did, label in PAIR:
    x, sr = load(did)
    loud = measure_loudness(x, sr)
    st = sample_stats(x, sr)
    data[role] = {"id": did, "label": label, "x": x, "sr": sr, "loud": loud, "stats": st}

raw_level_diff = data["normal"]["loud"]["integrated_lufs"] - data["thin"]["loud"]["integrated_lufs"]

# ---------------- 统一目标响度 ----------------
target_lufs = min(-18.0, *[d["loud"]["integrated_lufs"] - 3.2 - d["loud"]["true_peak_dbtp"]
                           for d in data.values()])

listen_root = OUT / "listening/audio"
records = []
for role, d in data.items():
    gain = target_lufs - d["loud"]["integrated_lufs"]
    dest = listen_root / f"{role}.wav"
    export_wav(d["x"], d["sr"], gain, dest)
    ye, esr = read_pcm16(dest)
    dm = measure_loudness(ye, esr)
    est = sample_stats(ye, esr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{role} loudness match failed: {dm['integrated_lufs']} vs {target_lufs}")
    if dm["true_peak_dbtp"] > -2.9:
        raise RuntimeError(f"{role} true peak too hot: {dm['true_peak_dbtp']}")
    records.append({
        "role": role, "dataset_id": d["id"], "label": d["label"],
        "src_lufs": d["loud"]["integrated_lufs"], "src_true_peak_dbtp": d["loud"]["true_peak_dbtp"],
        "src_peak_dbfs": round(d["stats"]["sample_peak_dbfs"], 2),
        "src_rms_dbfs": round(d["stats"]["rms_dbfs"], 2),
        "applied_gain_db": round(gain, 2),
        "out_lufs": dm["integrated_lufs"], "out_true_peak_dbtp": dm["true_peak_dbtp"],
        "out_peak_dbfs": round(est["sample_peak_dbfs"], 2),
        "out_rms_dbfs": round(est["rms_dbfs"], 2),
        "frames": len(ye), "clipped": int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)),
        "file": str(dest.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(dest),
    })

lufs_gap_after = abs(records[1]["out_lufs"] - records[0]["out_lufs"])

# ---------------- 试听页 ----------------
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-DIAG-3 响度对照（平凡之路）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:190px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}</style>
<h1>F1-DIAG-3 · 响度对照（平凡之路）</h1>
<div class="key"><b>本轮只做一件事：把两个文件的响度调成完全一样，不做任何音色处理。</b><br>
没有 EQ、没有谐波增强、没有压缩、没有饱和、没有限制器。两个文件都只是"整体音量乘了一个常数"。<br>
<b>要回答的问题：当音量完全一致时，你还觉得第一个"薄"吗？</b></div>
<div class="note"><b>客观信息：</b>原始录音中，薄音版比正常版低 {raw_level_diff:.2f} dB 响度；为公平对比，薄音版被提升了 {records[0]["applied_gain_db"]:+.2f} dB，正常版被调整了 {records[1]["applied_gain_db"]:+.2f} dB，现在两者响度差仅 {lufs_gap_after:.2f} LU。<br>
两者是你自己唱的两次，歌词段落可能不完全相同，这是原始录音本身的差别，不是处理造成的。</div>
<div class="card"><h3>试听（正常标注，未盲听）</h3>
<div class="row"><label>薄音版（已调至统一响度）</label><audio controls preload="metadata" src="audio/thin.wav"></audio></div>
<div class="row"><label>正常版（已调至统一响度）</label><audio controls preload="metadata" src="audio/normal.wav"></audio></div></div>
<p class="meta">统一响度 {target_lufs:.2f} LUFS；真峰值均 ≤ {max(r["out_true_peak_dbtp"] for r in records):.1f} dBTP；无削波；原素材未改动。</p>
<p class="meta"><b>请回答三件事：</b>①现在两者还有"厚/薄"的明显区别吗？②如果能听出，具体差在哪里（是"没劲"还是"音色不同"）？③第一个还是让你不太想听吗？</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- 记录 ----------------
qa = {
    "experiment": "F1-EXP-013", "phase": "F1-DIAG-3",
    "hypothesis": "薄音主要来自能量水平（唱得轻/录得小），而非可修的频谱缺陷",
    "design": "薄音版与正常版各自施加纯常数增益，统一到同一响度；无任何音色处理、无限制器",
    "pair": {"song": "平凡之路", "thin": "A_self_01", "normal": "A_self_02"},
    "target_lufs": target_lufs,
    "raw_loudness_gap_db": round(raw_level_diff, 2),
    "loudness_gap_after_match_lu": round(lufs_gap_after, 3),
    "records": records,
    "sealed_not_processed": SEALED,
    "interpretation_rule": {
        "若响度抹平后不再薄": "假设成立：薄=能量不足；方向应为录音电平/演唱音量，停止 DSP 修音色",
        "若仍能听出明显厚薄差异": "存在真实音色差异；剩余未试只剩整体频谱包络迁移（预期低）",
    },
    "caveat": "同曲两次录音，段落/元音不完全相同；单对样本不能外推；Agent 未听音",
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["role", "dataset_id", "label", "src_lufs", "src_tp", "src_peak_dbfs", "src_rms_dbfs",
                "applied_gain_db", "out_lufs", "out_tp", "out_peak_dbfs", "out_rms_dbfs", "clipped"])
    for r in records:
        w.writerow([r["role"], r["dataset_id"], r["label"], r["src_lufs"], r["src_true_peak_dbtp"],
                    r["src_peak_dbfs"], r["src_rms_dbfs"], r["applied_gain_db"], r["out_lufs"],
                    r["out_true_peak_dbtp"], r["out_peak_dbfs"], r["out_rms_dbfs"], r["clipped"]])

params = {
    "experiment": "F1-EXP-013", "phase": "F1-DIAG-3",
    "question": "响度完全一致时，薄音版是否仍被听作'薄'",
    "processing": "纯常数增益；无 EQ/谐波/压缩/饱和/限制器",
    "pair": "A_self_01 (薄音版) vs A_self_02 (正常版) — 平凡之路",
    "target_lufs": target_lufs, "loudness_tolerance_lu": 0.3,
    "sealed": SEALED,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-013", "phase": "F1-DIAG-3", "status": "pending_jade",
    "pair": "平凡之路 薄音版 vs 正常版",
    "raw_loudness_gap_db": round(raw_level_diff, 2),
    "gains_applied_db": [r["applied_gain_db"] for r in records],
    "target_lufs": target_lufs,
    "technical_pass": all(abs(r["out_lufs"] - target_lufs) <= 0.3 and r["clipped"] == 0 for r in records),
    "preview_page": "listening/index.html",
    "processing": "constant gain only; no timbre processing",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-013\nphase: F1-DIAG-3\n"
    f"loudness-only verification; constant gain only; no timbre processing\n"
    f"target_lufs: {target_lufs}\ntechnical_pass: {summary['technical_pass']}\n"
    f"sealed: {', '.join(SEALED)}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-013 结果（pending_jade）\n\n"
    "响度-only 验证：薄音版与正常版各自施加纯常数增益、统一到同一响度，不做任何音色处理。\n\n"
    f"- 原始响度差：薄音版比正常版低 {raw_level_diff:.2f} dB\n"
    f"- 施加增益：薄音版 {records[0]['applied_gain_db']:+.2f} dB，正常版 {records[1]['applied_gain_db']:+.2f} dB\n"
    f"- 匹配后响度差：{lufs_gap_after:.2f} LU（目标 {target_lufs:.2f} LUFS）\n"
    f"- 削波：0；真峰值 ≤ {max(r['out_true_peak_dbtp'] for r in records):.1f} dBTP；原素材未改动\n\n"
    "## 判定规则\n\n"
    "- 若响度抹平后不再觉得薄 → 假设成立：薄=能量不足，方向转为录音电平/演唱音量\n"
    "- 若仍能听出明显厚薄差异 → 存在真实音色差异，剩余只剩整体频谱包络迁移\n\n"
    "听感结论待 Jade 试听后填写。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "raw_loudness_gap_db": round(raw_level_diff, 2),
    "loudness_gap_after_lu": round(lufs_gap_after, 3),
    "records": [{k: r[k] for k in ("role", "label", "src_lufs", "applied_gain_db",
                                    "out_lufs", "out_true_peak_dbtp", "clipped")} for r in records],
}, ensure_ascii=False, indent=1))
