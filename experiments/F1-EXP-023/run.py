"""F1-EXP-023 / F1-P9: 选择性向上动态处理（弱段抬升）。

基线 = EXP-020 S45（X1+M2+F1 最大档，原样引用）。
方向与 EXP-008 的向下压缩相反：只抬"低于正常水平的弱段"，强段不动。

公式（dB 域，逐帧）：
  R      = 浊音帧能量 P75（该素材"正常水平"）
  D      = max(0, R - DZ - L)          # 弱出多少（DZ 死区 3dB）
  g_tgt  = min(GMAX, RATIO_UP * D)     # 只补一部分，保留相对差异
  非浊音帧 g_tgt = 0
  然后对 g 做 attack/release 平滑（上快下慢），避免突变与抽吸。

两档只改 GMAX：+4dB 与 +12dB，用于大跨度试听，之后再缩小。
"""
import csv, hashlib, json, math, os, pathlib, platform, random, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats, estimate_pitch


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

OUT = ROOT / "experiments/F1-EXP-023"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-023 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

TARGET_ID = "A_self_01"
BASE_PATH = ROOT / "experiments/F1-EXP-020/outputs/processed_raw/S45.wav"
BASE_QA = ROOT / "experiments/F1-EXP-020/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
STEPS = [("U1", 6.0, 0.35), ("U2", 20.0, 1.00)]   # (id, GMAX, RATIO_UP)

WIN_S, HOP_S = 0.025, 0.010
FMIN, FMAX = 90.0, 450.0
R_VOICED = 0.45
GATE_FLOOR_DB = 45.0
REF_PERCENTILE = 75
DEADZONE_DB = 3.0
RATIO_UP = 0.60
ATTACK_S = 0.020
RELEASE_S = 0.030
SEED = 20261111


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def frame_f0_ac(win, sr):
    w = np.hanning(len(win)) * (win - win.mean())
    if not np.any(w):
        return float("nan"), 0.0
    c = signal.correlate(w, w, method="fft")[len(w) - 1:]
    if c[0] <= 0:
        return float("nan"), 0.0
    c = c / c[0]
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    if hi >= len(c):
        return float("nan"), 0.0
    k = int(np.argmax(c[lo:hi + 1])) + lo
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        k_f, r = k + d, float(y1)
    else:
        k_f, r = float(k), float(c[k])
    f0 = sr / k_f
    if not (FMIN <= f0 <= FMAX):
        return float("nan"), r
    return float(f0), float(r)


def frame_analysis(x, sr):
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    starts = list(range(0, len(x) - win + 1, hop))
    rms, rv = [], []
    for s in starts:
        fr = x[s:s + win]
        rms.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        _, r = frame_f0_ac(fr, sr)
        rv.append(r)
    rms = np.array(rms)
    rv = np.array(rv)
    gate = rms.max() - GATE_FLOOR_DB
    voiced = (rv >= R_VOICED) & (rms >= gate)
    return rms, voiced, hop, win


def upward_compression(x, sr, gmax_db, ratio_up):
    rms, voiced, hop, win = frame_analysis(x, sr)
    n = len(rms)
    if voiced.sum() < 100:
        raise RuntimeError("浊音帧过少，无法可靠处理")
    ref = float(np.percentile(rms[voiced], REF_PERCENTILE))

    target = np.zeros(n)
    for i in range(n):
        if not voiced[i]:
            continue
        deficit = ref - DEADZONE_DB - rms[i]
        if deficit > 0:
            target[i] = min(gmax_db, ratio_up * deficit)

    # attack/release 平滑（dB 域）
    a_coef = 1.0 - math.exp(-HOP_S / ATTACK_S)
    r_coef = 1.0 - math.exp(-HOP_S / RELEASE_S)
    g = np.zeros(n)
    prev = 0.0
    for i in range(n):
        coef = a_coef if target[i] > prev else r_coef
        prev = prev + coef * (target[i] - prev)
        g[i] = prev

    # 逐样本展开 + 汉宁交叠平滑，避免帧边界台阶
    g_sample = np.repeat(g, hop) if len(g) * hop <= len(x) else np.zeros(len(x))
    if len(g_sample) < len(x):
        g_sample = np.pad(g_sample, (0, len(x) - len(g_sample)), mode="edge")
    W = win
    kernel = np.hanning(W)
    kernel = kernel / kernel.sum()
    g_smooth = np.convolve(g_sample, kernel, mode="same")

    y = x * (10.0 ** (g_smooth / 20.0))
    stats = {
        "ref_level_db": round(ref, 2),
        "frames": int(n),
        "voiced_frames": int(voiced.sum()),
        "frames_boosted_ge1db": int(np.count_nonzero(g_smooth >= 1.0)),
        "boosted_frame_fraction_ge1db": round(float(np.mean(g_smooth >= 1.0)), 3),
        "gain_median_when_active_db": round(float(np.median(g_smooth[g_smooth >= 1.0])), 2)
            if np.any(g_smooth >= 1.0) else 0.0,
        "gain_max_db": round(float(np.max(g_smooth)), 2),
        "gain_p90_db": round(float(np.percentile(g_smooth, 90)), 2),
    }
    return y, stats, g_smooth, rms, voiced, ref


def region_gain(x_old, x_new, sr, rms, voiced, hop, win):
    """按电平分层测实际增益：弱段 / 中段 / 强段。"""
    n = min(len(x_old), len(x_new))
    ref = float(np.percentile(rms[voiced], REF_PERCENTILE))
    weak, mid, strong = [], [], []
    w = win
    step = hop
    for i, s in enumerate(range(0, n - w + 1, step)):
        if i >= len(rms) or not voiced[i]:
            continue
        eo = float(np.sqrt(np.mean(x_old[s:s + w] ** 2)))
        en = float(np.sqrt(np.mean(x_new[s:s + w] ** 2)))
        if eo < 1e-8:
            continue
        d = 20.0 * np.log10(max(en, 1e-12) / max(eo, 1e-12))
        if rms[i] < ref - DEADZONE_DB:
            weak.append(d)
        elif rms[i] > ref + 1.0:
            strong.append(d)
        else:
            mid.append(d)
    def med(v):
        return round(float(np.median(v)), 2) if v else None
    def p90(v):
        return round(float(np.percentile(v, 90)), 2) if v else None
    return {
        "weak_frames": len(weak), "weak_median_db": med(weak), "weak_p90_db": p90(weak),
        "mid_frames": len(mid), "mid_median_db": med(mid),
        "strong_frames": len(strong), "strong_median_db": med(strong),
    }


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


x_orig, sr = load(TARGET_ID)
if not BASE_PATH.exists():
    raise FileNotFoundError("EXP-020 S45.wav missing")
exp020 = json.loads(BASE_QA.read_text(encoding="utf-8"))
base_rec = next(v for v in exp020["versions"] if v["id"] == "S45")
base_sha = sha(BASE_PATH)
x_base, sr_b = read_pcm16(BASE_PATH)
if sr_b != sr:
    raise RuntimeError("sample rate mismatch")
if len(x_base) != base_rec["frames"]:
    raise ValueError("baseline frame count drift")

raw_root = OUT / "outputs/processed_raw"
records = []
for vid, gmax, ratio in STEPS:
    y, pstat, g_smooth, rms, voiced, ref = upward_compression(x_base, sr, gmax, ratio)
    reg = region_gain(x_base, y, sr, rms, voiced, int(HOP_S * sr), int(WIN_S * sr))
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    pitch_base = float(np.nanmedian(estimate_pitch(x_base, sr)))
    hard, flags = [], []
    if reg["weak_median_db"] is None:
        hard.append("弱段增益无法测量")
    elif reg["weak_median_db"] < 0.5:
        hard.append(f"弱段几乎未被抬升 实测{reg['weak_median_db']:+.2f}dB")
    if reg["strong_median_db"] is None or abs(reg["strong_median_db"]) > 0.6:
        hard.append(f"强段被误改 {reg['strong_median_db']}")
    if abs(pitch - pitch_base) > 1.0:
        hard.append(f"F0 偏移 {pitch - pitch_base:+.2f}Hz")
    if len(y) != len(x_base):
        hard.append(f"样本数变化 {len(y)}")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768))
    if clipped > 0:
        hard.append(f"削波 {clipped}")
    if pstat["boosted_frame_fraction_ge1db"] < 0.02:
        flags.append("有效抬升覆盖过低")

    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, 0.0, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    records.append({
        "id": vid, "gmax_db": gmax, "ratio_up": ratio,
        "proc": pstat, "regions": reg,
        "pitch_hz": round(pitch, 2), "pitch_base_hz": round(pitch_base, 2),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2),
        "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "frames": len(y), "clipped": clipped,
        "raw_lufs": round(ye_loud["integrated_lufs"], 2),
        "raw_true_peak_dbtp": round(ye_loud["true_peak_dbtp"], 2),
        "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    })

loud_orig = measure_loudness(x_orig, sr)
loud_base = measure_loudness(x_base, sr)
loud_all = {"original": loud_orig, "baseline": loud_base,
            **{r["id"]: {"integrated_lufs": r["raw_lufs"], "true_peak_dbtp": r["raw_true_peak_dbtp"]}
               for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loud_all.values()])

listen_root = OUT / "listening/audio"
exports = []
for tag, data_, loud in (("original", x_orig, loud_orig), ("baseline", x_base, loud_base)):
    dest = listen_root / f"{tag}.wav"
    export_wav(data_, sr, target_lufs - loud["integrated_lufs"], dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{tag} loudness match failed")
    exports.append({"tag": tag, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                    "lufs": dm["integrated_lufs"], "true_peak_dbtp": dm["true_peak_dbtp"], "sha256": sha(dest)})

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-023")
    ym, ysr = read_pcm16(src)
    m = measure_loudness(ym, ysr)
    gain = target_lufs - m["integrated_lufs"]
    dest = listen_root / f"{r['id']}.wav"
    export_wav(ym, ysr, gain, dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{r['id']} loudness match failed")
    if dm["true_peak_dbtp"] > -2.9:
        raise RuntimeError(f"{r['id']} true peak too hot")
    r["matched"] = {"gain_db": round(gain, 2), "lufs": dm["integrated_lufs"],
                    "true_peak_dbtp": dm["true_peak_dbtp"], "sha256": sha(dest)}

rng = random.Random(SEED)
deliverables = [r["id"] for r in records if r["deliverable"]]
letters = ["A", "B", "C"][:len(deliverables)]
blind = {letter: vid for letter, vid in zip(letters, rng.sample(deliverables, len(deliverables)))}
for letter, vid in blind.items():
    shutil.copyfile(listen_root / f"{vid}.wav", listen_root / f"{letter}.wav")
    (listen_root / f"{vid}.wav").unlink()


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


rows = []
for letter in blind.keys():
    r = next(v for v in records if v["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P9 弱段抬升（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:9px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P9 · 弱段抬升（A01）</h1>
<div class="key"><b>方向与之前所有动态实验相反：这次只抬"低于正常水平的弱段"，强段不动。</b><br>
判据：以浊音帧电平 P75 为"正常水平"，低出 3dB 以上才处理，只补其中 60%，并有上限。<br>
两档<b>跨度很大</b>（上限差约 3 倍），目的是让你先听出"有没有效果"，之后再往回收。<br>
基线是你认可的最佳版（X1+M2+F1 最大档）。</div>
<div class="note"><b>请重点评价：</b>①弱下去的地方是否被托住了、更有发力感？②是否破坏强弱起伏（变成"每个字一样重"）？③是否把气声/呼吸一起抬起来？④是否更像自然的用力唱？</div>
<div class="card"><h3>试听</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>基线（当前最佳，未改动）</label><audio controls preload="metadata" src="audio/baseline.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">四版统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。两档强度已隐藏。<br>
客观：参考电平（浊音 P75）{records[0]['proc']['ref_level_db']}dB；有效抬升（≥1dB）覆盖 {records[0]['proc']['boosted_frame_fraction_ge1db']*100:.0f}% 的帧；弱段实测增益见报告。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-023", "phase": "F1-P9", "status": "pending_jade",
    "question": "选择性弱段抬升能否带来可听的发声力度提升",
    "method": "向上压缩：R=浊音帧P75；D=max(0,R-3-L)；g=min(GMAX,0.6D)；非浊音帧0；attack20ms/release60ms",
    "baseline": {"id": "EXP-020 S45", "sha256": base_sha},
    "target_lufs": target_lufs, "exports": exports,
    "versions": records, "sealed_not_processed": SEALED,
    "difference_vs_previous": "EXP-008 为向下压缩（压强的），本轮为向上压缩（抬弱的），方向相反",
    "no_other_modules": "只做弱段抬升；未改 F0、未加 EQ/饱和/谐波模块",
    "stop_condition": "两档均无正向改善，或破坏强弱起伏/抬起气声 → 停止本方向",
    "limitations": [
        "参考电平用浊音帧 P75，属工程近似，不等于真实'应该有多响'",
        "只处理浊音帧，浊/清边界靠 20ms 平滑过渡",
        "抬升会把弱段中的气声比例一并放大（无法区分'虚'与'弱'）",
        "自实现机制未与外部工具校验；Agent 未听音",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "gmax_db", "ratio_up", "ref_db", "boosted_frac_ge1db", "gain_med_active",
                "gain_p90", "gain_max", "weak_med", "weak_p90", "mid_med", "strong_med",
                "pitch", "crest", "clipped", "raw_lufs", "raw_tp", "matched_lufs", "hard", "flags"])
    for r in records:
        m = r.get("matched", {})
        rg = r["regions"]
        w.writerow([r["id"], r["gmax_db"], r["ratio_up"], r["proc"]["ref_level_db"],
                    r["proc"]["boosted_frame_fraction_ge1db"], r["proc"]["gain_median_when_active_db"],
                    r["proc"]["gain_p90_db"], r["proc"]["gain_max_db"],
                    rg["weak_median_db"], rg["weak_p90_db"], rg["mid_median_db"], rg["strong_median_db"],
                    r["pitch_hz"], r["crest_db"], r["clipped"], r["raw_lufs"], r["raw_true_peak_dbtp"],
                    m.get("lufs", ""), "; ".join(r["hard_fail"]), "; ".join(r["flags"])])

params = {
    "experiment": "F1-EXP-023", "phase": "F1-P9",
    "approved_scope": "选择性弱段抬升；大跨度两档（温和 ratio 0.35/GMAX 6dB，激进 ratio 1.0/GMAX 20dB）",
    "baseline": "EXP-020 S45（原样引用）", "target": TARGET_ID, "sealed": SEALED,
    "listening_target_lufs": target_lufs,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-023", "phase": "F1-P9", "status": "pending_jade",
    "target": TARGET_ID, "baseline": "EXP-020 S45",
    "gmax_db": {r["id"]: r["gmax_db"] for r in records},
    "ratio_up": {r["id"]: r["ratio_up"] for r in records},
    "ref_level_db": records[0]["proc"]["ref_level_db"],
    "boosted_fraction_ge1db": records[0]["proc"]["boosted_frame_fraction_ge1db"],
    "weak_region_gain_db": {r["id"]: r["regions"]["weak_median_db"] for r in records},
    "strong_region_gain_db": {r["id"]: r["regions"]["strong_median_db"] for r in records},
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(e["lufs"] - target_lufs) <= 0.3 for e in exports),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
    "next_rule": "有正向改善则缩小跨度细化；破坏起伏/抬起气声则停止本方向",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-023\nphase: F1-P9\n"
    f"selective upward compression on EXP-020 S45; GMAX 4/12 dB; strong sections untouched\n"
    f"sealed untouched: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "versions": [{k: r[k] for k in ("id", "gmax_db", "proc", "regions", "pitch_hz",
                                     "crest_db", "hard_fail", "flags")} for r in records],
}, ensure_ascii=False, indent=1))
