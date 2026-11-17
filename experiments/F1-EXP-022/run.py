"""F1-EXP-022 / F1-P8: 薄 B 局部动态下陷诊断（只读，不生成处理音频）。

问题：薄音版是否存在比正常版更明显、更频繁的非刻意局部能量下陷？
方法：
  1) 4 对自录（薄音版 / 正常版），统一响度到同一 LUFS 后分析；
  2) 25ms 窗 / 10ms 步计算短时 RMS 与归一化自相关峰值 r（浊音判据 r≥0.45）；
  3) 局部基准 = 前后各 600ms 内浊音帧能量的 P75（近似"所在乐句正常水平"）；
  4) dip = 局部基准 − 当前能量（dB，正值=下陷）。dip 是相对量，不随响度归一化改变；
  5) 事件 = 连续 dip ≥ 3dB 且持续 ≥ 100ms；
  6) 排除：静音/换气（浊音掩码）、清辅音（浊音掩码）、句尾渐弱（事件后 400ms 内无浊音）；
  7) 疑似刻意弱唱（dip>12dB 或持续>1.5s）单独标记，不进主统计；
  8) 4 对同向性检验 + DTW 对齐质量报告。
"""
import csv, hashlib, json, math, os, pathlib, platform, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav


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

OUT = ROOT / "experiments/F1-EXP-022"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-022 outputs already exist")
for d in (OUT / "qa/plots",):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

PAIRS = [
    ("平凡之路", "A_self_01", "A_self_02"),
    ("我们的歌", "A_self_03", "A_self_04"),
    ("雅俗共赏", "A_self_05", "A_self_06"),
    ("因为刚好遇见你", "A_self_07", "A_self_08"),
]
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]

WIN_S, HOP_S = 0.025, 0.010
FMIN, FMAX = 90.0, 450.0
R_VOICED = 0.45
GATE_FLOOR_DB = 45.0
CTX_SCALES = [("local", 0.600), ("phrase", 1.500)]   # 基准半窗：1.2s（音节级）与 3.0s（乐句级）
DIP_MIN_DB = 3.0
DIP_MIN_FRAMES = 10      # 100ms
TAIL_EXCLUDE_S = 0.400   # 事件后无浊音 → 视为句尾渐弱
SUSPECT_DIP_DB = 12.0
SUSPECT_DUR_S = 1.5
TARGET_LUFS = -20.0


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


def load(did):
    it = items[did]
    wav = DATA / it["wav"]["path"]
    raw = DATA / it["raw"]["path"]
    if sha(raw) != it["raw"]["sha256"] or sha(wav) != it["wav"]["sha256"]:
        raise ValueError(f"input changed: {did}")
    x, sr = read_pcm16(wav)
    if len(x) != it["wav"]["frames"]:
        raise ValueError(f"frames mismatch: {did}")
    return x, sr, it


def analyze(x, sr, ctx_s):
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    starts = list(range(0, len(x) - win + 1, hop))
    rms, rv, f0v = [], [], []
    for s in starts:
        fr = x[s:s + win]
        rms.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        f0, r = frame_f0_ac(fr, sr)
        rv.append(r)
        f0v.append(f0)
    rms = np.array(rms)
    rv = np.array(rv)
    gate = rms.max() - GATE_FLOOR_DB
    voiced = (rv >= R_VOICED) & (rms >= gate)
    if voiced.sum() < 100:
        raise RuntimeError("浊音帧过少，元音识别不可靠")

    W = int(ctx_s / HOP_S)
    n = len(rms)
    base = np.full(n, np.nan)
    vi = np.where(voiced)[0]
    for i in vi:
        lo, hi = max(0, i - W), min(n, i + W + 1)
        seg = rms[lo:hi][voiced[lo:hi]]
        if seg.size >= 20:
            base[i] = float(np.percentile(seg, 75))
    dip = np.where(voiced & np.isfinite(base), base - rms, np.nan)

    # 事件检测（连续 dip>=3）
    events = []
    i = 0
    while i < n:
        if np.isfinite(dip[i]) and dip[i] >= DIP_MIN_DB:
            j = i
            while j + 1 < n and np.isfinite(dip[j + 1]) and dip[j + 1] >= DIP_MIN_DB:
                j += 1
            if (j - i + 1) >= DIP_MIN_FRAMES:
                events.append([i, j])
            i = j + 1
        else:
            i += 1

    # 排除句尾渐弱
    tail_excluded = 0
    kept = []
    for a, b in events:
        after = np.where(voiced[b + 1:])[0]
        gap_s = (after[0] + 1) * HOP_S if after.size else 1e9
        if gap_s > TAIL_EXCLUDE_S:
            tail_excluded += 1
        else:
            kept.append([a, b])

    # 疑似刻意弱唱
    suspect = []
    main = []
    for a, b in kept:
        seg = dip[a:b + 1]
        dur = (b - a + 1) * HOP_S
        mx = float(np.nanmax(seg))
        if mx > SUSPECT_DIP_DB or dur > SUSPECT_DUR_S:
            suspect.append({"start_s": round(a * HOP_S, 2), "dur_s": round(dur, 2),
                            "max_dip_db": round(mx, 1)})
        else:
            main.append({"start_s": round(a * HOP_S, 2), "dur_s": round(dur, 2),
                         "max_dip_db": round(mx, 1), "mean_dip_db": round(float(np.nanmean(seg)), 1)})

    total_sec = len(x) / sr
    voiced_sec = voiced.sum() * HOP_S
    event_frames = sum(b - a + 1 for a, b in kept)
    valid = np.isfinite(dip)
    stats = {
        "ctx_s": ctx_s,
        "duration_s": round(total_sec, 2),
        "voiced_fraction": round(float(voiced.mean()), 3),
        "voiced_seconds": round(voiced_sec, 2),
        "events_excluded_tail": tail_excluded,
        "events_suspect_intentional": len(suspect),
        "events_main": len(main),
        "events_all_before_exclusion": len(events),
        "events_per_min": round(len(main) / (voiced_sec / 60.0), 2) if voiced_sec > 0 else None,
        "deep_dip_count_per_min": round(len(suspect) / (voiced_sec / 60.0), 2) if voiced_sec > 0 else None,
        "dip_event_max_median_db": round(float(np.median([e["max_dip_db"] for e in main])), 1) if main else None,
        "dip_event_max_p90_db": round(float(np.percentile([e["max_dip_db"] for e in main], 90)), 1) if main else None,
        "dip_event_dur_median_s": round(float(np.median([e["dur_s"] for e in main])), 2) if main else None,
        "dip_time_fraction": round(float(event_frames / max(voiced.sum(), 1)), 3),
        "dip_mean_all_voiced_db": round(float(np.nanmean(dip[valid])), 2),
        "dip_p90_all_voiced_db": round(float(np.nanpercentile(dip[valid], 90)), 2),
        "dip_max_all_voiced_db": round(float(np.nanmax(dip[valid])), 2),
        "voiced_energy_sd_db": round(float(np.nanstd(rms[voiced])), 2),
        "voiced_energy_mean_db": round(float(np.nanmean(rms[voiced])), 2),
    }
    return {"stats": stats, "main_events": main, "suspect": suspect,
            "rms": rms, "base": base, "dip": dip, "voiced": voiced, "hop_s": HOP_S}


def dtw_align(a, b, band_ratio=0.15):
    """带 Sakoe-Chiba 带的 DTW，返回平均绝对差与相关系数（对齐后）。"""
    n, m = len(a), len(b)
    band = max(10, int(band_ratio * max(n, m)))
    INF = 1e18
    D = np.full((n, m), INF)
    D[0, 0] = abs(a[0] - b[0])
    for i in range(1, n):
        center = int(i * m / n)
        lo, hi = max(1, center - band), min(m, center + band + 1)
        for j in range(lo, hi):
            c = abs(a[i] - b[j])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # 回溯
    i, j = n - 1, m - 1
    idx = []
    while i > 0 and j > 0:
        idx.append((i, j))
        prev = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if prev == 0:
            i, j = i - 1, j - 1
        elif prev == 1:
            i -= 1
        else:
            j -= 1
    idx.append((0, 0))
    idx = np.array(idx[::-1])
    pa = a[idx[:, 0]]
    pb = b[idx[:, 1]]
    mad = float(np.mean(np.abs(pa - pb)))
    cc = float(np.corrcoef(pa, pb)[0, 1])
    return mad, cc, len(idx)


results = {}
plot_data = []
for name, thin_id, normal_id in PAIRS:
    x_t, sr, it_t = load(thin_id)
    x_n, sr_n, it_n = load(normal_id)
    if sr != sr_n:
        raise RuntimeError("sample rate mismatch within pair")
    lt, ln = measure_loudness(x_t, sr), measure_loudness(x_n, sr)
    gt, gn = TARGET_LUFS - lt["integrated_lufs"], TARGET_LUFS - ln["integrated_lufs"]
    x_t_n = x_t * (10 ** (gt / 20))
    x_n_n = x_n * (10 ** (gn / 20))

    scales = {}
    local_a = None
    for scale_name, ctx_s in CTX_SCALES:
        a_t = analyze(x_t_n, sr, ctx_s)
        a_n = analyze(x_n_n, sr, ctx_s)
        if scale_name == "local":
            local_a = (a_t, a_n)
            plot_data.append((name, a_t, a_n))
        st, sn = a_t["stats"], a_n["stats"]
        delta = {k: (round(st[k] - sn[k], 2) if (st[k] is not None and sn[k] is not None) else None)
                 for k in ("events_per_min", "deep_dip_count_per_min", "dip_event_max_median_db",
                           "dip_event_max_p90_db", "dip_time_fraction", "dip_mean_all_voiced_db",
                           "dip_p90_all_voiced_db", "voiced_energy_sd_db", "voiced_energy_mean_db")}
        scales[scale_name] = {
            "ctx_s": ctx_s,
            "thin_stats": st, "normal_stats": sn, "delta_thin_minus_normal": delta,
            "thin_events": a_t["main_events"], "normal_events": a_n["main_events"],
            "thin_suspect": a_t["suspect"], "normal_suspect": a_n["suspect"],
        }

    # 对齐质量（在统一响度后的原始能量包络上，用 z 分数）
    a_t0, a_n0 = local_a
    ea = (a_t0["rms"] - np.mean(a_t0["rms"])) / max(np.std(a_t0["rms"]), 1e-9)
    eb = (a_n0["rms"] - np.mean(a_n0["rms"])) / max(np.std(a_n0["rms"]), 1e-9)
    mad, cc, path_len = dtw_align(ea, eb)

    results[name] = {
        "thin_id": thin_id, "normal_id": normal_id,
        "loudness_before": {"thin": round(lt["integrated_lufs"], 2), "normal": round(ln["integrated_lufs"], 2)},
        "gain_applied": {"thin": round(gt, 2), "normal": round(gn, 2)},
        "scales": scales,
        "dtw": {"mean_abs_dev_db": round(mad, 2), "corr": round(cc, 3), "path_len": path_len},
    }

# ---- 同向性判定（按尺度） ----
KEYS = ["events_per_min", "deep_dip_count_per_min", "dip_event_max_median_db",
        "dip_event_max_p90_db", "dip_time_fraction", "dip_mean_all_voiced_db",
        "dip_p90_all_voiced_db", "voiced_energy_sd_db"]
summary_dir = {}
for scale_name, _ in CTX_SCALES:
    per = {}
    for k in KEYS:
        ds = [results[n]["scales"][scale_name]["delta_thin_minus_normal"][k] for n, _, _ in PAIRS]
        ds = [d for d in ds if d is not None]
        if not ds:
            per[k] = {"n": 0, "positive_pairs": 0, "negative_pairs": 0, "same_sign": None, "median_delta": None}
            continue
        pos = sum(1 for d in ds if d > 0)
        neg = sum(1 for d in ds if d < 0)
        per[k] = {"n": len(ds), "positive_pairs": pos, "negative_pairs": neg,
                  "same_sign": max(pos, neg), "median_delta": round(float(np.median(ds)), 2)}
    summary_dir[scale_name] = per

consistency = {}
for scale_name, _ in CTX_SCALES:
    consistency[scale_name] = {
        "keys_all_four_same_sign": sum(1 for k in KEYS if summary_dir[scale_name][k]["same_sign"] == 4),
        "keys_at_least_three_same_sign": sum(1 for k in KEYS if summary_dir[scale_name][k]["same_sign"] in (4, 3)),
        "keys_tested": len(KEYS),
    }

# ---- 可视化 ----
for name, at, an in plot_data:
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=False)
    for ax, a, title in ((axes[0], at, "薄音版"), (axes[1], an, "正常版")):
        t = np.arange(len(a["rms"])) * a["hop_s"]
        ax.plot(t, a["rms"], lw=0.8, color="#2f3640", label="短时能量")
        ax.plot(t, a["base"], lw=1.0, color="#0984e3", alpha=0.8, label="局部基准(P75)")
        for e in a["main_events"]:
            ax.axvspan(e["start_s"], e["start_s"] + e["dur_s"], color="#e17055", alpha=0.22)
        ax.set_title(f"{name} · {title}  （红色=合格局部下陷事件）", fontsize=11)
        ax.set_ylabel("dB")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(OUT / f"qa/plots/{name}.png", dpi=130)
    plt.close(fig)

# ---- 输出 ----
fp = OUT / "qa/diagnostics.json"
fp.write_text(json.dumps({
    "experiment": "F1-EXP-022", "phase": "F1-P8", "status": "diagnosis_only",
    "question": "薄音版是否存在比正常版更明显、更频繁的非刻意局部能量下陷",
    "settings": {
        "win_s": WIN_S, "hop_s": HOP_S, "r_voiced": R_VOICED, "gate_floor_db": GATE_FLOOR_DB,
        "context_scales": [{"name": n, "half_window_s": c} for n, c in CTX_SCALES],
        "local_baseline_percentile": 75,
        "dip_min_db": DIP_MIN_DB, "dip_min_frames": DIP_MIN_FRAMES,
        "tail_exclude_s": TAIL_EXCLUDE_S,
        "suspect_dip_db": SUSPECT_DIP_DB, "suspect_dur_s": SUSPECT_DUR_S,
        "target_lufs": TARGET_LUFS,
        "note": "dip 为相对量，响度归一化不改变其数值",
    },
    "pairs": results,
    "direction_summary": summary_dir,
    "consistency": consistency,
    "sealed_not_processed": SEALED,
    "limitations": [
        "局部基准取±0.6s/±1.5s 内浊音帧 P75，是'乐句正常水平'的近似，不等于真实乐句切分",
        "浊音判据用自相关周期性，气声极重的段可能被漏判",
        "句尾渐弱按'事件后 400ms 内无浊音'排除，可能误排真正的长停顿前弱唱",
        "深度下陷（>12dB）单列为'疑似刻意弱唱/间隙'，未计入主统计——但也可能是真实塌陷，需人工核对时间点",
        "DTW 只用于验证两版是否在唱同一段，不用于事件统计（事件在原始时间轴上计）",
        "Agent 未听音，'是否有听感意义'需用户确认",
    ],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair", "scale", "version", "events_main", "events_per_min", "deep_per_min",
                "dip_max_med", "dip_max_p90", "dur_med_s", "dip_time_frac", "dip_mean_all",
                "dip_p90_all", "voiced_frac", "energy_sd", "excluded_tail", "suspect"])
    for name, _, _ in PAIRS:
        r = results[name]
        for scale_name, _ in CTX_SCALES:
            sc = r["scales"][scale_name]
            for ver, key in (("thin", "thin_stats"), ("normal", "normal_stats")):
                s = sc[key]
                w.writerow([name, scale_name, ver, s["events_main"], s["events_per_min"],
                            s["deep_dip_count_per_min"], s["dip_event_max_median_db"],
                            s["dip_event_max_p90_db"], s["dip_event_dur_median_s"],
                            s["dip_time_fraction"], s["dip_mean_all_voiced_db"],
                            s["dip_p90_all_voiced_db"], s["voiced_fraction"],
                            s["voiced_energy_sd_db"], s["events_excluded_tail"],
                            s["events_suspect_intentional"]])

# ---- HTML 报告页 ----
def fmt(v, plus=False):
    if v is None:
        return "—"
    return f"{v:+}" if plus else f"{v}"


rows = []
for name, _, _ in PAIRS:
    r = results[name]
    for scale_name, _ in CTX_SCALES:
        sc = r["scales"][scale_name]
        d = sc["delta_thin_minus_normal"]
        rows.append(f"""<tr><td>{name}</td><td>{scale_name}</td>
<td>{fmt(sc['thin_stats']['events_per_min'])}</td><td>{fmt(sc['normal_stats']['events_per_min'])}</td><td>{fmt(d['events_per_min'], True)}</td>
<td>{fmt(sc['thin_stats']['deep_dip_count_per_min'])}</td><td>{fmt(sc['normal_stats']['deep_dip_count_per_min'])}</td><td>{fmt(d['deep_dip_count_per_min'], True)}</td>
<td>{fmt(sc['thin_stats']['dip_time_fraction'])}</td><td>{fmt(sc['normal_stats']['dip_time_fraction'])}</td><td>{fmt(d['dip_time_fraction'], True)}</td>
<td>{fmt(sc['thin_stats']['dip_p90_all_voiced_db'])}</td><td>{fmt(sc['normal_stats']['dip_p90_all_voiced_db'])}</td><td>{fmt(d['dip_p90_all_voiced_db'], True)}</td>
<td>{r['dtw']['corr']}</td></tr>""")

summary_html = ""
for scale_name, _ in CTX_SCALES:
    summary_html += f"<h3>尺度：{scale_name}</h3><table><tr><th>指标</th><th>同向对数</th><th>中位差值（薄−正常）</th><th>正/负</th></tr>"
    for k in KEYS:
        v = summary_dir[scale_name][k]
        summary_html += (f"<tr><td>{k}</td><td>{v['same_sign']}/{v['n']}</td>"
                         f"<td>{fmt(v['median_delta'], True)}</td>"
                         f"<td>{v['positive_pairs']}/{v['negative_pairs']}</td></tr>")
    summary_html += "</table>"

html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1-P8 薄B局部下陷诊断</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:1080px;margin:26px auto;padding:0 18px}}
h1{{font-size:23px}}h2{{font-size:18px;margin-top:26px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}
th,td{{border:1px solid #dce0e6;padding:6px 8px;text-align:center}}th{{background:#eef1f5}}
img{{width:100%;border:1px solid #dce0e6;border-radius:8px;margin:10px 0;background:#fff}}
.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}}</style>
<h1>F1-P8 · 薄 B 局部动态下陷诊断（只读）</h1>
<div class="note">判定标准：下陷 = 相对"所在处浊音帧 P75 基准"低 3dB 以上、持续 ≥100ms；已排除静音/换气/清辅音/句尾渐弱；疑似刻意弱唱（&gt;12dB 或 &gt;1.5s）单列不计入主统计。<br>
红色阴影 = 合格的下陷事件。请重点看：<b>薄音版的红块是不是更多、更深、更集中在句中</b>。</div>
<h2>配对统计（两尺度）</h2>
<table><tr><th rowspan="2">曲目</th><th rowspan="2">尺度</th><th colspan="3">中等下陷频率（次/分）</th><th colspan="3">深度下陷频率（次/分，&gt;12dB）</th><th colspan="3">下陷时间占比</th><th colspan="3">浊音帧下陷 P90（dB）</th><th rowspan="2">对齐相关</th></tr>
<tr><th>薄</th><th>正常</th><th>差</th><th>薄</th><th>正常</th><th>差</th><th>薄</th><th>正常</th><th>差</th><th>薄</th><th>正常</th><th>差</th></tr>
{''.join(rows)}</table>
<h2>四对同向性</h2>
{summary_html}
<h2>可视化（红块=合格下陷事件）</h2>
{''.join(f'<img src="qa/plots/{n}.png" alt="{n}">' for n,_,_ in PAIRS)}
</html>"""
(OUT / "report.html").write_text(html, encoding="utf-8")

summary = {
    "experiment": "F1-EXP-022", "phase": "F1-P8", "status": "diagnosis_only",
    "pairs": [n for n, _, _ in PAIRS],
    "scales": [s for s, _ in CTX_SCALES],
    "direction_summary": summary_dir,
    "consistency": consistency,
    "report_page": "report.html",
    "diagnostics": "qa/diagnostics.json", "pairs_csv": "qa/pairs.csv",
    "sealed_not_processed": SEALED,
    "no_processing": "只读诊断，未生成任何处理音频，未修改现有处理主线",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "params.json").write_text(json.dumps({
    "experiment": "F1-EXP-022", "phase": "F1-P8",
    "read_only": True,
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-022\nphase: F1-P8\n"
    f"read-only dip diagnosis on 4 self-recorded pairs; no processing audio generated\n"
    f"sealed untouched: {', '.join(SEALED)}\n", encoding="utf-8")

print(json.dumps({"summary": summary, "pairs": results}, ensure_ascii=False, indent=1))
