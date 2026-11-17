"""F1-EXP-009 / F1-DIAG-1: 薄音成因诊断——自录 8 份（4 对 薄音/正常 对照）只读声学分析。

问题：为什么 EQ / tanh 饱和 / 并行压缩都无法带来稳定可听的饱满度改善？
假设：薄音与正常版的差异主要在声源侧（谐波结构、声源谱倾斜、气声比例），
而非静态频谱包络或电平分布；若成立，包络类(EQ)、宽带非线性(饱和)、
电平域(压缩)处理都无法复现正常版的听感。
方法：同曲成对对照（同一人、同一手机录音链），帧级有声谐波分析 +
谐噪比代理 + 简化 CPP + LTAS 频段能量 + 控制变量（F0/响度/动态）。
只读分析：不生成音频、不改素材、不做听感结论（Agent 无听音能力）。
"""
import csv, hashlib, json, pathlib, subprocess, sys, shutil, os
from datetime import datetime, timezone
import numpy as np
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, sample_stats, frame_rms


def ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return
    links = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
        return
    raise SystemExit("ffmpeg.exe not found; add its directory to PATH.")


ensure_ffmpeg_on_path()

OUT = ROOT / "experiments/F1-EXP-009"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-009 outputs already exist")
(OUT / "qa").mkdir(parents=True, exist_ok=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

# 4 对同曲对照（用户自录，薄音在前）
PAIRS = [
    ("平凡之路", "A_self_01", "薄音版", "A_self_02", "正常版"),
    ("我们的歌", "A_self_03", "薄音版", "A_self_04", "正常版"),
    ("雅俗共赏", "A_self_05", "清唱版(薄)", "A_self_06", "正常版"),
    ("因为刚好遇见你", "A_self_07", "清唱版(薄)", "A_self_08", "原版(正常)"),
]

# --- 分析配置（固定并记录） ---
FMIN, FMAX = 90.0, 450.0        # 唱歌 F0 搜索范围（自录样本 f0 中值 136-226，p90 至 350）
WIN_S, HOP_S = 0.060, 0.020     # 帧长 60ms / 步进 20ms
R_GATE = 0.35                    # 有声谐波分析帧的自相关周期性门限
FLOOR_DB = 40.0                  # 帧能量门限：低于最响帧 -40dB 判为非有声
HMAX_HZ = 8000.0                 # 谐波分析上限
BANDS = [(20, 80), (80, 300), (300, 500), (500, 1000), (1000, 2000),
         (2000, 4000), (4000, 6000), (6000, 8000), (80, 8000)]

NFFT_H = 1 << 15  # 谐波幅度采样用零填充 FFT（32768，bin≈1.46Hz）


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def frame_f0_ac(win, sr):
    """60ms 帧：去均值 + 汉宁窗，自相关基音估计（抛物线插值）。
    返回 (f0, 周期性 r)。自实现，未经 Praat 校验，仅用于同链路相对比较。"""
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
        k_f = k + d
        r = y1
    else:
        k_f, r = float(k), float(c[k])
    f0 = sr / k_f
    if not (FMIN <= f0 <= FMAX):
        return float("nan"), float(r)
    return float(f0), float(r)


def harmonic_amps_db(win, sr, f0):
    """按帧估计 H1..HK 谐波幅度（dB）。窗口汉宁 + 零填充 FFT 取最近 bin。
    幅度归一：amp = 2|X(f)|/sum(hann)，近似正弦峰值幅度。"""
    n = len(win)
    w = np.hanning(n) * win
    spec = np.fft.rfft(w, NFFT_H)
    gain = np.sum(np.hanning(n))
    amps, ks = [], []
    k = 1
    while k * f0 <= HMAX_HZ:
        idx = int(round(k * f0 * NFFT_H / sr))
        if idx >= len(spec):
            break
        amps.append(20.0 * np.log10(max(2.0 * abs(spec[idx]) / gain, 1e-10)))
        ks.append(k)
        k += 1
    return np.array(ks), np.array(amps)


def cpp_frame_db(win, sr):
    """简化 CPP：实倒谱在基音周期处的峰值超出倒谱线性基线的高度。
    自实现简化版（基线取搜索范围内线性回归），未与标准 CPPS 实现对齐，
    只用于同录音链文件间相对比较。"""
    n = len(win)
    w = np.hanning(n) * (win - win.mean())
    N = 1 << 13
    mag = np.abs(np.fft.rfft(w, N)) + 1e-10
    logmag = 20.0 * np.log10(mag)
    cep = np.fft.irfft(logmag, N)
    q_lo, q_hi = int(np.ceil(sr / FMAX)), int(sr / FMIN)
    if q_hi >= len(cep) // 2:
        return float("nan")
    seg = cep[q_lo:q_hi + 1]
    k = int(np.argmax(seg))
    x = np.arange(q_lo, q_hi + 1)
    coef = np.polyfit(x, cep[q_lo:q_hi + 1], 1)
    return float(cep[q_lo + k] - np.polyval(coef, q_lo + k))


def analyze_file(item):
    wav = DATA / item["wav"]["path"]
    raw = DATA / item["raw"]["path"]
    if sha(raw) != item["raw"]["sha256"] or sha(wav) != item["wav"]["sha256"]:
        raise ValueError(f"input changed: {item['id']}")
    x, sr = read_pcm16(wav)
    win, hop = int(round(sr * WIN_S)), int(round(sr * HOP_S))
    if len(x) < win:
        raise RuntimeError(f"too short: {item['id']}")

    # 有声门限：60ms 帧 RMS 相对帧最大值
    starts = np.arange(0, len(x) - win + 1, hop)
    frms_db = 20.0 * np.log10(np.maximum(frame_rms(x, sr, WIN_S), 1e-12))
    gate_db = frms_db.max() - FLOOR_DB

    rows = []
    for s in starts:
        seg = x[s:s + win]
        rdb = 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-12))
        if rdb < gate_db:
            continue
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        ks, amps = harmonic_amps_db(seg, sr, f0)
        if len(ks) < 4:
            continue
        h1h2 = float(amps[0] - amps[1]) if len(ks) >= 2 else float("nan")
        h1h3 = float(amps[0] - amps[2]) if len(ks) >= 3 else float("nan")
        h2h4 = float(amps[1] - amps[3]) if len(ks) >= 4 else float("nan")
        slope = float(np.polyfit(np.log2(ks), amps, 1)[0])  # dB/倍频程
        richness = float(10.0 * np.log10(max(np.sum(10 ** (amps[1:] / 10)) / 10 ** (amps[0] / 10), 1e-10)))
        r_c = min(max(r, 1e-4), 0.9999)
        hnr = float(10.0 * np.log10(r_c / (1.0 - r_c)))
        cpp = cpp_frame_db(seg, sr)
        rows.append((f0, r, h1h2, h1h3, h2h4, slope, richness, hnr, cpp))

    if not rows:
        raise RuntimeError(f"no voiced frames: {item['id']}")
    a = np.array(rows)  # N×9
    def med(col): return float(np.median(a[:, col]))
    def p(col, q): return float(np.percentile(a[:, col], q))

    # LTAS 频段能量（Welch）
    f, psd = signal.welch(x, sr, nperseg=min(8192, len(x)))
    band_db = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        band_db[f"{lo}_{hi}"] = float(10.0 * np.log10(max(float(np.sum(psd[m])), 1e-20)))
    total_db = band_db["80_8000"]
    rel = {k: round(v - total_db, 2) for k, v in band_db.items()}

    loud = measure_loudness(x, sr)
    st = sample_stats(x, sr)
    f_, p_ = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m_ = (f_ >= 20) & (f_ < 8000)
    f_, p_ = f_[m_], p_[m_]
    tot = max(float(np.sum(p_)), 1e-20)
    centroid = float(np.sum(f_ * p_) / tot)

    return {
        "dataset_id": item["id"], "song": item.get("song", ""), "variant": item.get("variant", ""),
        "frames_total": int(len(starts)), "frames_analyzed": int(len(rows)),
        "voiced_fraction_of_active": round(len(rows) / max(int(np.count_nonzero(frms_db >= gate_db)), 1), 3),
        "f0_median_hz": round(med(0), 1), "f0_p10_hz": round(p(0, 10), 1), "f0_p90_hz": round(p(0, 90), 1),
        "periodicity_median": round(med(1), 3),
        "h1_minus_h2_db": round(med(2), 2), "h1_minus_h3_db": round(med(3), 2), "h2_minus_h4_db": round(med(4), 2),
        "harm_slope_db_per_oct": round(med(5), 2),
        "harm_richness_db": round(med(6), 2),
        "hnr_proxy_db": round(med(7), 2),
        "cpp_db": round(med(8), 2),
        "h1h2_iqr": [round(p(2, 25), 2), round(p(2, 75), 2)],
        "slope_iqr": [round(p(5, 25), 2), round(p(5, 75), 2)],
        "lufs": loud["integrated_lufs"], "lra": loud["loudness_range_lu"],
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "rms_dbfs": round(st["rms_dbfs"], 2),
        "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "centroid_hz": round(centroid, 0),
        "band_abs_db": {k: round(v, 2) for k, v in band_db.items()},
        "band_rel_db": rel,
        "wav_sha256": sha(wav),
    }


# ---------------- 运行 ----------------
files = {}
inputs_meta = []
for song, tid, tlab, nid, nlab in PAIRS:
    for did, lab in ((tid, tlab), (nid, nlab)):
        m = files.setdefault(did, analyze_file(items[did]))
        m["role"] = lab
        m["pair_song"] = song
        inputs_meta.append({"dataset_id": did, "song": song, "role": lab})

(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-009", "phase": "F1-DIAG-1",
    "purpose": "薄音成因诊断：4 对同曲 薄音/正常 对照（同录音链、同歌手），只读分析",
    "hypothesis": "薄/正常差异主要在声源侧（谐波结构/声源谱倾斜/气声比例），而非频谱包络或电平分布；"
                  "若成立，可解释 EQ/饱和/压缩三路线失败",
    "items": inputs_meta,
    "config": {"f0_search_hz": [FMIN, FMAX], "win_s": WIN_S, "hop_s": HOP_S,
               "periodicity_gate": R_GATE, "active_floor_db": FLOOR_DB,
               "harmonic_ceiling_hz": HMAX_HZ, "bands": [list(b) for b in BANDS]},
    "method_notes": [
        "谐波幅度/斜率/H1-H2/丰富度：自实现帧级谐波 DFT 采样（60ms 汉宁窗，零填充 32768）",
        "HNR 代理：归一化自相关峰值 r 换算 10log10(r/(1-r))，非 Praat 谐噪比",
        "CPP：简化实倒谱峰值突出度（线性基线回归），未与标准 CPPS 对齐",
        "以上为相对比较量，仅用于同一录音链内的成对比较；未做 Praat 校验（列为未验证事项）",
        "标签 薄音/正常 由用户指定，Agent 未听音；分析为声学结构差异描述，非听感结论",
    ],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 配对差值 ----------------
KEYS = ["f0_median_hz", "h1_minus_h2_db", "h1_minus_h3_db", "h2_minus_h4_db",
        "harm_slope_db_per_oct", "harm_richness_db", "hnr_proxy_db", "cpp_db",
        "periodicity_median", "lufs", "rms_dbfs", "crest_db", "lra",
        "band_rel_db"]
pair_records = []
for song, tid, tlab, nid, nlab in PAIRS:
    t, n = files[tid], files[nid]
    delta = {}
    for k in KEYS:
        if k == "band_rel_db":
            delta[k] = {b: round(n["band_rel_db"][b] - t["band_rel_db"][b], 2) for b in t["band_rel_db"]}
        else:
            delta[k] = round(n[k] - t[k], 2)
    pair_records.append({
        "song": song, "thin": {"id": tid, "label": tlab}, "normal": {"id": nid, "label": nlab},
        "thin_metrics": {k: t[k] for k in KEYS if k != "band_rel_db"},
        "normal_metrics": {k: n[k] for k in KEYS if k != "band_rel_db"},
        "delta_normal_minus_thin": delta,
    })

# 跨对一致性：核心声源指标方向统计
core = ["h1_minus_h2_db", "harm_slope_db_per_oct", "harm_richness_db", "hnr_proxy_db", "cpp_db", "f0_median_hz", "lufs"]
consistency = {}
for k in core:
    signs = [("+" if pr["delta_normal_minus_thin"][k] > 0 else "-" if pr["delta_normal_minus_thin"][k] < 0 else "0")
             for pr in pair_records]
    consistency[k] = {"signs_per_pair": signs,
                      "all_same_sign": len(set(s for s in signs if s != "0")) == 1,
                      "median_delta": round(float(np.median([pr["delta_normal_minus_thin"][k] for pr in pair_records])), 2)}

qa = {
    "experiment": "F1-EXP-009", "phase": "F1-DIAG-1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "files": [files[i] for i in sorted(files)],
    "pairs": pair_records,
    "cross_pair_consistency": consistency,
    "limitations": [
        "同曲不同 take：歌词段落/音区分布可能不完全一致，F0 差异需对照看",
        "H1/H2 幅度受元音共振峰位置影响；同一人同曲近似受控，但不做共振峰校正（列为局限）",
        "自实现声学量未与 Praat 等标准实现校验，仅用于同链路相对比较",
        "手机录音链路本身高频衰减：跨频段绝对值不代表录音棚水平，但成对内一致",
        "标签由用户指定；本报告不构成听感结论",
    ],
}
(OUT / "qa/analysis.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    cols = ["pair_song", "role", "dataset_id", "variant", "f0_median_hz", "h1_minus_h2_db", "h1_minus_h3_db",
            "h2_minus_h4_db", "harm_slope_db_per_oct", "harm_richness_db", "hnr_proxy_db", "cpp_db",
            "periodicity_median", "lufs", "rms_dbfs", "crest_db", "lra", "centroid_hz",
            "rel_80_300", "rel_500_1000", "rel_1000_2000", "rel_2000_4000", "rel_4000_6000", "rel_6000_8000", "frames_analyzed"]
    w.writerow(cols)
    for did in sorted(files):
        m = files[did]
        r = m["band_rel_db"]
        w.writerow([m["pair_song"], m["role"], did, m["variant"], m["f0_median_hz"], m["h1_minus_h2_db"],
                    m["h1_minus_h3_db"], m["h2_minus_h4_db"], m["harm_slope_db_per_oct"], m["harm_richness_db"],
                    m["hnr_proxy_db"], m["cpp_db"], m["periodicity_median"], m["lufs"], m["rms_dbfs"],
                    m["crest_db"], m["lra"], m["centroid_hz"],
                    r["80_300"], r["500_1000"], r["1000_2000"], r["2000_4000"], r["4000_6000"], r["6000_8000"],
                    m["frames_analyzed"]])

with open(OUT / "qa/pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["song", "metric", "thin", "normal", "delta_normal_minus_thin"])
    for pr in pair_records:
        for k in [k for k in KEYS if k != "band_rel_db"]:
            w.writerow([pr["song"], k, pr["thin_metrics"][k], pr["normal_metrics"][k], pr["delta_normal_minus_thin"][k]])

(OUT / "summary.json").write_text(json.dumps({
    "experiment": "F1-EXP-009", "phase": "F1-DIAG-1", "status": "analysis_complete",
    "sample_count": 8, "pairs": [p["song"] for p in pair_records],
    "read_only": True, "audio_generated": False,
    "report": "RESULT.md", "full_data": "qa/analysis.json",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-009\n"
    f"read-only analysis of 8 self recordings (4 thin/normal pairs); no audio generated; no holdout processing\n",
    encoding="utf-8")

print(json.dumps({
    "pairs": [{ "song": pr["song"], "delta": {k: pr["delta_normal_minus_thin"][k] for k in core}}
              for pr in pair_records],
    "consistency": consistency,
}, ensure_ascii=False, indent=1))
