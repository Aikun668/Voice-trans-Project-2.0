"""F1-EXP-012 / F1-DIAG-2: 时域 / 相位 / 共振峰 三项对照诊断。

背景：静态频谱参数已对齐到正常版 91-97%（EXP-010/011），但听感仅为"音量加大一点点"，
     说明频谱幅度类别对"厚度"感知无效。本轮只做诊断，不做处理，回答一个问题：
     剩余未查的三类维度，是否真正区分薄音与正常版？

三项假设：
  H1 时域：薄音每个振动周期的能量更集中在前段（脉冲式），正常版更均匀
  H2 相位：薄音谐波相位更散乱（气声/闭合不足），正常版更同相（脉冲锁定）
  H3 共振峰：薄音 F1/F2 带宽更宽（阻尼大 / 共鸣不足）

方法：4 对同曲对照（同一人、同一录音链），逐帧测量 + 配对比较；
      4/4 同向才判"有区分"。只读分析，不生成音频，不做听感结论。
"""
import csv, hashlib, json, math, os, pathlib, platform, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness


def ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return
    links = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
        return
    raise SystemExit("ffmpeg.exe not found; add its directory to PATH.")


ensure_ffmpeg_on_path()

OUT = ROOT / "experiments/F1-EXP-012"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-012 outputs already exist")
(OUT / "qa").mkdir(parents=True, exist_ok=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

PAIRS = [
    ("平凡之路", "A_self_01", "薄音版", "A_self_02", "正常版"),
    ("我们的歌", "A_self_03", "薄音版", "A_self_04", "正常版"),
    ("雅俗共赏", "A_self_05", "清唱版(薄)", "A_self_06", "正常版"),
    ("因为刚好遇见你", "A_self_07", "清唱版(薄)", "A_self_08", "原版(正常)"),
]

# --- 配置（与 F1-EXP-009 一致的帧门限，保证可比） ---
FMIN, FMAX = 90.0, 450.0
WIN_S, HOP_S = 0.060, 0.020
R_GATE = 0.35
FLOOR_DB = 40.0
CYCLE_POINTS = 64          # 折叠后每周期采样点数
PHASE_KMAX = 10            # 相位相干性用到 H1..H10（≤5000Hz）
FORMANT_SR = 16000         # LPC 分析采样率
LPC_ORDER = 14
FORMANT_MAX_HZ = 4000.0
FORMANT_BW_MAX = 700.0


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def frame_f0_ac(win, sr):
    """自相关基音估计（与 F1-EXP-009 相同），返回 (f0, 周期性 r)。"""
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


# ---------------- H1: 周期内能量分布 ----------------
def cycle_energy(seg, sr, f0):
    """周期折叠（epoch folding）：按 F0 把帧内各周期对齐平均，
    以最大绝对值点为相位原点（近似声门冲击点），输出能量分布指标。"""
    T = sr / f0
    n_cycles = int(len(seg) // T)
    if n_cycles < 3:
        return None
    phases = np.arange(CYCLE_POINTS) / CYCLE_POINTS * T
    idx_base = np.arange(len(seg))
    acc = []
    for m in range(n_cycles):
        pos = m * T + phases
        if pos[-1] >= len(seg) - 1:
            break
        acc.append(np.interp(pos, idx_base, seg))
    if len(acc) < 3:
        return None
    avg = np.mean(acc, axis=0)
    k = int(np.argmax(np.abs(avg)))
    avg = np.roll(avg, -k)                       # 相位原点 = 最大冲击点
    e = avg ** 2
    e = np.convolve(e, np.ones(3) / 3.0, mode="same")
    total = float(e.sum())
    if total <= 0:
        return None
    pos01 = np.arange(CYCLE_POINTS) / CYCLE_POINTS
    centroid = float(np.sum(pos01 * e) / total)  # 能量重心位置（0=冲击点）
    cum = np.cumsum(e) / total
    i10 = int(np.argmax(cum >= 0.10))
    i90 = int(np.argmax(cum >= 0.90))
    rise_frac = float((i90 - i10) / CYCLE_POINTS)  # 能量主体占周期比例
    half = CYCLE_POINTS // 2
    front_ratio = float(e[:half].sum() / total)    # 前半周期能量占比
    best = np.max(np.abs(avg))
    crest = float(best / max(float(np.sqrt(np.mean(avg ** 2))), 1e-12))  # 周期波形尖锐度
    return {"cycle_centroid": centroid, "cycle_rise_frac": rise_frac,
            "cycle_front_ratio": front_ratio, "cycle_crest": crest}


# ---------------- H2: 谐波相位相干性 ----------------
def phase_coherence(seg, sr, f0, kmax=PHASE_KMAX):
    """谐波相位耦合度：对 H2..K 计算 angle(H_k) - k*angle(H_1)，
    幅度加权求 |平均单位复数|。1=完全脉冲锁定，0=相位散乱。"""
    nfft = 1 << 15
    w = np.hanning(len(seg)) * seg
    spec = np.fft.rfft(w, nfft)
    amps, phases = [], []
    for k in range(1, kmax + 1):
        f = k * f0
        if f > 5000.0:
            break
        idx = int(round(f * nfft / sr))
        if idx >= len(spec):
            break
        amps.append(float(abs(spec[idx])))
        phases.append(float(np.angle(spec[idx])))
    if len(amps) < 4:
        return None
    amps = np.array(amps)
    phases = np.array(phases)
    ks = np.arange(2, len(amps) + 1)
    dev = phases[1:] - ks * phases[0]
    wts = amps[1:]
    if wts.sum() <= 0:
        return None
    c = np.sum(wts * np.exp(1j * dev)) / wts.sum()
    return {"phase_coherence": float(abs(c))}


# ---------------- H3: 共振峰带宽（LPC） ----------------
def _levinson(r, order):
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0])
    if e <= 0:
        return a
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[1:i][::-1]))
        k = -acc / e
        anew = a.copy()
        for j in range(1, i):
            anew[j] = a[j] + k * a[i - j]
        anew[i] = k
        a = anew
        e *= (1.0 - k * k)
        if e <= 1e-12:
            break
    return a


def formants(seg16, sr16, order=LPC_ORDER):
    """LPC 求 F1/F2 频率与带宽（16kHz，order 14）。"""
    if len(seg16) < order * 4:
        return None
    x = seg16 - np.mean(seg16)
    x = np.append(x[0], x[1:] - 0.63 * x[:-1])       # 预加重
    x = x * np.hanning(len(x))
    r = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + order]
    if r[0] <= 0:
        return None
    a = _levinson(r, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    if len(roots) == 0:
        return None
    freqs = np.angle(roots) * sr16 / (2.0 * np.pi)
    bws = -np.log(np.maximum(np.abs(roots), 1e-12)) * sr16 / np.pi
    cand = [(f, b) for f, b in zip(freqs, bws)
            if 150.0 <= f <= FORMANT_MAX_HZ and 20.0 <= b <= FORMANT_BW_MAX]
    cand.sort()
    if len(cand) < 2:
        return None
    f1, b1 = cand[0]
    f2, b2 = cand[1]
    return {"f1_hz": float(f1), "b1_hz": float(b1), "f2_hz": float(f2), "b2_hz": float(b2),
            "bw_sum_hz": float(b1 + b2)}


# ---------------- 单文件分析 ----------------
def analyze(item):
    wav = DATA / item["wav"]["path"]
    raw = DATA / item["raw"]["path"]
    if sha(raw) != item["raw"]["sha256"] or sha(wav) != item["wav"]["sha256"]:
        raise ValueError(f"input changed: {item['id']}")
    x, sr = read_pcm16(wav)
    win, hop = int(round(sr * WIN_S)), int(round(sr * HOP_S))
    if len(x) < win:
        raise RuntimeError(f"too short: {item['id']}")

    gcd = math.gcd(sr, FORMANT_SR)
    x16 = signal.resample_poly(x, FORMANT_SR // gcd, sr // gcd)   # 整段一次性下采样
    ratio = sr // FORMANT_SR

    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB

    rows = []
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        ce = cycle_energy(seg, sr, f0)
        pc = phase_coherence(seg, sr, f0)
        s16 = s // ratio
        seg16 = x16[s16:s16 + win // ratio]
        fm = formants(seg16, FORMANT_SR)
        row = {"f0": f0}
        if ce:
            row.update(ce)
        if pc:
            row.update(pc)
        if fm:
            row.update(fm)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no voiced frames: {item['id']}")
    keys = ["f0", "cycle_centroid", "cycle_rise_frac", "cycle_front_ratio", "cycle_crest",
            "phase_coherence", "f1_hz", "b1_hz", "f2_hz", "b2_hz", "bw_sum_hz"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        if len(vals) < 20:
            stats[k] = {"median": None, "mad": None, "n": int(len(vals))}
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        stats[k] = {"median": round(med, 4), "mad": round(mad, 4), "n": int(len(vals))}

    loud = measure_loudness(x, sr)
    return {"dataset_id": item["id"], "song": item.get("song", ""), "variant": item.get("variant", ""),
            "voiced_frames": len(rows), "stats": stats,
            "lufs": loud["integrated_lufs"], "wav_sha256": sha(wav)}


# ---------------- 运行 ----------------
files = {}
meta = []
for song, tid, tlab, nid, nlab in PAIRS:
    for did, lab in ((tid, tlab), (nid, nlab)):
        m = files.setdefault(did, analyze(items[did]))
        m["role"] = lab
        m["pair_song"] = song
        meta.append({"dataset_id": did, "song": song, "role": lab})

(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-012", "phase": "F1-DIAG-2",
    "purpose": "时域/相位/共振峰三项对照诊断（只读），判断剩余维度是否真正区分薄音与正常版",
    "hypotheses": {
        "H1_时域": "薄音每周期能量更集中前段（脉冲式），正常版更均匀",
        "H2_相位": "薄音谐波相位更散乱，正常版更同相（脉冲锁定）",
        "H3_共振峰": "薄音 F1/F2 带宽更宽（阻尼/共鸣不足）",
    },
    "items": meta,
    "config": {"f0_search_hz": [FMIN, FMAX], "win_s": WIN_S, "hop_s": HOP_S,
               "periodicity_gate": R_GATE, "active_floor_db": FLOOR_DB,
               "cycle_points": CYCLE_POINTS, "phase_kmax": PHASE_KMAX,
               "formant_sr": FORMANT_SR, "lpc_order": LPC_ORDER},
    "method_notes": [
        "周期能量分布：按 F0 周期折叠平均，以最大绝对值点为相位原点（近似声门冲击点）",
        "相位相干性：angle(H_k) − k·angle(H_1) 的幅度加权平均单位复数模长；1=脉冲锁定，0=散乱",
        "共振峰：16kHz 整段下采样 + 预加重 + LPC(14) + Levinson-Durbin，取 F1/F2 频率与带宽",
        "全部为自实现，未与 Praat 校验；仅用于同一录音链内的成对相对比较",
        "薄/正常标签由用户指定；Agent 未听音；本报告为结构差异描述，非听感结论",
    ],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 配对比较 ----------------
CORE = {
    "时域": ["cycle_centroid", "cycle_rise_frac", "cycle_front_ratio", "cycle_crest"],
    "相位": ["phase_coherence"],
    "共振峰": ["f1_hz", "b1_hz", "f2_hz", "b2_hz", "bw_sum_hz"],
}
pairs_out = []
for song, tid, tlab, nid, nlab in PAIRS:
    t, n = files[tid], files[nid]
    delta = {}
    effect = {}
    for _, ks in CORE.items():
        for k in ks:
            tm, nm = t["stats"][k]["median"], n["stats"][k]["median"]
            if tm is None or nm is None:
                delta[k] = None
                effect[k] = None
                continue
            delta[k] = round(nm - tm, 4)
            pool = 0.5 * (t["stats"][k]["mad"] + n["stats"][k]["mad"])
            effect[k] = round((nm - tm) / pool, 2) if pool and pool > 1e-9 else None
    pairs_out.append({
        "song": song, "thin": {"id": tid, "label": tlab}, "normal": {"id": nid, "label": nlab},
        "thin_medians": {k: t["stats"][k]["median"] for _, ks in CORE.items() for k in ks},
        "normal_medians": {k: n["stats"][k]["median"] for _, ks in CORE.items() for k in ks},
        "delta_normal_minus_thin": delta,
        "effect_size_vs_frame_mad": effect,
    })

consistency = {}
for dim, ks in CORE.items():
    for k in ks:
        signs = []
        deltas, effects = [], []
        for p in pairs_out:
            d = p["delta_normal_minus_thin"][k]
            if d is None:
                signs.append("?")
            else:
                signs.append("+" if d > 0 else ("-" if d < 0 else "0"))
                deltas.append(d)
            e = p["effect_size_vs_frame_mad"][k]
            if e is not None:
                effects.append(e)
        valid = [s for s in signs if s not in ("0", "?")]
        consistency[k] = {
            "dimension": dim, "signs_per_pair": signs,
            "all_same_sign": len(set(valid)) == 1 and len(valid) == len(PAIRS),
            "n_valid": len(valid),
            "median_delta": round(float(np.median(deltas)), 4) if deltas else None,
            "median_effect_size": round(float(np.median(np.abs(effects))), 2) if effects else None,
            "max_effect_size": round(float(np.max(np.abs(effects))), 2) if effects else None,
        }

# 判别力总表：4/4 同向 且 效应量 ≥ 1.0 才算"有区分"
for k, v in consistency.items():
    v["discriminates"] = bool(v["all_same_sign"] and v["median_effect_size"]
                              and v["median_effect_size"] >= 1.0)

verdict = {dim: any(consistency[k]["discriminates"] for k in ks) for dim, ks in CORE.items()}

qa = {
    "experiment": "F1-EXP-012", "phase": "F1-DIAG-2",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "files": [files[i] for i in sorted(files)],
    "pairs": pairs_out,
    "consistency": consistency,
    "dimension_verdict": verdict,
    "decision_rule": "4/4 对同向 且 中位效应量 ≥1.0（相对帧内 MAD）才判该指标有区分力",
    "limitations": [
        "自实现指标未与 Praat 校验；H1/H2 受共振峰与元音影响，同人同曲近似受控",
        "同曲不同 take，段落/音区分布不完全一致；F0 差异需对照看",
        "手机录音链路高频受限；绝对带宽数值不代表录音棚水平，成对内一致",
        "LPC 在 16kHz、order 14 下只覆盖到 4kHz 以内的 F1/F2；F3+ 未测",
        "标签由用户指定；本报告不构成听感结论",
    ],
}
(OUT / "qa/analysis.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

COLS = ["f0", "cycle_centroid", "cycle_rise_frac", "cycle_front_ratio", "cycle_crest",
        "phase_coherence", "f1_hz", "b1_hz", "f2_hz", "b2_hz", "bw_sum_hz"]
with open(OUT / "qa/metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair_song", "role", "dataset_id", "variant", "voiced_frames", "lufs"] + COLS)
    for did in sorted(files):
        m = files[did]
        w.writerow([m["pair_song"], m["role"], did, m["variant"], m["voiced_frames"], m["lufs"]]
                   + [m["stats"][k]["median"] for k in COLS])

with open(OUT / "qa/pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["song", "metric", "dimension", "thin", "normal", "delta_normal_minus_thin",
                "effect_size_vs_mad", "same_sign_across_4_pairs", "discriminates"])
    for p in pairs_out:
        for dim, ks in CORE.items():
            for k in ks:
                w.writerow([p["song"], k, dim, p["thin_medians"][k], p["normal_medians"][k],
                            p["delta_normal_minus_thin"][k], p["effect_size_vs_frame_mad"][k],
                            consistency[k]["all_same_sign"], consistency[k]["discriminates"]])

with open(OUT / "qa/verdict.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dimension", "metric", "signs", "all_same_sign", "median_delta",
                "median_effect_size", "discriminates"])
    for k, v in consistency.items():
        w.writerow([v["dimension"], k, "".join(v["signs_per_pair"]), v["all_same_sign"],
                    v["median_delta"], v["median_effect_size"], v["discriminates"]])

(OUT / "summary.json").write_text(json.dumps({
    "experiment": "F1-EXP-012", "phase": "F1-DIAG-2", "status": "analysis_complete",
    "read_only": True, "audio_generated": False, "sample_count": 8,
    "pairs": [p["song"] for p in pairs_out],
    "dimension_verdict": verdict,
    "discriminating_metrics": [k for k, v in consistency.items() if v["discriminates"]],
    "report": "RESULT.md", "full_data": "qa/analysis.json",
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-012\n"
    f"read-only three-dimension diagnosis on 8 self recordings (4 thin/normal pairs)\n"
    f"no audio generated; no holdout processed\n",
    encoding="utf-8")

print(json.dumps({
    "dimension_verdict": verdict,
    "discriminating_metrics": [k for k, v in consistency.items() if v["discriminates"]],
    "consistency": {k: {"signs": "".join(v["signs_per_pair"]), "same": v["all_same_sign"],
                        "median_delta": v["median_delta"], "effect": v["median_effect_size"],
                        "ok": v["discriminates"]} for k, v in consistency.items()},
}, ensure_ascii=False, indent=1))
