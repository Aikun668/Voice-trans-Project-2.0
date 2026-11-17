"""F1-EXP-010 / F1-P4A: F0 锁定的谐波选择性处理（方案 A 首轮）。

诊断结论（F1-EXP-009）：薄音的声源指纹 = H1−H2 偏大 ~4dB、上方谐波丰富度缺 ~4dB，
能量过度集中于基频；谐波整体斜率/周期性与正常版基本一致。
本轮机制：逐帧跟踪 F0，只在 H2..H8 谐波位置施加窄带增益（不动 H1、
不动谐波之间的噪声与气声），增益表按"本人正常版（A02 同曲对照）"的
谐波剖面差值导出；两档强度 50% / 100%。F0 不可靠帧整帧回退原声。
只处理 A01；Holdout 不动；不叠加任何其他模块。
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

OUT = ROOT / "experiments/F1-EXP-010"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-010 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; DISPLAY = "平凡之路 / 薄音版"
TARGET_ID = "A_self_02"                      # 目标参考：同曲正常版（只读，不处理、不交付）
CROSS_CHECK_IDS = ["A_self_04", "A_self_06", "A_self_08"]  # 其余正常版剖面（仅交叉核对）
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
SEED = 20261102

# --- 机制参数（固定并记录） ---
K_MIN, K_MAX = 2, 8              # 只增强第 2..8 个谐波（H1 不动）
K_PROFILE = 12                    # 剖面测量到 H12（H9-H12 仅记录缺口，不处理）
CAP_DB = 9.0                      # 单谐波增益上限
MIN_DEFICIT_DB = 2.0              # 缺口 <2dB 的谐波不处理
STRENGTHS = [("V1", 0.5), ("V2", 1.0)]
STFT_N, STFT_H = 2048, 512        # 42.7ms 窗 / 10.7ms 步进（48kHz）
R_GATE = 0.35                     # STFT 帧周期性门限（低于则整帧回退原声）
FLOOR_DB = 40.0                   # 帧能量门限：低于最响帧 -40dB 回退
ALPHA_MASK = 0.6                  # 掩膜时域平滑（新帧权重）
VIBRATO_SPAN = 0.06                # 掩膜半宽按 ±6% 颤音幅度设计
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020  # 剖面/自检分析帧（与 F1-EXP-009 一致）
FMIN, FMAX = 90.0, 450.0


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def frame_f0_ac(win, sr):
    """自相关基音估计（与 F1-EXP-009 相同实现）：返回 (f0, 周期性 r)。"""
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


def harmonic_profile(x, sr, kmax_profile=K_PROFILE):
    """60ms 窗 / 20ms 步进的帧谐波剖面：P[k] = median(A_k − A_1)，k=1..kmax_profile。
    门限与 F1-EXP-009 一致（-40dB 能量 + 周期性 0.35），保证前后可比。"""
    win, hop = int(round(sr * ANA_WIN_S)), int(round(sr * ANA_HOP_S))
    if len(x) < win:
        raise RuntimeError("too short for profile")
    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB
    rel = {k: [] for k in range(1, kmax_profile + 1)}
    n_voiced = 0
    nfft = 1 << 15
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        n_voiced += 1
        w = np.hanning(win) * seg
        spec = np.fft.rfft(w, nfft)
        gain = np.sum(np.hanning(win))
        a1 = 20.0 * np.log10(max(2.0 * abs(spec[int(round(f0 * nfft / sr))]) / gain, 1e-10))
        for k in range(1, kmax_profile + 1):
            f = k * f0
            if f > 8000.0:
                break
            idx = int(round(f * nfft / sr))
            if idx >= len(spec):
                break
            ak = 20.0 * np.log10(max(2.0 * abs(spec[idx]) / gain, 1e-10))
            rel[k].append(ak - a1)
    if n_voiced < 50:
        raise RuntimeError("voiced frames too few for profile")
    P = {k: float(np.median(v)) for k, v in rel.items() if len(v) >= 30}
    return P, n_voiced


def key_metrics(x, sr):
    """处理后自检：与诊断表同口径的 H1-H2 / 丰富度等关键量。"""
    P, n_voiced = harmonic_profile(x, sr)
    h1h2 = P[2]
    rich_parts = [10 ** (P[k] / 10) for k in P if k >= 2]
    richness = float(10.0 * np.log10(max(sum(rich_parts), 1e-10)))
    return {"h1_minus_h2_db": round(h1h2, 2), "harm_richness_db_k12": round(richness, 2),
            "profile_rel_db": {k: round(v, 2) for k, v in P.items()},
            "voiced_frames": n_voiced}


def stft_harmonic_boost(x, sr, gains_db, kmin=K_MIN, kmax=K_MAX):
    """核心机制：STFT 逐帧 F0 跟踪 → 谐波位置窄带增益掩膜 → WOLA 重叠合成。
    - H1 与谐波间隙不动（掩膜为 1）；
    - F0 不可靠 / 能量过低的帧掩膜为 1（整帧原声回退，宁可不加不加错）；
    - 掩膜半宽按 ±6% 颤音摆幅设计，逐帧随 F0 更新；
    - 掩膜做一阶时域平滑，避免帧间跳变。"""
    N, H = STFT_N, STFT_H
    w = np.hanning(N)
    bin_hz = sr / N
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    gains = {k: g for k, g in gains_db.items() if g > 0.01 and kmin <= k <= kmax}
    if not gains:
        raise RuntimeError("empty gain table")

    # 帧能量与 F0 跟踪
    rms_db, f0_track = [], []
    for i in range(n_frames):
        fr = xp[i * H:i * H + N]
        rms_db.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        f0, r = frame_f0_ac(fr, sr)
        f0_track.append(f0 if (np.isfinite(f0) and r >= R_GATE) else float("nan"))
    rms_db = np.array(rms_db)
    gate = rms_db.max() - FLOOR_DB
    # F0 三帧中值滤波（只平滑孤立抖动，保留颤音）；无声帧保持无声，不借邻居 F0
    f0_med = f0_track.copy()
    for i in range(n_frames):
        if not np.isfinite(f0_track[i]):
            continue
        seg = [f0_track[j] for j in range(max(0, i - 1), min(n_frames, i + 2))]
        v = [t for t in seg if np.isfinite(t)]
        if v:
            f0_med[i] = float(np.median(v))

    nb = N // 2 + 1
    masks = np.ones((n_frames, nb))
    applied = {k: [] for k in gains}
    n_boosted = 0
    for i in range(n_frames):
        if rms_db[i] < gate or not np.isfinite(f0_med[i]):
            continue
        f0 = f0_med[i]
        width = max(2, int(math.ceil(K_MAX * VIBRATO_SPAN * f0 / bin_hz)))
        mask = masks[i]
        touched = False
        for k, g_db in gains.items():
            center = k * f0 / bin_hz
            lo = max(int(math.floor(center)) - width, 1)
            hi = min(int(math.ceil(center)) + width, nb - 1)
            if hi <= lo:
                continue
            g_lin = 10.0 ** (g_db / 20.0)
            for b in range(lo, hi + 1):
                t = 1.0 - min(abs(b - center) / (width + 0.5), 1.0)
                shape = 0.5 - 0.5 * math.cos(math.pi * t)
                val = 1.0 + (g_lin - 1.0) * shape
                if val > mask[b]:
                    mask[b] = val
            touched = True
            applied[k].append(g_db)
        if touched:
            n_boosted += 1
        # 时域平滑
    for i in range(1, n_frames):
        masks[i] = ALPHA_MASK * masks[i] + (1.0 - ALPHA_MASK) * masks[i - 1]

    # STFT 分析 → 掩膜 → 重叠合成
    y_ola = np.zeros(len(xp))
    norm = np.zeros(len(xp))
    for i in range(n_frames):
        fr = xp[i * H:i * H + N] * w
        spec = np.fft.rfft(fr)
        y = np.fft.irfft(spec * masks[i], N)
        s = i * H
        y_ola[s:s + N] += y * w
        norm[s:s + N] += w * w
    out = y_ola / np.maximum(norm, 1e-8)
    y = out[pad:pad + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    stats = {
        "frames_total": int(n_frames), "frames_boosted": int(n_boosted),
        "boost_fraction": round(n_boosted / n_frames, 3),
        "gain_applied_mean_db": {str(k): round(float(np.mean(v)), 2) for k, v in applied.items() if v},
        "gain_applied_frames": {str(k): len(v) for k, v in applied.items()},
    }
    return y, stats


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    yseg = y[:len(seg)]
    c = signal.correlate(yseg, seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


def spectral_centroid(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    f, p = f[m], p[m]
    return float(np.sum(f * p) / max(float(np.sum(p)), 1e-20))


# ---------------- 输入与目标剖面 ----------------
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


x, sr, item = load(DATASET_ID)
xt, srt, item_t = load(TARGET_ID)  # 目标参考（同曲正常版）
x_cross = {did: load(did) for did in CROSS_CHECK_IDS}

listening_contract = {
    "原音问题登记": "A01 自评薄音（单薄、不够充实）；诊断指纹 = 上方谐波结构与 2-4kHz 谐波能量缺失（F1-EXP-009）",
    "素材与片段": "A01 全长 13.14s，不截取",
    "预期可听变化": "元音更实、声音有芯、整句更充实；变化方向应为『厚』而非『亮/响』；仍是本人音色",
    "否决副作用": "金属感/机器嗡嗡声、个别字发假或毛刺、齿音变重、闷糊、像别人；只变亮或只变响不算通过",
    "机制边界": "本轮只动 H2-H8 谐波结构，不叠加 EQ/压缩/饱和；H1 与谐波间噪声不动",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-010", "phase": "F1-P4A",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "display": DISPLAY,
    "target_reference": {"dataset_id": TARGET_ID, "display": item_t.get("song", "") + " / " + item_t.get("variant", ""),
                         "use": "只读谐波剖面作为增益目标，不处理不交付"},
    "cross_check": CROSS_CHECK_IDS,
    "sealed_not_processed": SEALED,
    "wav_sha256": sha(DATA / item["wav"]["path"]), "raw_sha256": sha(DATA / item["raw"]["path"]),
    "frames": len(x), "sample_rate": sr,
    "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 剖面与增益表 ----------------
P_thin, n_thin = harmonic_profile(x, sr)
P_target, n_target = harmonic_profile(xt, srt)
P_cross = {did: harmonic_profile(*x_cross[did][:2])[0] for did in CROSS_CHECK_IDS}
P_cross_median = {k: float(np.median([P_cross[d][k] for d in P_cross if k in P_cross[d]]))
                  for k in range(2, K_PROFILE + 1)
                  if all(k in P_cross[d] for d in P_cross)}

deficit = {k: round(P_target[k] - P_thin[k], 2) for k in range(K_MIN, K_PROFILE + 1)
           if k in P_target and k in P_thin}
gain_table_100 = {k: round(min(max(deficit.get(k, 0.0), 0.0), CAP_DB), 2) for k in range(K_MIN, K_MAX + 1)}
gain_table_100 = {k: v for k, v in gain_table_100.items() if deficit.get(k, 0) >= MIN_DEFICIT_DB}
if not gain_table_100:
    raise RuntimeError("no harmonic exceeds minimum deficit")
gain_tables = {vid: {k: round(s * g, 2) for k, g in gain_table_100.items()} for vid, s in STRENGTHS}

baseline = key_metrics(x, sr)
target_metrics = key_metrics(xt, srt)
orig_pitch = float(np.nanmedian(estimate_pitch(x, sr)))

# ---------------- 处理与自检 ----------------
raw_root = OUT / "outputs/processed_raw"
records = []
for vid, strength in STRENGTHS:
    y, proc_stats = stft_harmonic_boost(x, sr, gain_tables[vid])
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    km = key_metrics(y, sr)
    lag = lag_between(y, x, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    centroid = spectral_centroid(y, sr)
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0
    hard, flags = [], []
    # 自检：机制是否把指标朝目标移动（按"与目标距离的收敛比例"判定，与符号约定无关）
    gap_h1h2 = abs(baseline["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"])
    gap_rich = abs(baseline["harm_richness_db_k12"] - target_metrics["harm_richness_db_k12"])
    move_h1h2 = km["h1_minus_h2_db"] - baseline["h1_minus_h2_db"]      # 有符号移动量（A2-A1 约定）
    move_rich = km["harm_richness_db_k12"] - baseline["harm_richness_db_k12"]
    closed_h1h2 = 1.0 - abs(km["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"]) / max(gap_h1h2, 1e-9)
    closed_rich = 1.0 - abs(km["harm_richness_db_k12"] - target_metrics["harm_richness_db_k12"]) / max(gap_rich, 1e-9)
    need_closed = 0.25 if vid == "V1" else 0.5
    if closed_h1h2 < need_closed:
        hard.append(f"H1-H2 未有效向目标移动 closed={closed_h1h2:.0%} (需≥{need_closed:.0%})")
    if closed_rich < need_closed:
        hard.append(f"丰富度未有效向目标移动 closed={closed_rich:.0%} (需≥{need_closed:.0%})")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if abs(pitch - orig_pitch) > 2.0:
        hard.append(f"音高异常 Δ={pitch - orig_pitch:.2f}Hz")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    if clipped > 0:
        hard.append(f"削波 {clipped} samples")
    # 诊断标记
    if proc_stats["boost_fraction"] < 0.4:
        flags.append(f"处理帧占比偏低 boost_fraction={proc_stats['boost_fraction']:.2f}")
    max_gain = max(gain_tables[vid].values())
    if max_gain >= CAP_DB - 0.1:
        flags.append(f"单谐波增益触顶 {max_gain:.1f}dB，注意金属感")
    # 导出（峰值保护性常数增益，非限制器）
    guard = min(0.0, -0.5 - st["sample_peak_dbfs"])
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, guard, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
        hard.append("导出削波")
    records.append({
        "id": vid, "strength": strength, "gain_table_db": gain_tables[vid],
        "proc_stats": proc_stats,
        "baseline_h1_minus_h2_db": baseline["h1_minus_h2_db"],
        "achieved_h1_minus_h2_db": km["h1_minus_h2_db"],
        "target_h1_minus_h2_db": target_metrics["h1_minus_h2_db"],
        "move_h1_minus_h2_db": round(move_h1h2, 2),
        "gap_closed_h1h2": round(closed_h1h2, 3),
        "baseline_richness_db": baseline["harm_richness_db_k12"],
        "achieved_richness_db": km["harm_richness_db_k12"],
        "target_richness_db": target_metrics["harm_richness_db_k12"],
        "move_richness_db": round(move_rich, 2),
        "gap_closed_richness": round(closed_rich, 3),
        "achieved_profile_rel_db": km["profile_rel_db"],
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "rms_dbfs": round(st["rms_dbfs"], 2),
        "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "centroid_hz": round(centroid, 0),
        "lag_samples": lag, "pitch_median_hz": round(pitch, 1),
        "raw_export_guard_db": guard, "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha(rp), "raw_lufs": ye_loud["integrated_lufs"],
        "raw_true_peak_dbtp": ye_loud["true_peak_dbtp"], "raw_lra": ye_loud["loudness_range_lu"],
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    })

# ---------------- 统一响度 + 盲听 ----------------
orig_loud = measure_loudness(x, sr)
loudness_all = {"original": orig_loud, **{r["id"]: {"integrated_lufs": r["raw_lufs"], "true_peak_dbtp": r["raw_true_peak_dbtp"]} for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loudness_all.values()])

listen_root = OUT / "listening/audio"
export_wav(x, sr, target_lufs - orig_loud["integrated_lufs"], listen_root / "original.wav")
om, osr = read_pcm16(listen_root / "original.wav")
if abs(measure_loudness(om, osr)["integrated_lufs"] - target_lufs) > 0.3:
    raise RuntimeError("original loudness match failed")

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-010")
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
                    "true_peak_dbtp": dm["true_peak_dbtp"], "frames": len(de), "sha256": sha(dest)}

rng = random.Random(SEED)
deliverables = [r["id"] for r in records if r["deliverable"]]
letters = ["A", "B", "C", "D"][:len(deliverables)]
blind = {letter: vid for letter, vid in zip(letters, rng.sample(deliverables, len(deliverables)))}
for letter, vid in blind.items():
    shutil.copyfile(listen_root / f"{vid}.wav", listen_root / f"{letter}.wav")
    (listen_root / f"{vid}.wav").unlink()

# ---------------- 试听页 ----------------
rows = []
for letter in blind.keys():
    rec = next(r for r in records if r["id"] == blind[letter])
    badge = ' <span class="badge">带工程诊断标记</span>' if rec["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label><audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4A 谐波机制盲听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P4A · 谐波机制盲听（仅 A01）</h1>
<div class="note"><b>本轮登记（先于生成音频）：</b>原音问题 = A01 自评薄音。<br>
预期可听变化 = 元音更实、声音有芯、整句更充实；变化应为『厚』而不是『亮/响』；仍是本人音色。<br>
否决副作用 = 金属感/机器嗡嗡声、个别字发假或毛刺、齿音变重、闷糊、像别人；只变亮或只变响不算通过。<br>
<b>若认为原音没有"薄"的问题，请直接说明，本轮记 not_applicable。</b></div>
<div class="card"><h3>试听说明</h3><p class="meta">先听原音，再逐个听版本 A/B。两版来自同一机制（按你自己的正常版录音校准的泛音增强），强度已隐藏，映射存于 qa/blind_key.json，建议试听后再看。请分别判断：①有无变化；②饱满度是否更好；③是否自然、像本人；④有无否决副作用。</p>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；全部文件统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。本轮只动泛音结构，未叠加任何其他模块。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA 与记录 ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件，以免破坏盲听",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-010", "phase": "F1-P4A", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "sample_display": DISPLAY,
    "mechanism": {
        "name": "F0 锁定谐波选择性增强（方案 A）",
        "stft": {"n": STFT_N, "hop": STFT_H, "window": "hann", "synthesis": "WOLA + 精确归一化"},
        "harmonics": f"H{K_MIN}-H{K_MAX} 窄带增益，H1 与谐波间间隙不动（掩膜=1）",
        "mask_halfwidth": f"按 ±{VIBRATO_SPAN*100:.0f}% 颤音摆幅逐帧计算（F0×k×6% 换算 bin）",
        "fallback": f"周期性<{R_GATE} 或帧能量低于最响帧{FLOOR_DB}dB 的帧整帧回退原声；掩膜一阶平滑 α={ALPHA_MASK}",
        "gain_source": "A01 谐波剖面 vs A_self_02（同曲正常版）谐波剖面的差值，逐谐波裁剪 [0,+9dB]，缺口<2dB 不处理",
        "strengths": {"V1": "50%", "V2": "100%"},
    },
    "profiles": {
        "thin_A01": {"rel_db": baseline["profile_rel_db"], "voiced_frames": baseline["voiced_frames"]},
        "target_A02": {"rel_db": target_metrics["profile_rel_db"], "voiced_frames": target_metrics["voiced_frames"]},
        "cross_check_median": {str(k): round(v, 2) for k, v in P_cross_median.items()},
        "deficit_thin_vs_target_db": deficit,
    },
    "baseline_metrics": baseline, "target_metrics": target_metrics,
    "target_lufs": target_lufs,
    "versions": records,
    "screen_rules": {
        "hard": ["H1-H2/丰富度向目标的距离收敛比例不足（V1<25% / V2<50%）",
                 "|lag|>1 样本", "|Δpitch|>2Hz", "非有限值", "削波"],
        "diag": ["处理帧占比<40%", "单谐波增益触顶 9dB（金属感风险）"],
        "loudness": "常数增益统一响度，无限制器；误差≤0.3 LU",
    },
    "limitations": [
        "自实现谐波剖面/掩膜机制，未与外部工具校验；指标仅用于成对相对比较",
        "增益表来自 A02 单曲剖面（F0 中值 171.8 vs A01 175.5，高度匹配）；其余正常版中位剖面已交叉核对",
        "H9-H12 缺口仅记录不处理；丰富度目标可能无法完全闭合",
        "掩膜宽度按 ±6% 颤音设计，极端颤音瞬间部分谐波增益可能未完全覆盖（欠增益而非杂音）",
        "Agent 无听音能力；技术自检通过不代表听感改善，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["id", "strength", "gain_table_db", "move_h1h2_db", "move_richness_db",
                 "gap_closed_h1h2", "gap_closed_richness",
                 "achieved_h1h2_db", "target_h1h2_db", "achieved_richness_db", "target_richness_db",
                 "boost_fraction", "peak_dbfs", "centroid_hz", "lag", "pitch_hz",
                 "raw_lufs", "raw_tp_dbtp", "hard_fail", "flags", "matched_lufs", "matched_tp_dbtp"])
    for r in records:
        m = r.get("matched", {})
        wr.writerow([r["id"], r["strength"], json.dumps(r["gain_table_db"], ensure_ascii=False),
                     r["move_h1_minus_h2_db"], r["move_richness_db"],
                     r["gap_closed_h1h2"], r["gap_closed_richness"],
                     r["achieved_h1_minus_h2_db"], r["target_h1_minus_h2_db"],
                     r["achieved_richness_db"], r["target_richness_db"],
                     r["proc_stats"]["boost_fraction"], r["peak_dbfs"], r["centroid_hz"],
                     r["lag_samples"], r["pitch_median_hz"],
                     round(r["raw_lufs"], 2), round(r["raw_true_peak_dbtp"], 2),
                     "; ".join(r["hard_fail"]), "; ".join(r["flags"]),
                     m.get("lufs", ""), m.get("true_peak_dbtp", "")])

params = {
    "experiment": "F1-EXP-010", "phase": "F1-P4A",
    "question": "把薄音的谐波结构向本人正常版靠拢，能否获得可听、自然的饱满度改善（方案 A 首轮，单机制）",
    "single_module": True, "no_stacking": True, "samples": [SAMPLE_ID], "sealed": SEALED,
    "mechanism": qa["mechanism"], "gain_table_100": gain_table_100, "gain_tables": gain_tables,
    "deficit_profile_db": deficit,
    "target_reference": TARGET_ID, "cross_check": CROSS_CHECK_IDS,
    "listening_target_lufs": target_lufs, "loudness_tolerance_lu": 0.3,
    "references": {
        "diagnosis": "experiments/F1-EXP-009/RESULT.md — 薄音指纹：H1-H2 偏大、丰富度缺 4dB、2-4kHz 弱",
        "paper_D_glottal_source": "https://www.isca-archive.org/smm_2019/cabral19_smm.html — 声源特征变换方向的工程化第一步",
        "stop_condition": "初测+最多一次修订；无自然饱满改善或出现金属感/失真 → 转 C（接受边界），不扫参数",
    },
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-010", "phase": "F1-P4A", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "versions": [r["id"] for r in records],
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "diag_marked": [r["id"] for r in records if r["diag_mark"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(r.get("matched", {}).get("lufs", target_lufs + 99) - target_lufs) <= 0.3
                for r in records if r["deliverable"]),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-010\nphase: F1-P4A\n"
    f"sample: {SAMPLE_ID}; target profile ref: {TARGET_ID} (read-only)\n"
    f"sealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\ntechnical_pass: {summary['technical_pass']}\n"
    f"note: single harmonic-selective mechanism; no limiter; originals untouched\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-010 结果（pending_jade）\n\n"
    "F1-P4A 谐波机制首轮：F0 锁定 H2-H8 窄带增益，目标剖面 = 本人正常版（A_self_02 同曲对照），"
    "两档强度（50%/100%），只处理 A01，统一响度盲听。机制自检与增益表见 qa/qa_report.json。\n\n"
    f"- 交付：{', '.join(r['id'] for r in records if r['deliverable'])}（匿名 A/B，映射存 qa/blind_key.json）\n"
    f"- 硬性故障：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    f"- 工程诊断标记：{', '.join(r['id'] for r in records if r['diag_mark']) or '无'}\n"
    "- 听感结论待 Jade 试听；技术自检通过不代表听感改善。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "deficit_profile_db": deficit,
    "gain_tables": gain_tables,
    "baseline": {"h1_minus_h2_db": baseline["h1_minus_h2_db"], "richness_db": baseline["harm_richness_db_k12"]},
    "target": {"h1_minus_h2_db": target_metrics["h1_minus_h2_db"], "richness_db": target_metrics["harm_richness_db_k12"]},
    "versions": [{"id": r["id"], "strength": r["strength"],
                  "move_h1_minus_h2_db": r["move_h1_minus_h2_db"],
                  "move_richness_db": r["move_richness_db"],
                  "achieved_h1_minus_h2_db": r["achieved_h1_minus_h2_db"],
                  "achieved_richness_db": r["achieved_richness_db"],
                  "boost_fraction": r["proc_stats"]["boost_fraction"],
                  "centroid_hz": r["centroid_hz"], "lag_samples": r["lag_samples"],
                  "hard_fail": r["hard_fail"], "flags": r["flags"]}
                 for r in records],
}, ensure_ascii=False, indent=1))
