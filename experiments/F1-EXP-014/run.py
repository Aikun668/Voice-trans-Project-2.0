"""F1-EXP-014 / F1-P4B: 全谐波逐帧对齐（方案 A 的最后一次、最彻底尝试）。

背景：
  EXP-010/011 只对齐了 H2-H8 / H2-H12，从未动 H1，也从未覆盖 2-4kHz（H13-H23）。
  听感始终为"只是音量/亮度加了一点点"。
本轮：
  1) 谐波覆盖扩到 H1-H23（约 175Hz-4025Hz，覆盖真正的 2-4kHz 区）
  2) 逐帧测量每帧谐波剖面，逐帧补偿（而非整段统一增益表）
  3) 剖面归一化改为"相对全带能量"，使 H1 也能被正确调整
两档（唯一差别 = H1 是否参与对齐）：
  W1 = 完整对齐（H1 可降，最接近正常版结构）
  W2 = H1 锁定不动（仅 H2-H23 对齐，用于隔离"H1 调整"的影响）

预期诚实说明：正常版 A02 的频谱重心 403Hz 明显高于薄音版 294Hz，
"完整对齐"在客观上是把重心推得更高，可能听感更亮。本轮正是要测这个。
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

OUT = ROOT / "experiments/F1-EXP-014"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-014 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; DISPLAY = "平凡之路 / 薄音版"
TARGET_ID = "A_self_02"; TARGET_DISPLAY = "平凡之路 / 正常版"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
SEED = 20261103

# --- 机制参数 ---
K_MIN, K_MAX = 1, 23              # H1-H23（175Hz 基频时约覆盖到 4025Hz）
K_PROFILE = 23
CAP_UP_DB = 9.0                   # 单谐波最大提升（与上轮一致的稳定上限）
CAP_DOWN_DB = 6.0                 # 单谐波最大衰减（防止 H1 被砍出空洞）
STFT_N, STFT_H = 2048, 512
R_GATE = 0.35
FLOOR_DB = 40.0
VIBRATO_SPAN = 0.06
PEAK_SEARCH_HZ = 20.0             # 每谐波局部峰值搜索半宽（汉宁窗主瓣约 ±23Hz）
SMOOTH_TAU_S = 0.050              # 增益时间平滑（50ms 一阶低通）
ALPHA_MASK = 0.6
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
FMIN, FMAX = 90.0, 450.0
BAND_EDGES = 8000.0
VARIANTS = [("W1", True), ("W2", False)]   # (id, h1_adjustable)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


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


def harmonic_power_profile(seg, sr, f0, kmax=K_PROFILE):
    """逐帧谐波功率剖面（相对全带功率，dB）。
    汉宁窗零填充到 32768；每个谐波在中心 ±20Hz 内取局部峰值功率；
    全带 = 80-8000Hz 功率和。返回 dict k->相对dB（k 从 1 起）。"""
    n = len(seg)
    nfft = 1 << 15
    w = np.hanning(n)
    spec = np.fft.rfft(w * seg, nfft)
    power = np.abs(spec) ** 2
    bin_hz = sr / nfft
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= BAND_EDGES)
    total = float(power[band].sum())
    if total <= 0:
        return None, 0.0
    halfw = int(round(PEAK_SEARCH_HZ / bin_hz))
    prof = {}
    for k in range(1, kmax + 1):
        f = k * f0
        if f > BAND_EDGES - PEAK_SEARCH_HZ:
            break
        c0 = int(round(f / bin_hz))
        lo = max(c0 - halfw, 1)
        hi = min(c0 + halfw, len(power) - 1)
        if hi <= lo:
            break
        pk = float(power[lo:hi + 1].max())
        prof[k] = 10.0 * np.log10(max(pk, 1e-22) / total)
    return prof, total


def static_profile(x, sr, kmax=K_PROFILE):
    """整段统计谐波剖面（60ms 帧 / 20ms 步进的中位数），门限与 EXP-009 一致。"""
    win, hop = int(round(sr * ANA_WIN_S)), int(round(sr * ANA_HOP_S))
    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB
    coll = {k: [] for k in range(1, kmax + 1)}
    n_voiced = 0
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        prof, _ = harmonic_power_profile(seg, sr, f0, kmax)
        if not prof:
            continue
        n_voiced += 1
        for k, v in prof.items():
            coll[k].append(v)
    if n_voiced < 50:
        raise RuntimeError("voiced frames too few")
    return {k: float(np.median(v)) for k, v in coll.items() if len(v) >= 30}, n_voiced


def part_richness(prof, k_lo, k_hi):
    vals = [10 ** (prof[k] / 10.0) for k in range(k_lo, k_hi + 1) if k in prof]
    if not vals:
        return None
    return float(10.0 * np.log10(max(sum(vals), 1e-20)))


def key_metrics(x, sr):
    P, nv = static_profile(x, sr)
    return {"h1_minus_h2_db": round(P[1] - P[2], 2) if 2 in P else None,
            "rich_2_23_db": round(part_richness(P, 2, K_PROFILE), 2),
            "rich_13_23_db": round(part_richness(P, 13, K_PROFILE), 2) if 13 in P else None,
            "profile": {k: round(v, 2) for k, v in P.items()},
            "voiced_frames": nv}


def spectral_centroid(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    f, p = f[m], p[m]
    return float(np.sum(f * p) / max(float(np.sum(p)), 1e-20))


def band_rel_energy(x, sr, lo, hi, ref_lo=80, ref_hi=8000):
    f, p = signal.welch(x, sr, nperseg=min(8192, len(x)))
    tot = float(p[(f >= ref_lo) & (f < ref_hi)].sum())
    e = float(p[(f >= lo) & (f < hi)].sum())
    return float(10.0 * np.log10(max(e, 1e-20) / max(tot, 1e-20)))


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    yseg = y[:len(seg)]
    c = signal.correlate(yseg, seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


def align_target_profile(target_prof, kmax=K_MAX):
    """把目标整段剖面整理成 k->相对dB，截断到 K_MAX。"""
    return {k: v for k, v in target_prof.items() if K_MIN <= k <= kmax}


def harmonic_align(x, sr, target_prof, h1_adjustable):
    """核心机制：STFT 逐帧测 F0 与谐波剖面 → 逐帧求增益 → 时间平滑 → WOLA 合成。"""
    N, H = STFT_N, STFT_H
    w = np.hanning(N)
    bin_hz = sr / N
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H

    rms_db, f0_track = [], []
    for i in range(n_frames):
        fr = xp[i * H:i * H + N]
        rms_db.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        f0, r = frame_f0_ac(fr, sr)
        f0_track.append(f0 if (np.isfinite(f0) and r >= R_GATE) else float("nan"))
    rms_db = np.array(rms_db)
    gate = rms_db.max() - FLOOR_DB
    f0_med = f0_track.copy()
    for i in range(n_frames):
        if not np.isfinite(f0_track[i]):
            continue
        seg = [f0_track[j] for j in range(max(0, i - 1), min(n_frames, i + 2))]
        v = [t for t in seg if np.isfinite(t)]
        if v:
            f0_med[i] = float(np.median(v))

    # 逐帧增益
    gains = np.zeros((n_frames, K_MAX + 1))   # index k
    valid = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if rms_db[i] < gate or not np.isfinite(f0_med[i]):
            continue
        prof, _ = harmonic_power_profile(xp[i * H:i * H + N], sr, f0_med[i])
        if not prof:
            continue
        valid[i] = True
        for k in range(K_MIN, K_MAX + 1):
            if k not in prof or k not in target_prof:
                continue
            if k == 1 and not h1_adjustable:
                continue
            g = target_prof[k] - prof[k]
            gains[i, k] = min(max(g, -CAP_DOWN_DB), CAP_UP_DB)

    # 时间平滑（一阶低通），只对有效帧间传递
    alpha = math.exp(-(STFT_H / sr) / SMOOTH_TAU_S)
    sm = np.zeros_like(gains)
    prev = np.zeros(K_MAX + 1)
    have_prev = False
    for i in range(n_frames):
        if valid[i]:
            prev = gains[i] if not have_prev else alpha * prev + (1 - alpha) * gains[i]
            have_prev = True
            sm[i] = prev
        else:
            sm[i] = prev if have_prev else 0.0

    # 掩膜与合成
    masks = np.ones((n_frames, N // 2 + 1))
    nb = N // 2 + 1
    n_boosted = 0
    for i in range(n_frames):
        if not valid[i]:
            continue
        f0 = f0_med[i]
        spacing = f0 / bin_hz
        width_cap = max(2, int(spacing // 2))
        touched = False
        for k in range(K_MIN, K_MAX + 1):
            g_db = sm[i, k]
            if abs(g_db) < 0.15:
                continue
            center = k * f0 / bin_hz
            if center >= nb - 1:
                break
            width = min(max(2, int(math.ceil(k * VIBRATO_SPAN * f0 / bin_hz))), width_cap)
            lo = max(int(math.floor(center)) - width, 1)
            hi = min(int(math.ceil(center)) + width, nb - 1)
            if hi <= lo:
                continue
            g_lin = 10.0 ** (g_db / 20.0)
            for b in range(lo, hi + 1):
                t = 1.0 - min(abs(b - center) / (width + 0.5), 1.0)
                shape = 0.5 - 0.5 * math.cos(math.pi * t)
                masks[i, b] = 1.0 + (g_lin - 1.0) * shape
            touched = True
        if touched:
            n_boosted += 1
    for i in range(1, n_frames):
        masks[i] = ALPHA_MASK * masks[i] + (1.0 - ALPHA_MASK) * masks[i - 1]

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

    applied = {k: float(np.mean(sm[valid, k])) for k in range(K_MIN, K_MAX + 1) if valid.any()}
    stats = {"frames_total": int(n_frames), "frames_processed": int(valid.sum()),
             "process_fraction": round(float(valid.mean()), 3),
             "gain_mean_db": {str(k): round(v, 2) for k, v in applied.items() if abs(v) >= 0.15},
             "max_up_db": round(float(sm.max()), 2), "max_down_db": round(float(sm.min()), 2)}
    return y, stats


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


# ---------------- 输入与目标 ----------------
x, sr = load(DATASET_ID)
xt, srt = load(TARGET_ID)

P_target, nv_target = static_profile(xt, srt, K_PROFILE)
P_thin, nv_thin = static_profile(x, sr, K_PROFILE)
target_prof = align_target_profile(P_target)
deficit = {k: round(target_prof[k] - P_thin[k], 2) for k in target_prof if k in P_thin}

baseline = key_metrics(x, sr)
target_metrics = key_metrics(xt, srt)
orig_pitch = float(np.nanmedian(estimate_pitch(x, sr)))
orig_centroid = spectral_centroid(x, sr)
target_centroid = spectral_centroid(xt, srt)

listening_contract = {
    "原音问题登记": "A01 自评薄音；用户描述'轻、有点嗲'，正常版为'自然地去唱'（EXP-013）",
    "素材与片段": "A01 全长 13.14s，不截取",
    "预期可听变化": "本轮为频谱路线最彻底一次：全谐波 H1-H23 + 逐帧对齐；目标是结构最接近正常版",
    "否决副作用": "金属感/嗡嗡声、嘶嘶声、齿音变重、闷或空洞、像别人；只变亮/只变响不算通过",
    "预先声明": "若本轮仍只有'一点点'，频谱路线确定性收工，不再有'还没试过'的悬念",
    "诚实提示": "正常版频谱重心 403Hz 高于薄音版 294Hz；'完整对齐'客观上会把重心推更高，可能更亮",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-014", "phase": "F1-P4B",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "display": DISPLAY,
    "target_reference": {"dataset_id": TARGET_ID, "display": TARGET_DISPLAY,
                         "use": "只读统计剖面作为对齐目标，不处理不交付"},
    "sealed_not_processed": SEALED,
    "frames": len(x), "sample_rate": sr,
    "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 处理两档 ----------------
raw_root = OUT / "outputs/processed_raw"
records = []
for vid, h1_adj in VARIANTS:
    y, pstat = harmonic_align(x, sr, target_prof, h1_adj)
    st = sample_stats(y, sr)
    km = key_metrics(y, sr)
    lag = lag_between(y, x, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    centroid = spectral_centroid(y, sr)
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0
    hard, flags = [], []

    if km["h1_minus_h2_db"] is None or abs(km["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"]) > 2.0:
        hard.append(f"H1-H2 未对齐 achieved={km['h1_minus_h2_db']} target={target_metrics['h1_minus_h2_db']}")
    if km["rich_2_23_db"] is None or abs(km["rich_2_23_db"] - target_metrics["rich_2_23_db"]) > 2.0:
        hard.append(f"丰富度(2-23) 未对齐 achieved={km['rich_2_23_db']} target={target_metrics['rich_2_23_db']}")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if abs(pitch - orig_pitch) > 2.0:
        hard.append(f"音高异常 Δ={pitch - orig_pitch:.2f}Hz")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    if clipped > 0:
        hard.append(f"削波 {clipped} samples")
    if pstat["process_fraction"] < 0.4:
        flags.append(f"处理帧占比偏低 {pstat['process_fraction']:.2f}")
    if pstat["max_up_db"] >= CAP_UP_DB - 0.1:
        flags.append(f"增益触顶 +{pstat['max_up_db']:.1f}dB，金属感风险")
    if pstat["max_down_db"] <= -CAP_DOWN_DB + 0.1:
        flags.append(f"衰减触底 {pstat['max_down_db']:.1f}dB，空洞/闷风险")

    guard = min(0.0, -0.5 - st["sample_peak_dbfs"])
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, guard, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
        hard.append("导出削波")

    records.append({
        "id": vid, "h1_adjustable": h1_adj, "proc_stats": pstat,
        "h1_minus_h2_db": km["h1_minus_h2_db"], "target_h1_minus_h2_db": target_metrics["h1_minus_h2_db"],
        "rich_2_23_db": km["rich_2_23_db"], "target_rich_2_23_db": target_metrics["rich_2_23_db"],
        "rich_13_23_db": km["rich_13_23_db"], "target_rich_13_23_db": target_metrics["rich_13_23_db"],
        "profile": km["profile"],
        "centroid_hz": round(centroid, 0), "orig_centroid_hz": round(orig_centroid, 0),
        "target_centroid_hz": round(target_centroid, 0),
        "band_2_4k_rel_db": round(band_rel_energy(y, sr, 2000, 4000), 2),
        "orig_band_2_4k_rel_db": round(band_rel_energy(x, sr, 2000, 4000), 2),
        "target_band_2_4k_rel_db": round(band_rel_energy(xt, srt, 2000, 4000), 2),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "lag_samples": lag, "pitch_median_hz": round(pitch, 1),
        "raw_guard_db": guard, "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha(rp), "raw_lufs": ye_loud["integrated_lufs"],
        "raw_true_peak_dbtp": ye_loud["true_peak_dbtp"],
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    })

# ---------------- 统一响度 + 盲听 ----------------
orig_loud = measure_loudness(x, sr)
loud_all = {"original": orig_loud, **{r["id"]: {"integrated_lufs": r["raw_lufs"],
                                               "true_peak_dbtp": r["raw_true_peak_dbtp"]} for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loud_all.values()])

listen_root = OUT / "listening/audio"
export_wav(x, sr, target_lufs - orig_loud["integrated_lufs"], listen_root / "original.wav")
om, osr = read_pcm16(listen_root / "original.wav")
if abs(measure_loudness(om, osr)["integrated_lufs"] - target_lufs) > 0.3:
    raise RuntimeError("original loudness match failed")

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-014")
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
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

rows = []
for letter in blind.keys():
    r = next(x_ for x_ in records if x_["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4B 全谐波对齐盲听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:880px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P4B · 全谐波逐帧对齐（仅 A01）</h1>
<div class="note"><b>这是频谱路线的最后一次、最彻底尝试。</b><br>
本轮把第 1 到第 23 个泛音（约 175Hz–4025Hz，真正覆盖到 2–4kHz）<b>逐帧</b>对齐到你的正常版，
并第一次允许<b>降低基频</b>。两档唯一差别就是"基频是否参与对齐"。<br>
<b>预先约定：若这轮仍只有"一点点"，频谱路线确定性收工，不再有"还没试过"的悬念。</b></div>
<div class="card"><h3>试听（匿名，参数已隐藏）</h3><p class="meta">先听原音，再逐个听 A/B。判断：①有无变化；②是否更饱满（而非更亮/更响）；③是否自然像本人；④有无金属感、嘶嘶声、齿音、空洞。</p>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。
诚实提示：你的正常版频谱重心约 {target_centroid:.0f}Hz，薄音版约 {orig_centroid:.0f}Hz，所以"完整对齐"客观上会把重心推高，可能听起来更亮——这正是本轮要验证的。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-014", "phase": "F1-P4B", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "sample_display": DISPLAY,
    "mechanism": {
        "name": "全谐波逐帧对齐（方案 A 最后一次尝试）",
        "harmonics": f"H{K_MIN}-H{K_MAX}（基频 175Hz 时约覆盖到 4025Hz）",
        "frame_wise": True,
        "profile_normalization": "相对全带能量（80-8000Hz），使 H1 也可被调整",
        "caps": {"max_boost_db": CAP_UP_DB, "max_cut_db": CAP_DOWN_DB},
        "smoothing": f"增益时间一阶低通 tau={SMOOTH_TAU_S*1000:.0f}ms；掩膜时域 alpha={ALPHA_MASK}",
        "mask_width": f"逐谐波 k×±{VIBRATO_SPAN*100:.0f}% 颤音，谐波间距一半封顶",
        "fallback": f"周期性<{R_GATE} 或帧能量低于最响帧 {FLOOR_DB}dB → 不处理",
        "variants": {"W1": "H1 可调（完整对齐）", "W2": "H1 锁定不动（仅 H2-H23）"},
    },
    "profiles": {
        "thin": {"rel_db": baseline["profile"], "voiced_frames": baseline["voiced_frames"]},
        "target_normal": {"rel_db": target_metrics["profile"], "voiced_frames": target_metrics["voiced_frames"]},
        "deficit_thin_to_target_db": deficit,
    },
    "baseline_metrics": baseline, "target_metrics": target_metrics,
    "original_centroid_hz": round(orig_centroid, 0), "target_centroid_hz": round(target_centroid, 0),
    "target_lufs": target_lufs,
    "variants": records,
    "screen_rules": {
        "hard": ["H1-H2 与目标差 >2dB", "丰富度(2-23) 与目标差 >2dB", "|lag|>1",
                 "|Δpitch|>2Hz", "非有限值", "削波"],
        "diag": ["处理帧占比<40%", "增益触顶 +9dB", "衰减触底 -6dB"],
    },
    "limitations": [
        "逐帧剖面在 42.7ms 帧内测量，H13+ 谐波绝对能量低（-36 至 -45dB rel），测量与增益均噪声敏感",
        "目标剖面是 A02 的整段统计中位数，无法逐音节对应（不同 take）",
        "H1 降低可能带来空洞/变暗；这是本轮 W1 与 W2 的对照目的",
        "自实现机制未与外部工具校验；Agent 未听音，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "h1_adjustable", "h1_minus_h2_db", "target_h1_h2", "rich_2_23_db", "target_rich",
                "rich_13_23_db", "centroid_hz", "orig_centroid", "target_centroid", "band_2_4k_rel",
                "orig_2_4k", "target_2_4k", "process_fraction", "max_up", "max_down",
                "pitch", "lag", "raw_lufs", "raw_tp", "hard", "flags", "matched_lufs"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["h1_adjustable"], r["h1_minus_h2_db"], r["target_h1_minus_h2_db"],
                    r["rich_2_23_db"], r["target_rich_2_23_db"], r["rich_13_23_db"],
                    r["centroid_hz"], r["orig_centroid_hz"], r["target_centroid_hz"],
                    r["band_2_4k_rel_db"], r["orig_band_2_4k_rel_db"], r["target_band_2_4k_rel_db"],
                    r["proc_stats"]["process_fraction"], r["proc_stats"]["max_up_db"],
                    r["proc_stats"]["max_down_db"], r["pitch_median_hz"], r["lag_samples"],
                    round(r["raw_lufs"], 2), round(r["raw_true_peak_dbtp"], 2),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"]), m.get("lufs", "")])

params = {
    "experiment": "F1-EXP-014", "phase": "F1-P4B",
    "question": "全谐波 H1-H23 逐帧对齐（频谱路线最彻底形式）能否产生可听的饱满改善",
    "single_module": True, "no_stacking": True, "samples": [SAMPLE_ID], "sealed": SEALED,
    "target_reference": TARGET_ID,
    "mechanism": qa["mechanism"],
    "deficit_profile_db": deficit,
    "listening_target_lufs": target_lufs, "loudness_tolerance_lu": 0.3,
    "stop_condition": "本轮仍只有一点点 → 频谱路线确定性收工，转方案C接受边界",
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-014", "phase": "F1-P4B", "status": "pending_jade",
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
    "next_rule": "仍只有一点点 → 频谱路线确定性收工（方案C）；明确改善 → A05 复验",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-014\nphase: F1-P4B\n"
    f"full-harmonic H1-H23 frame-wise alignment; two variants (H1 adjustable vs locked)\n"
    f"sealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-014 结果（pending_jade）\n\n"
    "F1-P4B 全谐波逐帧对齐：H1-H23（约 175Hz–4025Hz，覆盖 2–4kHz），逐帧测量并补偿，"
    "剖面改为相对全带能量归一化，首次允许降低基频。两档唯一差别 = H1 是否参与对齐。\n\n"
    f"- 交付：{', '.join(r['id'] for r in records if r['deliverable'])}（匿名 A/B）\n"
    f"- 硬性故障：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    f"- 工程标记：{', '.join(r['id'] for r in records if r['diag_mark']) or '无'}\n\n"
    "预先约定：若本轮仍只有一点点，频谱路线确定性收工，转方案 C。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "baseline": {"h1_minus_h2": baseline["h1_minus_h2_db"], "rich_2_23": baseline["rich_2_23_db"],
                 "centroid": round(orig_centroid, 0)},
    "target": {"h1_minus_h2": target_metrics["h1_minus_h2_db"], "rich_2_23": target_metrics["rich_2_23_db"],
               "centroid": round(target_centroid, 0)},
    "variants": [{"id": r["id"], "h1_adj": r["h1_adjustable"],
                  "h1_minus_h2": r["h1_minus_h2_db"], "rich_2_23": r["rich_2_23_db"],
                  "rich_13_23": r["rich_13_23_db"], "centroid": r["centroid_hz"],
                  "band_2_4k": r["band_2_4k_rel_db"],
                  "process_frac": r["proc_stats"]["process_fraction"],
                  "max_up": r["proc_stats"]["max_up_db"], "max_down": r["proc_stats"]["max_down_db"],
                  "hard": r["hard_fail"], "flags": r["flags"]}
                 for r in records],
}, ensure_ascii=False, indent=1))
