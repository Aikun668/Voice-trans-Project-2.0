"""F1-EXP-015 / F1-P4B-fix: 修复 EXP-014 技术缺陷后的重跑。

EXP-014 的问题（已核查定位）：
  逐帧测量 + 逐帧不同掩膜 → STFT/WOLA 各帧相位不一致 → 互相抵消
  + 时间平滑摊薄，导致实际施加增益仅为目标的 20-40%，
  500-1000Hz 仅 +0.17dB、1000-2000Hz 甚至 -1.66dB。

本轮修复：
  增益表改为"整段统计的一把固定表"（像 EXP-011 那样有效），
  每帧掩膜按该帧 F0 定位谐波中心但增益值恒定 → 相位一致，无抵消。
  覆盖仍保持 H1-H23（含 2-4kHz）。
两档唯一差别：H1 是否参与对齐（X1 允许降低基频 -1.84dB，X2 锁定）。
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

OUT = ROOT / "experiments/F1-EXP-015"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-015 outputs already exist")
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
SEED = 20261104

K_MIN, K_MAX = 1, 23
CAP_UP_DB = 9.0
CAP_DOWN_DB = 6.0
STFT_N, STFT_H = 2048, 512
R_GATE = 0.35
FLOOR_DB = 40.0
VIBRATO_SPAN = 0.06
PEAK_SEARCH_HZ = 20.0
ALPHA_MASK = 0.6
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
FMIN, FMAX = 90.0, 450.0
BAND_EDGES = 8000.0
VARIANTS = [("X1", True), ("X2", False)]

# 验收门槛：实际频段增量必须达到"有效处理"的最低线（低于此说明修复失败）
MIN_EFFECTIVE_BAND_GAIN_DB = 2.0
CHECK_BANDS = [("500_1000", 500, 1000), ("1000_2000", 1000, 2000)]


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


def harmonic_power_profile(seg, sr, f0, kmax=K_MAX):
    nfft = 1 << 15
    w = np.hanning(len(seg))
    spec = np.fft.rfft(w * seg, nfft)
    power = np.abs(spec) ** 2
    bin_hz = sr / nfft
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= BAND_EDGES)
    total = float(power[band].sum())
    if total <= 0:
        return None
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
        prof[k] = 10.0 * np.log10(max(float(power[lo:hi + 1].max()), 1e-22) / total)
    return prof


def static_profile(x, sr, kmax=K_MAX):
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
        prof = harmonic_power_profile(seg, sr, f0, kmax)
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
    return float(10.0 * np.log10(max(sum(vals), 1e-20))) if vals else None


def key_metrics(x, sr):
    P, nv = static_profile(x, sr)
    return {"h1_minus_h2_db": round(P[1] - P[2], 2) if 2 in P else None,
            "rich_2_23_db": round(part_richness(P, 2, K_MAX), 2),
            "rich_13_23_db": round(part_richness(P, 13, K_MAX), 2),
            "profile": {k: round(v, 2) for k, v in P.items()},
            "voiced_frames": nv}


def spectral_centroid(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    f, p = f[m], p[m]
    return float(np.sum(f * p) / max(float(np.sum(p)), 1e-20))


def band_level_db(x, sr, lo, hi):
    f, p = signal.welch(x, sr, nperseg=min(8192, len(x)))
    return 10.0 * np.log10(max(float(p[(f >= lo) & (f < hi)].sum()), 1e-20))


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    yseg = y[:len(seg)]
    c = signal.correlate(yseg, seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


def apply_static_gains(x, sr, gain_table, h1_adjustable):
    """整段统一增益表 + 逐帧 F0 定位掩膜。与 EXP-011 相同的有效结构：
    增益值恒定，仅谐波中心随 F0 移动 → 各帧掩膜形状一致，无相位抵消。"""
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

    nb = N // 2 + 1
    masks = np.ones((n_frames, nb))
    n_applied = 0
    gains = {k: g for k, g in gain_table.items() if abs(g) >= 0.15}
    if not h1_adjustable:
        gains.pop(1, None)
    for i in range(n_frames):
        if rms_db[i] < gate or not np.isfinite(f0_med[i]):
            continue
        f0 = f0_med[i]
        spacing = f0 / bin_hz
        width_cap = max(2, int(spacing // 2))
        mask = masks[i]
        touched = False
        for k, g_db in gains.items():
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
                mask[b] = 1.0 + (g_lin - 1.0) * shape
            touched = True
        if touched:
            n_applied += 1
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
    stats = {"frames_total": int(n_frames), "frames_processed": int(n_applied),
             "process_fraction": round(n_applied / n_frames, 3),
             "gain_table": {str(k): round(v, 2) for k, v in gains.items()}}
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


# ---------------- 输入与增益表 ----------------
x, sr = load(DATASET_ID)
xt, srt = load(TARGET_ID)

P_thin, nv_thin = static_profile(x, sr)
P_target, nv_target = static_profile(xt, srt)

deficit = {k: round(P_target[k] - P_thin[k], 2) for k in range(K_MIN, K_MAX + 1)
           if k in P_target and k in P_thin}
gain_table = {k: round(min(max(deficit[k], -CAP_DOWN_DB), CAP_UP_DB), 2)
              for k in deficit}
if not gain_table:
    raise RuntimeError("empty gain table")

baseline = key_metrics(x, sr)
target_metrics = key_metrics(xt, srt)
orig_pitch = float(np.nanmedian(estimate_pitch(x, sr)))
orig_centroid = spectral_centroid(x, sr)
target_centroid = spectral_centroid(xt, srt)
orig_bands = {lab: band_level_db(x, sr, lo, hi) for lab, lo, hi in CHECK_BANDS}

listening_contract = {
    "原音问题登记": "A01 自评薄音（'轻、有点嗲'）；EXP-013 已排除响度主因",
    "素材与片段": "A01 全长 13.14s，不截取",
    "预期可听变化": "修复版：整段统一增益真正施加（500-2000Hz 实际提升应 ≥2dB），覆盖 H1-H23",
    "否决副作用": "金属感/嗡嗡声、嘶嘶声、齿音变重、闷或空洞、像别人；只变亮/只变响不算通过",
    "本轮特别说明": "EXP-014 因逐帧相位抵消未真正生效；本轮为同机制的正确实现",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-015", "phase": "F1-P4B-fix",
    "fix_of": "F1-EXP-014（逐帧掩膜相位抵消缺陷）",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "display": DISPLAY,
    "target_reference": {"dataset_id": TARGET_ID, "display": TARGET_DISPLAY},
    "sealed_not_processed": SEALED,
    "frames": len(x), "sample_rate": sr,
    "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 处理两档 ----------------
raw_root = OUT / "outputs/processed_raw"
records = []
for vid, h1_adj in VARIANTS:
    y, pstat = apply_static_gains(x, sr, gain_table, h1_adj)
    st = sample_stats(y, sr)
    km = key_metrics(y, sr)
    lag = lag_between(y, x, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    centroid = spectral_centroid(y, sr)
    y_bands = {lab: band_level_db(y, sr, lo, hi) for lab, lo, hi in CHECK_BANDS}
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0
    hard, flags = [], []

    # 核心修复验收：实际频段增量是否真的达到有效处理线
    band_gain = {lab: round(y_bands[lab] - orig_bands[lab], 2) for lab, _, _ in CHECK_BANDS}
    raw_lufs_now = measure_loudness(y, sr)
    for lab in band_gain:
        # 以相对处理前的原始响度补偿后再比较（避免响度差异混淆）
        effective = band_gain[lab] - (raw_lufs_now["integrated_lufs"] - measure_loudness(x, sr)["integrated_lufs"])
        if effective < MIN_EFFECTIVE_BAND_GAIN_DB:
            hard.append(f"{lab} 实际增量不足 {effective:+.2f}dB (<{MIN_EFFECTIVE_BAND_GAIN_DB}dB) → 修复未生效")
    if h1_adj and (km["h1_minus_h2_db"] is None
                   or abs(km["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"]) > 2.0):
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
    if max(gain_table.values()) >= CAP_UP_DB - 0.1:
        flags.append(f"增益触顶 +{max(gain_table.values()):.1f}dB，金属感风险")

    guard = min(0.0, -0.5 - st["sample_peak_dbfs"])
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, guard, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
        hard.append("导出削波")

    records.append({
        "id": vid, "h1_adjustable": h1_adj, "proc_stats": pstat,
        "band_gain_raw_db": band_gain,
        "h1_minus_h2_db": km["h1_minus_h2_db"], "target_h1_minus_h2_db": target_metrics["h1_minus_h2_db"],
        "rich_2_23_db": km["rich_2_23_db"], "target_rich_2_23_db": target_metrics["rich_2_23_db"],
        "rich_13_23_db": km["rich_13_23_db"],
        "centroid_hz": round(centroid, 0), "orig_centroid_hz": round(orig_centroid, 0),
        "target_centroid_hz": round(target_centroid, 0),
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
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-015")
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
    r = next(v for v in records if v["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4B 修复版盲听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:880px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P4B · 修复版（仅 A01）</h1>
<div class="note"><b>这轮修的是上一轮的技术 bug，不是换方案。</b><br>
上一轮的逐帧增益互相抵消，实际只提升了 0.17dB（1000–2000Hz 甚至是负的），所以你听不出来。
本轮改成整段统一增益表，覆盖仍然是第 1–23 个泛音（约 175–4025Hz），<b>这次增益是真施加下去的</b>。<br>
两档唯一差别：是否允许降低基频。</div>
<div class="card"><h3>试听（匿名）</h3><p class="meta">先听原音，再逐个听 A/B。判断：①有无变化；②是否更饱满（而不是更亮/更响）；③是否自然像本人；④有无金属感、嘶嘶声、齿音、空洞。</p>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。
本轮工程门槛已提高到"500–1000Hz 与 1000–2000Hz 实际增量必须 ≥2dB"，达不到会直接判技术失败，不会再出现"没生效却当结论"的情况。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-015", "phase": "F1-P4B-fix", "status": "pending_jade",
    "fix_of": "F1-EXP-014",
    "root_cause_in_014": {
        "issue": "逐帧测量 + 逐帧不同掩膜 → STFT/WOLA 相位不一致 → 相互抵消；时间平滑进一步摊薄",
        "evidence": "EXP-014 实测 W1 相对原音：500-1000Hz 仅 +0.17dB，1000-2000Hz −1.66dB；样本差异 RMS 仅 −12.57dB（V3 为 −10.33dB）",
        "consequence": "EXP-014 的听感数据不可用于判定频谱路线有效性",
    },
    "fix": "增益表改为整段统计的固定表；每帧掩膜按该帧 F0 定位谐波中心但增益值恒定 → 各帧相位一致",
    "mechanism": {
        "harmonics": f"H{K_MIN}-H{K_MAX}",
        "gain_table_db": gain_table,
        "caps": {"max_boost_db": CAP_UP_DB, "max_cut_db": CAP_DOWN_DB},
        "variants": {"X1": "H1 参与对齐（可降低）", "X2": "H1 锁定"},
    },
    "deficit_thin_to_target_db": deficit,
    "baseline_metrics": baseline, "target_metrics": target_metrics,
    "original_centroid_hz": round(orig_centroid, 0), "target_centroid_hz": round(target_centroid, 0),
    "target_lufs": target_lufs,
    "variants": records,
    "new_gate": {
        "rule": f"500-1000Hz 与 1000-2000Hz 的实际增量（扣除响度变化）必须 ≥ {MIN_EFFECTIVE_BAND_GAIN_DB}dB",
        "purpose": "防止再次出现'处理未生效却被当作结论'",
    },
    "limitations": [
        "2-4kHz 的缺口主要是宽带能量而非谐波，谐波增益最多补上约 22%，本轮不改变这一点",
        "目标剖面是 A02 整段统计中位数，无法逐音节对应",
        "自实现机制未与外部工具校验；Agent 未听音，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "h1_adjustable", "h1_minus_h2_db", "target_h1_h2", "rich_2_23_db", "target_rich",
                "rich_13_23_db", "band_500_1000_gain", "band_1000_2000_gain",
                "centroid_hz", "orig_centroid", "target_centroid", "process_fraction",
                "pitch", "lag", "raw_lufs", "raw_tp", "hard", "flags", "matched_lufs"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["h1_adjustable"], r["h1_minus_h2_db"], r["target_h1_minus_h2_db"],
                    r["rich_2_23_db"], r["target_rich_2_23_db"], r["rich_13_23_db"],
                    r["band_gain_raw_db"]["500_1000"], r["band_gain_raw_db"]["1000_2000"],
                    r["centroid_hz"], r["orig_centroid_hz"], r["target_centroid_hz"],
                    r["proc_stats"]["process_fraction"], r["pitch_median_hz"], r["lag_samples"],
                    round(r["raw_lufs"], 2), round(r["raw_true_peak_dbtp"], 2),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"]), m.get("lufs", "")])

params = {
    "experiment": "F1-EXP-015", "phase": "F1-P4B-fix",
    "question": "修复相位抵消后，全谐波对齐真正生效时，听感是否产生可听的饱满改善",
    "fix_of": "F1-EXP-014", "single_module": True, "no_stacking": True,
    "samples": [SAMPLE_ID], "sealed": SEALED, "target_reference": TARGET_ID,
    "gain_table_db": gain_table, "deficit_profile_db": deficit,
    "listening_target_lufs": target_lufs,
    "effective_gain_gate_db": MIN_EFFECTIVE_BAND_GAIN_DB,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-015", "phase": "F1-P4B-fix", "status": "pending_jade",
    "fix_of": "F1-EXP-014",
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
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-015\nphase: F1-P4B-fix\n"
    f"fixed EXP-014 frame-wise phase cancellation by using one static gain table\n"
    f"sealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-015 结果（pending_jade）\n\n"
    "修复 EXP-014 的相位抵消缺陷：增益表改为整段统一（每帧掩膜形状一致），"
    "覆盖仍为 H1–H23。两档唯一差别 = 是否允许降低基频。\n\n"
    f"- 交付：{', '.join(r['id'] for r in records if r['deliverable'])}\n"
    f"- 技术失败：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    f"- 工程标记：{', '.join(r['id'] for r in records if r['diag_mark']) or '无'}\n"
    f"- 新增门槛：500–1000Hz / 1000–2000Hz 实际增量必须 ≥{MIN_EFFECTIVE_BAND_GAIN_DB}dB\n\n"
    "听感结论待 Jade 试听后填写。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "gain_table_db": gain_table,
    "baseline": {"h1_minus_h2": baseline["h1_minus_h2_db"], "rich_2_23": baseline["rich_2_23_db"],
                 "centroid": round(orig_centroid, 0)},
    "target": {"h1_minus_h2": target_metrics["h1_minus_h2_db"], "rich_2_23": target_metrics["rich_2_23_db"],
               "centroid": round(target_centroid, 0)},
    "variants": [{"id": r["id"], "h1_adj": r["h1_adjustable"],
                  "h1_minus_h2": r["h1_minus_h2_db"], "rich_2_23": r["rich_2_23_db"],
                  "band_500_1000": r["band_gain_raw_db"]["500_1000"],
                  "band_1000_2000": r["band_gain_raw_db"]["1000_2000"],
                  "centroid": r["centroid_hz"],
                  "process_frac": r["proc_stats"]["process_fraction"],
                  "hard": r["hard_fail"], "flags": r["flags"]}
                 for r in records],
}, ensure_ascii=False, indent=1))
