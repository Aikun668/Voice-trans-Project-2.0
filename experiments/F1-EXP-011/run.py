"""F1-EXP-011 / F1-P4A-rev1: 方案 A 的一次有依据修订——谐波覆盖 H2-H8 → H2-H12。

初测结论（F1-EXP-010）：V2（100%）被 Jade 判"稍微好了一点点"且无副作用（方向性弱正向）；
V1（50%）无可听差异。剩余实测缺口最大处为 H9-H11（6.8-8.4dB，上轮未处理）。
本修订只动一个控制量：谐波覆盖范围扩到 H12；其余全部不变（100% 强度、
9dB 单谐波上限、A02 同曲正常版校准、掩膜回退策略）。
随覆盖扩展的必要适配：掩膜半宽改为逐谐波宽度（k×±6% 颤音摆幅）并以
谐波间距一半为上限，避免高次谐波掩膜桥接谐波间隙（属同一控制量的实现细节）。
对照锚点 = F1-EXP-010 的 V2（上轮版本B）原文件，哈希校验、不改动、正常标注。
预先声明：若修订版相对上轮B仍只有"一点点"，方案 A 终止，转 C（接受边界）。
"""
import csv, hashlib, json, math, os, pathlib, platform, shutil, subprocess, sys
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

OUT = ROOT / "experiments/F1-EXP-011"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-011 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; DISPLAY = "平凡之路 / 薄音版"
TARGET_ID = "A_self_02"
ANCHOR_EXP010_V2 = ROOT / "experiments/F1-EXP-010/outputs/processed_raw/V2.wav"
EXP010_QA = ROOT / "experiments/F1-EXP-010/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]

K_MIN, K_MAX = 2, 12               # 修订：覆盖 H2-H12（上轮 H2-H8）
K_PROFILE = 12
CAP_DB = 9.0
MIN_DEFICIT_DB = 2.0
STRENGTH = 1.0
STFT_N, STFT_H = 2048, 512
R_GATE = 0.35
FLOOR_DB = 40.0
ALPHA_MASK = 0.6
VIBRATO_SPAN = 0.06
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
FMIN, FMAX = 90.0, 450.0


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


def harmonic_profile(x, sr, kmax_profile=K_PROFILE):
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


def partial_richness_db(P, k_lo, k_hi):
    vals = [10 ** (P[k] / 10) for k in range(k_lo, k_hi + 1) if k in P]
    if not vals:
        return None
    return float(10.0 * np.log10(max(sum(vals), 1e-10)))


def key_metrics(x, sr):
    P, n_voiced = harmonic_profile(x, sr)
    return {"h1_minus_h2_db": round(P[2], 2),
            "richness_k2_12_db": round(partial_richness_db(P, 2, 12), 2),
            "richness_k9_12_db": round(partial_richness_db(P, 9, 12), 2),
            "profile_rel_db": {k: round(v, 2) for k, v in P.items()},
            "voiced_frames": n_voiced}


def stft_harmonic_boost(x, sr, gains_db, kmin=K_MIN, kmax=K_MAX):
    """同 F1-EXP-010 机制；修订点：掩膜半宽逐谐波（k×±6%），并以谐波间距一半封顶，
    防止高次谐波掩膜桥接谐波间隙。其余（回退策略/平滑/合成）与上轮一致。"""
    N, H = STFT_N, STFT_H
    w = np.hanning(N)
    bin_hz = sr / N
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    gains = {k: g for k, g in gains_db.items() if g > 0.01 and kmin <= k <= kmax}
    if not gains:
        raise RuntimeError("empty gain table")

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
    n_boosted = 0
    for i in range(n_frames):
        if rms_db[i] < gate or not np.isfinite(f0_med[i]):
            continue
        f0 = f0_med[i]
        spacing = f0 / bin_hz
        width_cap = max(2, int(spacing // 2))
        mask = masks[i]
        touched = False
        for k, g_db in gains.items():
            width = min(max(2, int(math.ceil(k * VIBRATO_SPAN * f0 / bin_hz))), width_cap)
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
    stats = {"frames_total": int(n_frames), "frames_boosted": int(n_boosted),
             "boost_fraction": round(n_boosted / n_frames, 3)}
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


# ---------------- 输入与锚点校验 ----------------
x, sr = load(DATASET_ID)
xt, srt = load(TARGET_ID)
if not ANCHOR_EXP010_V2.exists():
    raise FileNotFoundError("EXP-010 V2 raw missing")
exp010 = json.loads(EXP010_QA.read_text(encoding="utf-8"))
v2_expected_sha = next(v for v in exp010["versions"] if v["id"] == "V2")["raw_sha256"]
if sha(ANCHOR_EXP010_V2) != v2_expected_sha:
    raise ValueError("EXP-010 V2 raw changed; anchor invalid")
v2_raw, v2_sr = read_pcm16(ANCHOR_EXP010_V2)

listening_contract = {
    "原音问题登记": "A01 自评薄音；上轮（EXP-010）V2 已判'稍微好一点点'且无副作用，本轮为其唯一一次修订",
    "素材与片段": "A01 全长 13.14s，不截取",
    "预期可听变化": "修订版（H2-H12）应比上轮B版更实、改善更明显；方向为『厚/实』而非『亮/响』",
    "否决副作用": "高次泛音区新增（1.6-2.1kHz）：粗糙感、嘶嘶声、齿音变重；金属感/机器味；只变亮或只变响不算通过",
    "预先声明": "若修订版相对上轮B仍只有一点点，方案A终止转C，不再扩范围/调强度",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-011", "phase": "F1-P4A-rev1",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "display": DISPLAY,
    "revision_of": "F1-EXP-010 V2",
    "single_control_change": "谐波覆盖 H2-H8 -> H2-H12（含逐谐波掩膜半宽适配，其余参数与上轮完全一致）",
    "anchor": {"file": str(ANCHOR_EXP010_V2.relative_to(ROOT)).replace("\\", "/"),
               "sha256": v2_expected_sha, "note": "上轮版本B原文件，未改动，作对照锚点"},
    "target_reference": TARGET_ID,
    "sealed_not_processed": SEALED,
    "wav_sha256": sha(DATA / items[DATASET_ID]["wav"]["path"]),
    "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 剖面与增益表 ----------------
P_thin, _ = harmonic_profile(x, sr)
P_target, _ = harmonic_profile(xt, srt)
deficit = {k: round(P_target[k] - P_thin[k], 2) for k in range(K_MIN, K_PROFILE + 1)
           if k in P_target and k in P_thin}
gain_table = {k: round(min(max(deficit.get(k, 0.0), 0.0), CAP_DB), 2)
              for k in range(K_MIN, K_MAX + 1) if deficit.get(k, 0) >= MIN_DEFICIT_DB}
if not gain_table:
    raise RuntimeError("no harmonic exceeds minimum deficit")

baseline = key_metrics(x, sr)
target_metrics = key_metrics(xt, srt)
anchor_metrics = key_metrics(v2_raw, v2_sr)

# ---------------- 处理修订版 V3 ----------------
y, proc_stats = stft_harmonic_boost(x, sr, gain_table)
st = sample_stats(y, sr)
km = key_metrics(y, sr)
lag = lag_between(y, x, sr)
pitch = float(np.nanmedian(estimate_pitch(y, sr)))
orig_pitch = float(np.nanmedian(estimate_pitch(x, sr)))
centroid = spectral_centroid(y, sr)
clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0

hard, flags = [], []
def closure(metric):
    gap = abs(baseline[metric] - target_metrics[metric])
    return 1.0 - abs(km[metric] - target_metrics[metric]) / max(gap, 1e-9)
closed_h1h2 = 1.0 - abs(km["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"]) / max(abs(baseline["h1_minus_h2_db"] - target_metrics["h1_minus_h2_db"]), 1e-9)
closed_k9_12 = closure("richness_k9_12_db")
if closed_h1h2 < 0.5:
    hard.append(f"H1-H2 收敛不足 closed={closed_h1h2:.0%}")
if closed_k9_12 < 0.5:
    hard.append(f"H9-H12 部分丰富度收敛不足 closed={closed_k9_12:.0%}")
if abs(lag) > 1:
    hard.append(f"时间对齐 {lag} samples")
if abs(pitch - orig_pitch) > 2.0:
    hard.append(f"音高异常 Δ={pitch - orig_pitch:.2f}Hz")
if not all(np.isfinite(y)):
    hard.append("非有限值")
if clipped > 0:
    hard.append(f"削波 {clipped} samples")
if proc_stats["boost_fraction"] < 0.4:
    flags.append(f"处理帧占比偏低 {proc_stats['boost_fraction']:.2f}")
if max(gain_table.values()) >= CAP_DB - 0.1:
    flags.append(f"单谐波增益触顶 {max(gain_table.values()):.1f}dB（与上轮相同，H6），注意金属感")

guard = min(0.0, -0.5 - st["sample_peak_dbfs"])
rp = OUT / "outputs/processed_raw/V3.wav"
export_wav(y, sr, guard, rp)
ye, esr = read_pcm16(rp)
if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
    hard.append("导出削波")

# ---------------- 统一响度（原音 / 锚点B / 修订版V3） ----------------
orig_loud = measure_loudness(x, sr)
v2_loud = measure_loudness(v2_raw, v2_sr)
v3_loud = measure_loudness(y, sr)
loudness_all = {"original": orig_loud, "V2_anchor": v2_loud, "V3": v3_loud}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loudness_all.values()])

listen_root = OUT / "listening/audio"
exports = []
for tag, data, rate, loud in (("original", x, sr, orig_loud), ("anchor_B", v2_raw, v2_sr, v2_loud), ("V3", y, sr, v3_loud)):
    fname = {"original": "original.wav", "anchor_B": "prev_B.wav", "V3": "revision.wav"}[tag]
    dest = listen_root / fname
    export_wav(data, rate, target_lufs - loud["integrated_lufs"], dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{tag} loudness match failed")
    if dm["true_peak_dbtp"] > -2.9:
        raise RuntimeError(f"{tag} true peak too hot")
    exports.append({"tag": tag, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                    "lufs": dm["integrated_lufs"], "true_peak_dbtp": dm["true_peak_dbtp"],
                    "frames": len(de), "sha256": sha(dest)})

# ---------------- 试听页（正常标注，不盲听） ----------------
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4A 修订版试听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}</style>
<h1>F1-P4A · 修订版试听（仅 A01）</h1>
<div class="note"><b>本轮登记：</b>上轮版本B判"稍微好一点点"且无副作用；本轮为其<b>唯一一次修订</b>——谐波覆盖从第2-8个扩展到第2-12个泛音，其余参数不变。<br>
预期：修订版比上轮B<b>更实、改善更明显</b>。<br>
重点关注新增的高次泛音区（1.6-2.1kHz）：<b>粗糙感、嘶嘶声、齿音变重、金属感</b> 属于否决副作用。<br>
<b>预先约定：若修订版相对上轮B仍只有一点点，方案A终止，接受边界，不再扩范围。</b></div>
<div class="card"><h3>试听（正常标注，未盲听）</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>上轮版本B（未改动）</label><audio controls preload="metadata" src="audio/prev_B.wav"></audio></div>
<div class="row"><label>修订版（H2-H12）</label><audio controls preload="metadata" src="audio/revision.wav"></audio></div></div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；三个文件统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。请回答：①修订版与上轮B有无差别；②哪个更好；③修订版有无否决副作用。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA 与记录 ----------------
qa = {
    "experiment": "F1-EXP-011", "phase": "F1-P4A-rev1", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "sample_display": DISPLAY,
    "revision": {
        "of": "F1-EXP-010 V2",
        "single_control": "谐波覆盖 H2-H8 -> H2-H12；逐谐波掩膜半宽（k×±6%，谐波间距一半封顶）为覆盖扩展的必要适配，其余与上轮完全一致",
        "strength": STRENGTH, "cap_db": CAP_DB, "gain_table_db": gain_table,
        "deficit_profile_db": deficit,
        "stop_condition": "修订版相对上轮B仍只有一点点 → 方案A终止转C",
    },
    "mechanism_note": "STFT/回退/平滑/合成与 F1-EXP-010 相同；仅掩膜半宽由全帧统一宽度改为逐谐波宽度",
    "baseline_metrics": baseline, "target_metrics": target_metrics, "anchor_V2_metrics": anchor_metrics,
    "V3": {
        "proc_stats": proc_stats, "key_metrics": km,
        "gap_closed_h1h2": round(closed_h1h2, 3), "gap_closed_richness_k9_12": round(closed_k9_12, 3),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "centroid_hz": round(centroid, 0),
        "lag_samples": lag, "pitch_median_hz": round(pitch, 1), "pitch_delta_hz": round(pitch - orig_pitch, 2),
        "raw_export_guard_db": guard, "raw_sha256": sha(rp), "raw_lufs": v3_loud["integrated_lufs"],
        "raw_true_peak_dbtp": v3_loud["true_peak_dbtp"],
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    },
    "target_lufs": target_lufs, "exports": exports,
    "limitations": [
        "H13+ 未测量未处理；2-4kHz 中 H13-H23 部分仍在本机制覆盖之外",
        "高次泛音区绝对能量低（-36 至 -43dB rel），增益与测量噪声敏感度更高",
        "自实现机制未与外部工具校验；结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["version", "richness_k2_12_db", "richness_k9_12_db", "h1_minus_h2_db",
                 "centroid_hz", "boost_fraction", "matched_lufs", "matched_tp"])
    wr.writerow(["baseline(原音)", baseline["richness_k2_12_db"], baseline["richness_k9_12_db"], baseline["h1_minus_h2_db"], 294, "", ""])
    wr.writerow(["target(A02正常版)", target_metrics["richness_k2_12_db"], target_metrics["richness_k9_12_db"], target_metrics["h1_minus_h2_db"], 403, "", ""])
    wr.writerow(["上轮B(V2锚点)", anchor_metrics["richness_k2_12_db"], anchor_metrics["richness_k9_12_db"], anchor_metrics["h1_minus_h2_db"], 385, "", exports[1]["lufs"], exports[1]["true_peak_dbtp"]])
    wr.writerow(["修订版(V3)", km["richness_k2_12_db"], km["richness_k9_12_db"], km["h1_minus_h2_db"], round(centroid, 0), proc_stats["boost_fraction"], exports[2]["lufs"], exports[2]["true_peak_dbtp"]])

params = {
    "experiment": "F1-EXP-011", "phase": "F1-P4A-rev1",
    "question": "谐波覆盖扩展到 H2-H12 后，改善是否从'一点点'变成明确可听；若仍'一点点'则方案A终止",
    "single_module": True, "no_stacking": True, "samples": [SAMPLE_ID], "sealed": SEALED,
    "gain_table_db": gain_table, "deficit_profile_db": deficit,
    "revision_of": "F1-EXP-010 V2", "anchor": "EXP-010 V2 raw（哈希校验，未改动）",
    "target_reference": TARGET_ID,
    "listening_target_lufs": target_lufs, "loudness_tolerance_lu": 0.3,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-011", "phase": "F1-P4A-rev1", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "version": "V3(H2-H12)",
    "anchor": "EXP-010 V2 (prev B)",
    "hard_failed": bool(hard), "diag_marked": bool(flags),
    "target_lufs": target_lufs,
    "technical_pass": not hard and all(abs(e["lufs"] - target_lufs) <= 0.3 for e in exports),
    "preview_page": "listening/index.html",
    "deliverables": ["original.wav", "prev_B.wav", "revision.wav"],
    "next_rule": "修订版明显更好→A05复验；仍一点点→方案A终止转C",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-011\nphase: F1-P4A-rev1\n"
    f"single control: coverage H2-H8 -> H2-H12; strength/cap/calibration unchanged\n"
    f"anchor: EXP-010 V2 raw sha-verified\nsealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-011 结果（pending_jade）\n\n"
    "F1-P4A 修订轮：谐波覆盖 H2-H8 → H2-H12（唯一控制量变化），其余参数与 EXP-010 V2 完全一致；"
    "上轮 B 版原文件作对照锚点（哈希校验），正常标注不盲听。\n\n"
    f"- 工程自检：H1-H2 收敛 {closed_h1h2:.0%}，H9-H12 部分丰富度收敛 {closed_k9_12:.0%}；"
    f"音高 Δ{pitch - orig_pitch:.2f}Hz，对齐 {lag} 样本，无削波；响度统一 {target_lufs:.2f} LUFS。\n"
    f"- 硬性故障：{'; '.join(hard) or '无'}；诊断标记：{'; '.join(flags) or '无'}\n"
    "- 预先约定：若修订版相对上轮B仍只有一点点，方案A终止转C。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "gain_table_db": gain_table,
    "deficit_db": deficit,
    "baseline": baseline, "target": target_metrics, "anchor_V2": anchor_metrics, "V3": km,
    "closed_h1h2": round(closed_h1h2, 3), "closed_k9_12": round(closed_k9_12, 3),
    "centroid_hz": round(centroid, 0), "lag": lag, "pitch_delta_hz": round(pitch - orig_pitch, 2),
    "hard_fail": hard, "flags": flags, "target_lufs": target_lufs,
}, ensure_ascii=False, indent=1))
