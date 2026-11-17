"""F1-EXP-019 / F1-P5: 同歌手跨歌曲参考迁移验证。

问题：用**同一歌手的另一首歌的正常演唱**作为参考，能否美化目标歌曲的单薄版本？

参考：A_self_06（雅俗共赏·正常版），F0 中值 ≈187Hz
目标：A_self_01（平凡之路·薄音版），F0 中值 ≈176Hz

硬约束（本轮）：
  - 不读取 A_self_02（A01 的同曲正常版）用于任何参数拟合；
    A02 仅作为对照基准存在（其最佳产物 EXP-016 M2.wav 直接引用，不重算）。
  - 不新增 DSP 模块；X1/M2 机制原样保留，只更换参考。
  - 不改音高/节奏/咬字；不碰 Holdout（A06 仅作参考，不被处理、不被评估）。

跨歌曲的关键差异（必须在报告中记录）：
  同曲时可以用"逐谐波差值"；跨歌曲时两首歌的元音/段落/音区不同，
  直接逐 k 对齐会把"元音差异"当成"薄→正常差异"。
  因此本轮只搬**统计层面**的差异（谐波剖面 + 包络），并以 A08（另一首正常版）
  做交叉验证，判断 A06 的剖面是否具备"正常演唱的代表性"。
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

OUT = ROOT / "experiments/F1-EXP-019"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-019 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

TARGET_ID = "A_self_01"       # 平凡之路 · 薄音版
REF_ID = "A_self_06"          # 雅俗共赏 · 正常版（跨歌曲参考）
CROSS_CHECK_ID = "A_self_08"  # 因为刚好遇见你 · 原版（仅只读交叉验证）
EXCLUDED_ID = "A_self_02"     # A01 同曲正常版：本轮禁止用于拟合
SAME_SONG_BEST = ROOT / "experiments/F1-EXP-016/outputs/processed_raw/M2.wav"
EXP016_QA = ROOT / "experiments/F1-EXP-016/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]

K_MAX = 23
CAP_UP_DB, CAP_DOWN_DB = 9.0, 6.0
STFT_N, STFT_H = 2048, 512
R_GATE, FLOOR_DB = 0.35, 40.0
VIBRATO_SPAN = 0.06
ALPHA_MASK = 0.6
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
FMIN, FMAX = 90.0, 450.0
PEAK_SEARCH_HZ = 20.0
BAND_EDGES = 8000.0
MORPH_LO, MORPH_HI, MORPH_STEP = 1000.0, 8000.0, 250.0
MORPH_CLAMP = (-3.0, 5.0)
GATE_MAX_DEVIATION_DB = 1.2
SEED = 20261108


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


def harmonic_profile(x, sr, kmax=K_MAX):
    win, hop = int(round(sr * ANA_WIN_S)), int(round(sr * ANA_HOP_S))
    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB
    coll = {k: [] for k in range(1, kmax + 1)}
    f0s = []
    nfft = 1 << 15
    halfw = int(round(PEAK_SEARCH_HZ / (sr / nfft)))
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        f0s.append(f0)
        w = np.hanning(win) * seg
        power = np.abs(np.fft.rfft(w, nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        total = float(power[(freqs >= 80) & (freqs <= BAND_EDGES)].sum())
        if total <= 0:
            continue
        for k in range(1, kmax + 1):
            fc = k * f0
            if fc > BAND_EDGES - PEAK_SEARCH_HZ:
                break
            c0 = int(round(fc / (sr / nfft)))
            lo, hi = max(c0 - halfw, 1), min(c0 + halfw, len(power) - 1)
            if hi <= lo:
                break
            coll[k].append(10.0 * np.log10(max(float(power[lo:hi + 1].max()), 1e-22) / total))
    if len(f0s) < 50:
        raise RuntimeError("voiced frames too few")
    P = {k: float(np.median(v)) for k, v in coll.items() if len(v) >= 30}
    return P, float(np.median(f0s))


def part_richness(prof, k_lo, k_hi):
    vals = [10 ** (prof[k] / 10.0) for k in range(k_lo, k_hi + 1) if k in prof]
    return float(10.0 * np.log10(max(sum(vals), 1e-20))) if vals else None


def spectral_centroid(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    return float(np.sum(f[m] * p[m]) / max(float(np.sum(p[m])), 1e-20))


def band_levels(x, sr, lo=MORPH_LO, hi=MORPH_HI, step=MORPH_STEP):
    f, psd = signal.welch(x, sr, nperseg=min(16384, len(x)))
    edges = np.arange(lo, hi + step / 2, step)
    out = {}
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        m = (f >= a) & (f < b)
        out[(float(a), float(b))] = 10.0 * np.log10(max(float(psd[m].sum()), 1e-20))
    return out


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    c = signal.correlate(y[:len(seg)], seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


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


def apply_harmonic_profile(x, sr, target_prof, src_prof):
    """X1 机制（与 EXP-015 修复版一致）：整段统一增益表 + 逐帧 F0 定位掩膜。
    增益 = 目标剖面 - 当前剖面，按**绝对频率**（k·F0）插值，钳位 [-6,+9]。"""
    N, H = STFT_N, STFT_H
    w = np.hanning(N)
    bin_hz = sr / N
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    nb = N // 2 + 1

    # 建立绝对频率增益曲线（用目标素材的 F0 中值把谐波序数映射到绝对频率）
    ks = sorted(set(target_prof) & set(src_prof))
    _, f0_tgt = harmonic_profile(x, sr)
    f_grid = [k * f0_tgt for k in ks]
    g_grid = [float(np.clip(target_prof[k] - src_prof[k], -CAP_DOWN_DB, CAP_UP_DB)) for k in ks]

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

    masks = np.ones((n_frames, nb))
    n_applied = 0
    kmax = int(BAND_EDGES // FMIN) + 2
    for i in range(n_frames):
        if rms_db[i] < gate or not np.isfinite(f0_med[i]):
            continue
        f0 = f0_med[i]
        spacing = f0 / bin_hz
        width_cap = max(2, int(spacing // 2))
        mask = masks[i]
        touched = False
        for k in range(1, kmax + 1):
            fc = k * f0
            if fc >= BAND_EDGES - PEAK_SEARCH_HZ:
                break
            g_db = float(np.interp(fc, f_grid, g_grid, left=0.0, right=0.0))
            if abs(g_db) < 0.15:
                continue
            center = fc / bin_hz
            if center >= nb - 1:
                break
            width = min(max(2, int(math.ceil(k * VIBRATO_SPAN * f0 / bin_hz))), width_cap)
            lo, hi = max(int(math.floor(center)) - width, 1), min(int(math.ceil(center)) + width, nb - 1)
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
        y = np.fft.irfft(np.fft.rfft(fr) * masks[i], N)
        s = i * H
        y_ola[s:s + N] += y * w
        norm[s:s + N] += w * w
    y = (y_ola / np.maximum(norm, 1e-8))[pad:pad + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y, {"process_fraction": round(n_applied / n_frames, 3),
               "gain_curve": {f"{round(f)}Hz": round(g, 2) for f, g in zip(f_grid, g_grid)}}


def apply_envelope(x, sr, x_ref, sr_ref):
    """M2 机制：把参考的剩余包络差应用到 x（响度归一后比较）。"""
    lx = measure_loudness(x, sr)["integrated_lufs"]
    lr = measure_loudness(x_ref, sr_ref)["integrated_lufs"]
    bx = band_levels(x, sr)
    br = band_levels(x_ref, sr_ref)
    keys = sorted(bx.keys())
    gaps = np.array([(br[k] - lr) - (bx[k] - lx) for k in keys])
    sm = np.copy(gaps)
    for i in range(len(gaps)):
        lo, hi = max(0, i - 1), min(len(gaps), i + 2)
        sm[i] = float(np.mean(gaps[lo:hi]))
    sm = np.clip(sm, MORPH_CLAMP[0], MORPH_CLAMP[1])
    centers = [(a + b) / 2 for (a, b) in keys]
    f_pts = [0.0, MORPH_LO] + centers + [MORPH_HI + MORPH_STEP]
    g_pts = [0.0, 0.0] + list(sm) + [0.0]
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.interp(f, f_pts, g_pts, left=0.0, right=0.0)
    y = np.fft.irfft(np.fft.rfft(x) * 10.0 ** (g / 20.0), n)
    return y, {"curve": {f"{int(c)}Hz": round(float(v), 2) for c, v in zip(centers, sm)}}


# ---------------- 读取（不读 A02） ----------------
x_tgt, sr = load(TARGET_ID)          # A01 薄音
x_ref, sr_ref = load(REF_ID)         # A06 正常（跨歌曲参考）
x_cross_chk, sr_chk = load(CROSS_CHECK_ID)   # A08 正常（仅只读交叉验证）

P_tgt, f0_tgt = harmonic_profile(x_tgt, sr)
P_ref, f0_ref = harmonic_profile(x_ref, sr_ref)
P_chk, f0_chk = harmonic_profile(x_cross_chk, sr_chk)

# ---------------- 可靠性检查：A06 是否具备"正常演唱"的代表性 ----------------
ks = sorted(set(P_ref) & set(P_chk))
diff_ref_chk = {k: round(P_ref[k] - P_chk[k], 2) for k in ks}
same_sign_ratio = float(np.mean([1.0 if (P_ref[k] - P_chk[k]) > 0 else 0.0 for k in ks]))
ref_chk_corr = float(np.corrcoef([P_ref[k] for k in ks], [P_chk[k] for k in ks])[0, 1])

pitch_gap_hz = f0_ref - f0_tgt
pitch_gap_semitone = 12.0 * np.log2(f0_ref / f0_tgt)
harmonic_coverage = {
    "target_f0_hz": round(f0_tgt, 1), "reference_f0_hz": round(f0_ref, 1),
    "pitch_gap_hz": round(pitch_gap_hz, 1), "pitch_gap_semitone": round(pitch_gap_semitone, 2),
    "target_max_harmonic_hz": round(min(K_MAX * f0_tgt, BAND_EDGES), 0),
    "reference_max_harmonic_hz": round(min(K_MAX * f0_ref, BAND_EDGES), 0),
}

# ---------------- 跨歌曲 X1+M2 ----------------
y_x1, x1_stats = apply_harmonic_profile(x_tgt, sr, P_ref, P_tgt)
y_x1_m2, m2_stats = apply_envelope(y_x1, sr, x_ref, sr_ref)

# ---------------- 同曲最佳版（引用 EXP-016，不重算） ----------------
if not SAME_SONG_BEST.exists():
    raise FileNotFoundError("EXP-016 M2.wav missing")
exp016 = json.loads(EXP016_QA.read_text(encoding="utf-8"))
exp016_m2_sha = next(v for v in exp016["versions"] if v["id"] == "M2")["raw_sha256"]
if sha(SAME_SONG_BEST) != exp016_m2_sha:
    raise ValueError("EXP-016 M2.wav changed; same-song reference invalid")
y_same, sr_same = read_pcm16(SAME_SONG_BEST)

# ---------------- 客观检查 ----------------
raw_root = OUT / "outputs/processed_raw"
versions = [("cross_song", y_x1_m2)]
records = []
for vid, y in versions:
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    prof, _ = harmonic_profile(y, sr)
    cen = spectral_centroid(y, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    lag = lag_between(y, x_tgt, sr)
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768))
    hard, flags = [], []
    if abs(pitch - float(np.nanmedian(estimate_pitch(x_tgt, sr)))) > 2.0:
        hard.append(f"音高异常 Δ={pitch - float(np.nanmedian(estimate_pitch(x_tgt, sr))):.2f}Hz")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    if clipped > 0:
        hard.append(f"削波 {clipped} samples")
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, 0.0, rp)
    records.append({
        "id": vid, "proc": {"x1": x1_stats, "m2": m2_stats},
        "h1h2": round(prof[1] - prof[2], 2), "rich_2_23": round(part_richness(prof, 2, K_MAX), 2),
        "centroid_hz": round(cen, 0), "pitch_median_hz": round(pitch, 1), "lag_samples": lag,
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "raw_lufs": round(loud["integrated_lufs"], 2), "raw_true_peak_dbtp": round(loud["true_peak_dbtp"], 2),
        "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "clipped": clipped, "hard_fail": hard, "flags": flags, "deliverable": not hard,
    })

# 同曲最佳版的客观指标（仅记录，用于差距对比）
same_prof, _ = harmonic_profile(y_same, sr_same)
same_stats = sample_stats(y_same, sr_same)
same_metrics = {
    "h1h2": round(same_prof[1] - same_prof[2], 2),
    "rich_2_23": round(part_richness(same_prof, 2, K_MAX), 2),
    "centroid_hz": round(spectral_centroid(y_same, sr_same), 0),
    "lufs": round(measure_loudness(y_same, sr_same)["integrated_lufs"], 2),
    "sha256": exp016_m2_sha,
}

# ---------------- 等响三版 ----------------
loud_tgt = measure_loudness(x_tgt, sr)
loud_cross = measure_loudness(y_x1_m2, sr)
loud_same = measure_loudness(y_same, sr_same)
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"]
                           for m in (loud_tgt, loud_cross, loud_same)])

listen_root = OUT / "listening/audio"
exports = []
for tag, data_, rate, loud in (("original", x_tgt, sr, loud_tgt),
                               ("same_song", y_same, sr_same, loud_same),
                               ("cross_song", y_x1_m2, sr, loud_cross)):
    dest = listen_root / f"{tag}.wav"
    export_wav(data_, rate, target_lufs - loud["integrated_lufs"], dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{tag} loudness match failed: {dm['integrated_lufs']}")
    if dm["true_peak_dbtp"] > -2.9:
        raise RuntimeError(f"{tag} true peak too hot: {dm['true_peak_dbtp']}")
    exports.append({"tag": tag, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                    "lufs": dm["integrated_lufs"], "true_peak_dbtp": dm["true_peak_dbtp"],
                    "frames": len(de), "sha256": sha(dest)})

# 跨歌曲版也做等响导出记录
for r in records:
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-019")
    ym, ysr = read_pcm16(src)
    m = measure_loudness(ym, ysr)
    r["matched_gain_db"] = round(target_lufs - m["integrated_lufs"], 2)

# ---------------- 试听页 ----------------
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P5 跨歌曲参考（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:230px 1fr;gap:10px;align-items:center;margin:10px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}</style>
<h1>F1-P5 · 同歌手跨歌曲参考迁移（A01）</h1>
<div class="key"><b>本轮问题：用「同一歌手的另一首歌的正常演唱」作参考，能不能美化这首单薄版？</b><br>
目标：A01 平凡之路·薄音版。参考：A06 雅俗共赏·<b>正常版</b>（另一首歌）。<br>
<b>A01 自己的同曲正常版未参与本轮任何参数拟合</b>，只作为下方"同曲参考最佳版"的历史对照。</div>
<div class="note"><b>参考音区差异：</b>参考 {f0_ref:.1f}Hz vs 目标 {f0_tgt:.1f}Hz（差 {pitch_gap_hz:+.1f}Hz ≈ {pitch_gap_semitone:+.2f} 半音），音区接近，谐波覆盖到约 {harmonic_coverage['target_max_harmonic_hz']:.0f}Hz。<br>
<b>请重点评价：</b>①跨歌曲版是否更饱满？②是否仍然自然、像本人？③与"同曲参考最佳版"的差距有多大？</div>
<div class="card"><h3>三版对照（正常标注，非盲听）</h3>
<div class="row"><label>① 原声（A01 薄音版）</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>② 同曲参考 X1+M2（历史最佳）</label><audio controls preload="metadata" src="audio/same_song.wav"></audio></div>
<div class="row"><label>③ 跨歌曲参考 X1+M2（本轮新增）</label><audio controls preload="metadata" src="audio/cross_song.wav"></audio></div></div>
<p class="meta">三版统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。②是上一轮你认可的版本，作为差距基准，不是本轮产物。Holdout 未处理；未新增 DSP 模块；未改音高/节奏/咬字。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA ----------------
qa = {
    "experiment": "F1-EXP-019", "phase": "F1-P5", "status": "pending_jade",
    "question": "同歌手跨歌曲参考（另一首歌的正常版）能否美化目标歌曲的单薄版本",
    "target": {"id": TARGET_ID, "display": items[TARGET_ID].get("song", "") + " / " + items[TARGET_ID].get("variant", ""),
               "f0_hz": round(f0_tgt, 1)},
    "reference": {"id": REF_ID, "display": items[REF_ID].get("song", "") + " / " + items[REF_ID].get("variant", ""),
                  "f0_hz": round(f0_ref, 1), "note": "同一歌手的另一首歌，正常演唱"},
    "excluded_from_fitting": {"id": EXCLUDED_ID, "reason": "A01 的同曲正常版，本轮禁止用于任何参数拟合或处理"},
    "same_song_best": {"source": "F1-EXP-016 M2.wav（历史产物，未重算）", **same_metrics},
    "cross_check": {
        "id": CROSS_CHECK_ID, "f0_hz": round(f0_chk, 1),
        "purpose": "只读交叉验证：判断 A06 的谐波剖面是否具备'正常演唱'的代表性",
        "profile_correlation_ref_vs_check": round(ref_chk_corr, 3),
        "ref_higher_than_check_ratio": round(same_sign_ratio, 3),
        "per_harmonic_diff_db": diff_ref_chk,
    },
    "pitch_and_coverage": harmonic_coverage,
    "profiles": {
        "target": {k: round(v, 2) for k, v in P_tgt.items()},
        "reference": {k: round(v, 2) for k, v in P_ref.items()},
    },
    "gain_curve_X1_cross": x1_stats["gain_curve"],
    "curve_M2_cross": m2_stats["curve"],
    "target_lufs": target_lufs, "exports": exports,
    "versions": records, "sealed_not_processed": SEALED,
    "no_new_modules": True, "no_pitch_timing_articulation_change": True,
    "limitations": [
        "参考与目标是不同歌曲：元音/段落/音区分布不同，剖面对齐可能把'元音差异'误当'薄→正常差异'",
        "同曲参考方案（②）使用了 A02，与本轮方案条件不同，差距不能直接归因于'参考歌不同'",
        "A06 曾出现在早期 sealed 编号列表（A04/A06 未绑定 dataset_id），本轮仅作参考、不处理不评估",
        "自实现机制未与外部工具校验；Agent 未听音，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "h1h2", "rich_2_23", "centroid_hz", "pitch", "lag", "peak_dbfs", "crest_db",
                "raw_lufs", "raw_tp", "clipped", "matched_lufs", "matched_tp", "hard", "flags"])
    w.writerow(["original(A01)", round(P_tgt[1] - P_tgt[2], 2), round(part_richness(P_tgt, 2, K_MAX), 2),
                round(spectral_centroid(x_tgt, sr), 0), round(float(np.nanmedian(estimate_pitch(x_tgt, sr))), 1),
                0, round(sample_stats(x_tgt, sr)["sample_peak_dbfs"], 2),
                round(sample_stats(x_tgt, sr)["sample_peak_dbfs"] - sample_stats(x_tgt, sr)["rms_dbfs"], 2),
                round(loud_tgt["integrated_lufs"], 2), round(loud_tgt["true_peak_dbtp"], 2), 0,
                next(e["lufs"] for e in exports if e["tag"] == "original"),
                next(e["true_peak_dbtp"] for e in exports if e["tag"] == "original"), "", ""])
    w.writerow(["same_song_best", same_metrics["h1h2"], same_metrics["rich_2_23"], same_metrics["centroid_hz"],
                "", "", round(same_stats["sample_peak_dbfs"], 2),
                round(same_stats["sample_peak_dbfs"] - same_stats["rms_dbfs"], 2),
                same_metrics["lufs"], "", "", next(e["lufs"] for e in exports if e["tag"] == "same_song"),
                next(e["true_peak_dbtp"] for e in exports if e["tag"] == "same_song"), "", ""])
    for r in records:
        w.writerow([f"cross_song({r['id']})", r["h1h2"], r["rich_2_23"], r["centroid_hz"], r["pitch_median_hz"],
                    r["lag_samples"], r["peak_dbfs"], r["crest_db"], r["raw_lufs"], r["raw_true_peak_dbtp"],
                    r["clipped"], next(e["lufs"] for e in exports if e["tag"] == "cross_song"),
                    next(e["true_peak_dbtp"] for e in exports if e["tag"] == "cross_song"),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"])])

params = {
    "experiment": "F1-EXP-019", "phase": "F1-P5",
    "question": "同歌手跨歌曲参考能否美化目标单薄版",
    "target": TARGET_ID, "reference": REF_ID, "excluded_from_fitting": EXCLUDED_ID,
    "same_song_best_reused": "F1-EXP-016 M2.wav（未重算，仅对照）",
    "mechanism_unchanged": "X1 谐波剖面整段增益 + M2 包络整形（与 EXP-015/016 相同）",
    "no_new_modules": True, "sealed": SEALED,
    "listening_target_lufs": target_lufs,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-019", "phase": "F1-P5", "status": "pending_jade",
    "target": TARGET_ID, "reference": REF_ID,
    "cross_song_runs": all(r["deliverable"] for r in records),
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(e["lufs"] - target_lufs) <= 0.3 for e in exports),
    "preview_page": "listening/index.html",
    "deliverables": ["original.wav", "same_song.wav", "cross_song.wav"],
    "next_rule": "若跨歌曲版明显劣于同曲版 → 个性化参考路线成立；若接近或更好 → 跨歌曲可用",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-019\nphase: F1-P5\n"
    f"cross-song reference transfer: target {TARGET_ID}, reference {REF_ID} (different song, same singer)\n"
    f"A02 excluded from all fitting; same-song best reused from EXP-016 without recompute\n"
    f"sealed untouched: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "pitch_and_coverage": harmonic_coverage,
    "cross_check_ref_vs_A08": {"correlation": round(ref_chk_corr, 3), "ref_higher_ratio": round(same_sign_ratio, 3)},
    "original": {"h1h2": round(P_tgt[1] - P_tgt[2], 2), "rich": round(part_richness(P_tgt, 2, K_MAX), 2),
                 "centroid": round(spectral_centroid(x_tgt, sr), 0)},
    "same_song_best": same_metrics,
    "cross_song": {k: records[0][k] for k in ("h1h2", "rich_2_23", "centroid_hz", "pitch_median_hz",
                                               "lag_samples", "clipped", "hard_fail", "flags")},
}, ensure_ascii=False, indent=1))
