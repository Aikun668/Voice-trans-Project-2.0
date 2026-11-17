"""F1-EXP-017 / F1-P4: X1 + M2 无配对参考迁移验证。

问题：不提供目标歌手的正常演唱参考，现有方法还能否改善单薄感？

依赖现状：
  EXP-015 X1 增益表 = A01 谐波剖面 vs 同曲正常版 A02 的差值 → 逐例参考定制
  EXP-016 M2 包络曲线 = X1 vs A02 的剩余包络差            → 逐例参考定制
  机制（F0 锁定谐波掩膜 / FFT 零相位包络整形）本身通用。

本轮做法：
  1) 从 4 对自录 Dev 数据提取**固定模板**（不再逐例定制）：
     T_X1(f)：每对 (正常-薄音) 谐波增益，按 f=k·F0 映射到绝对频率，4 对取中位；
     T_M2(f)：每对 (正常 - 该对经 T_X1 后) 的剩余包络差，4 对取中位。
  2) 把 T_X1 / T_M2 原样应用到 A05（GTSinger 女声 女A-002，F0≈290Hz），
     不读取 A05 的任何参考、不按 A05 结果重拟合。
  3) 输出 Original / X1 / X1+M2 三版，等响盲听。

跨基频适配（必须记录的决策）：
  自录 F0 中值 137-226Hz（模板定义域），A05 F0≈290Hz。
  模板按**绝对频率**索引而非谐波序数——因为"厚度"感知对应绝对频段
  （250-500Hz body 区等），跨基频时才物理一致。
  已知覆盖盲区：A05 在 250-500Hz 只有 H1，该区无法靠谐波选择性增密；
  如实记录，不用其他模块补救（本轮禁止新增模块）。
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

OUT = ROOT / "experiments/F1-EXP-017"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-017 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

DEV_PAIRS = [
    ("平凡之路", "A_self_01", "A_self_02"),
    ("我们的歌", "A_self_03", "A_self_04"),
    ("雅俗共赏", "A_self_05", "A_self_06"),
    ("因为刚好遇见你", "A_self_07", "A_self_08"),
]
TARGET_SAMPLE = ("A05", "A_gt_02", "GTSinger 女声 女A-002（来源未保留）")
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]

K_MAX_SELF = 23
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
TEMPLATE_GRID_N = 96
SEED = 20261106


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


def harmonic_profile(x, sr, kmax=K_MAX_SELF):
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


def band_levels(x, sr, lo=1000.0, hi=8000.0, step=250.0):
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


def apply_harmonic_template(x, sr, template_f, template_db):
    """X1 机制（与 EXP-015 相同）：STFT 逐帧 F0 定位谐波，施加几何形掩膜增益。
    唯一差异：增益值由固定模板按**绝对频率**查询，而非逐例差值。"""
    N, H = STFT_N, STFT_H
    w = np.hanning(N)
    bin_hz = sr / N
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    nb = N // 2 + 1

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
            g_db = float(np.interp(fc, template_f, template_db, left=0.0, right=0.0))
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
    return y, {"frames_total": int(n_frames), "frames_processed": int(n_applied),
               "process_fraction": round(n_applied / n_frames, 3)}


def apply_envelope_template(x, sr, active_f, active_db):
    """M2 机制（与 EXP-016 相同）：全段 FFT 零相位包络整形；作用区外增益 0。"""
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.interp(f, active_f, active_db, left=0.0, right=0.0)
    return np.fft.irfft(np.fft.rfft(x) * 10.0 ** (g / 20.0), n)


# ---------------- 阶段 1：模板提取 ----------------
thin_profiles, normal_profiles = {}, {}
pair_rows = []
for song, tid, nid in DEV_PAIRS:
    xt_, sr_t = load(tid)
    xn_, sr_n = load(nid)
    Pt, f0t = harmonic_profile(xt_, sr_t)
    Pn, f0n = harmonic_profile(xn_, sr_n)
    thin_profiles[tid] = Pt
    normal_profiles[nid] = Pn
    gain_by_k = {k: Pn[k] - Pt[k] for k in Pt if k in Pn}
    pair_rows.append({"song": song, "thin": tid, "normal": nid,
                      "f0_thin": round(f0t, 1), "f0_normal": round(f0n, 1),
                      "gain_by_k": {k: round(v, 2) for k, v in gain_by_k.items()}})
    pair_rows[-1]["_fk"] = np.array([k * f0t for k in sorted(gain_by_k)], dtype=float)
    pair_rows[-1]["_gk"] = np.clip(np.array([gain_by_k[k] for k in sorted(gain_by_k)], dtype=float),
                                   -CAP_DOWN_DB, CAP_UP_DB)

grid = np.geomspace(100.0, 8000.0, TEMPLATE_GRID_N)
per_pair_curves = []
for row in pair_rows:
    cur = np.interp(grid, row["_fk"], row["_gk"], left=0.0, right=0.0)
    per_pair_curves.append(cur)

stack = np.vstack(per_pair_curves)
tpl_x1_raw = np.median(stack, axis=0)
tpl_x1_spread = np.std(stack, axis=0)
sm = np.copy(tpl_x1_raw)
for i in range(len(tpl_x1_raw)):
    lo, hi = max(0, i - 1), min(len(tpl_x1_raw), i + 2)
    sm[i] = float(np.mean(tpl_x1_raw[lo:hi]))
tpl_x1 = np.clip(sm, -CAP_DOWN_DB, CAP_UP_DB)
tpl_x1[0] = 0.0

# ---------------- 阶段 2：Dev 上生成 X1（用于 M2 模板）+ 模板质量验证 ----------------
dev_x1, dev_check = {}, []
for song, tid, nid in DEV_PAIRS:
    xt_, sr_t = load(tid)
    y_x1, pstat = apply_harmonic_template(xt_, sr_t, grid, tpl_x1)
    dev_x1[tid] = (y_x1, sr_t)
    Pb = thin_profiles[tid]
    Pa, _ = harmonic_profile(y_x1, sr_t)
    Pg = normal_profiles[nid]
    dev_check.append({
        "song": song, "thin": tid,
        "h1h2_before": round(Pb[1] - Pb[2], 2), "h1h2_after": round(Pa[1] - Pa[2], 2),
        "h1h2_target": round(Pg[1] - Pg[2], 2),
        "rich_before": round(part_richness(Pb, 2, K_MAX_SELF), 2),
        "rich_after": round(part_richness(Pa, 2, K_MAX_SELF), 2),
        "rich_target": round(part_richness(Pg, 2, K_MAX_SELF), 2),
        "process_fraction": pstat["process_fraction"],
    })

env_curves = []
for song, tid, nid in DEV_PAIRS:
    y_x1, sr_t = dev_x1[tid]
    xn_, sr_n = load(nid)
    lx = measure_loudness(y_x1, sr_t)["integrated_lufs"]
    ln = measure_loudness(xn_, sr_n)["integrated_lufs"]
    bx = band_levels(y_x1, sr_t)
    bn = band_levels(xn_, sr_n)
    keys = sorted(bx.keys())
    gaps = np.array([(bn[k] - ln) - (bx[k] - lx) for k in keys])
    env_curves.append(gaps)
    centers = [(a + b) / 2 for (a, b) in keys]

tpl_m2 = np.median(np.vstack(env_curves), axis=0)
sm2 = np.copy(tpl_m2)
for i in range(len(tpl_m2)):
    lo, hi = max(0, i - 1), min(len(tpl_m2), i + 2)
    sm2[i] = float(np.mean(tpl_m2[lo:hi]))
tpl_m2 = np.clip(sm2, MORPH_CLAMP[0], MORPH_CLAMP[1])
env_f = [0.0, MORPH_LO] + centers + [MORPH_HI + MORPH_STEP]
env_db = [0.0, 0.0] + list(tpl_m2) + [0.0]

# ---------------- 阶段 3：应用到 A05 ----------------
sid, sdid, sdisplay = TARGET_SAMPLE
x_tg, sr_tg = load(sdid)
tg_loud = measure_loudness(x_tg, sr_tg)
tg_stats = sample_stats(x_tg, sr_tg)
tg_prof, tg_f0 = harmonic_profile(x_tg, sr_tg)
tg_centroid = spectral_centroid(x_tg, sr_tg)
tg_pitch = float(np.nanmedian(estimate_pitch(x_tg, sr_tg)))

y_x1_tg, tg_pstat = apply_harmonic_template(x_tg, sr_tg, grid, tpl_x1)
y_m2_tg = apply_envelope_template(y_x1_tg, sr_tg, env_f, env_db)

covered = []
for k in range(1, int(BAND_EDGES // tg_f0) + 1):
    fc = k * tg_f0
    if fc >= BAND_EDGES:
        break
    covered.append({"k": k, "freq_hz": round(fc, 1),
                    "gain_db": round(float(np.interp(fc, grid, tpl_x1, left=0.0, right=0.0)), 2)})

raw_root = OUT / "outputs/processed_raw"
records = []
for vid, y in [("X1", y_x1_tg), ("X1_M2", y_m2_tg)]:
    st = sample_stats(y, sr_tg)
    loud = measure_loudness(y, sr_tg)
    prof, _ = harmonic_profile(y, sr_tg)
    cen = spectral_centroid(y, sr_tg)
    pitch = float(np.nanmedian(estimate_pitch(y, sr_tg)))
    lag = lag_between(y, x_tg, sr_tg)
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0
    hard, flags = [], []
    if abs(pitch - tg_pitch) > 2.0:
        hard.append(f"音高异常 Δ={pitch - tg_pitch:.2f}Hz")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    if clipped > 0:
        hard.append(f"削波 {clipped} samples")
    if st["sample_peak_dbfs"] > -0.5:
        flags.append(f"峰值偏高 {st['sample_peak_dbfs']:.2f}dBFS")
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr_tg, 0.0, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
        hard.append("导出削波")
    records.append({
        "id": vid, "process_fraction": tg_pstat["process_fraction"] if vid == "X1" else None,
        "h1h2": round(prof[1] - prof[2], 2) if 1 in prof and 2 in prof else None,
        "rich_2_23": round(part_richness(prof, 2, K_MAX_SELF), 2),
        "centroid_hz": round(cen, 0),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2),
        "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "pitch_median_hz": round(pitch, 1), "lag_samples": lag,
        "raw_lufs": round(ye_loud["integrated_lufs"], 2),
        "raw_true_peak_dbtp": round(ye_loud["true_peak_dbtp"], 2),
        "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "clipped": clipped, "hard_fail": hard, "flags": flags, "deliverable": not hard,
    })

# ---------------- 等响 + 盲听 ----------------
loud_all = {"original": tg_loud, **{r["id"]: {"integrated_lufs": r["raw_lufs"],
                                              "true_peak_dbtp": r["raw_true_peak_dbtp"]} for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loud_all.values()])

listen_root = OUT / "listening/audio"
export_wav(x_tg, sr_tg, target_lufs - tg_loud["integrated_lufs"], listen_root / "original.wav")
om, osr = read_pcm16(listen_root / "original.wav")
if abs(measure_loudness(om, osr)["integrated_lufs"] - target_lufs) > 0.3:
    raise RuntimeError("original loudness match failed")

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-017")
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
letters = ["A", "B"][:len(deliverables)]
blind = {letter: vid for letter, vid in zip(letters, rng.sample(deliverables, len(deliverables)))}
for letter, vid in blind.items():
    shutil.copyfile(listen_root / f"{vid}.wav", listen_root / f"{letter}.wav")
    (listen_root / f"{vid}.wav").unlink()

rows = []
for letter in blind.keys():
    r = next(v for v in records if v["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["flags"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')


def esc_(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4 无参考迁移（{sid}）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:880px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P4 · 无配对参考迁移（{sid} · {esc_(sdisplay)}）</h1>
<div class="key"><b>本轮只验证一件事：不给这位歌手的正常版参考，方法还能不能改善单薄感。</b><br>
模板来自你自己的 4 对自录录音（固定，不再按素材重拟合），原样套用到这条 GTSinger 女声。
<b>没有读取这条素材的任何参考频谱，也没有根据它的结果调参。</b></div>
<div class="note"><b>先说一个已知限制：</b>这条素材音高约 {tg_f0:.0f}Hz，比你的自录（137–226Hz）高。模板在 250–500Hz"厚度区"只能作用到基频那一根，那一段无法靠本方法增密——这是方法的覆盖盲区，如实说明，没有用其他模块补救。<br>
<b>请判断：</b>①有无变化；②饱满度/清晰度是否改善；③是否自然、像本人（音色保真）；④有无明显副作用（金属感、嘶嘶声、齿音、闷、空洞、像别人）。</div>
<div class="card"><h3>试听（A/B 匿名）</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）；原素材未改动；Holdout（A04/A06/B03/C01–C03）未处理。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA 与记录 ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

dep = {
    "X1": {
        "gain_table_source": "A01 谐波剖面 vs 同曲正常版 A02 的逐谐波差值（EXP-015）",
        "reference_customized": True,
        "mechanism_generic": True,
        "detail": "增益值是逐例差值，属参考定制；STFT 谐波定位/掩膜/回退策略是通用机制",
    },
    "M2": {
        "envelope_curve_source": "X1 vs A02 的剩余包络差（EXP-016）",
        "reference_customized": True,
        "mechanism_generic": True,
        "detail": "曲线是逐例差值，属参考定制；FFT 零相位包络整形是通用机制",
    },
    "this_round": "两处逐例定制均替换为 4 对 Dev 中位模板，按绝对频率索引；A05 未读取任何参考",
}
qa = {
    "experiment": "F1-EXP-017", "phase": "F1-P4", "status": "pending_jade",
    "question": "不提供目标歌手的正常演唱参考，现有方法还能否改善单薄感",
    "reference_dependency": dep,
    "template_extraction": {
        "source": "4 对自录 Dev（A01/02、A03/04、A05/06、A07/08），仅用谐波剖面对比，未用听感结果",
        "indexing": "按绝对频率（f=k·F0）而非谐波序数；理由：'厚度'感知对应绝对频段",
        "T_X1_grid_hz": [float(grid[0]), float(grid[-1])],
        "T_X1_points": TEMPLATE_GRID_N,
        "T_X1_db": {f"{float(g):.0f}Hz": round(float(v), 2) for g, v in zip(grid, tpl_x1)},
        "T_X1_inter_pair_std_db": {f"{float(g):.0f}Hz": round(float(v), 2) for g, v in zip(grid, tpl_x1_spread)},
        "T_M2_resolution_hz": MORPH_STEP,
        "T_M2_db": {f"{int(c)}Hz": round(float(v), 2) for c, v in zip(centers, tpl_m2)},
        "pairs_gain_by_k": [{k: v for k, v in row.items() if not k.startswith("_")} for row in pair_rows],
    },
    "template_quality_on_dev": dev_check,
    "target_sample": {
        "id": sid, "dataset_id": sdid, "display": sdisplay,
        "f0_median_hz": round(tg_f0, 1), "lufs": tg_loud["integrated_lufs"],
        "peak_dbfs": round(tg_stats["sample_peak_dbfs"], 2),
        "centroid_hz": round(tg_centroid, 0),
        "h1h2": round(tg_prof[1] - tg_prof[2], 2) if 1 in tg_prof and 2 in tg_prof else None,
        "rich_2_23": round(part_richness(tg_prof, 2, K_MAX_SELF), 2),
        "coverage_500hz_only_H1": "250-500Hz 内仅有 H1，该区无法增密（方法盲区）",
    },
    "applied_gains_on_target": covered,
    "target_lufs": target_lufs,
    "versions": records,
    "sealed_not_processed": SEALED,
    "no_new_modules": "未新增饱和/压缩/EQ/谐波生成；仅复用 EXP-015/016 机制",
    "limitations": [
        "模板来自 4 对同年自录，对异歌手（GTSinger、女声、高音区）无适配",
        "A05 F0≈290Hz，250-500Hz body 区只有基频一根，方法无法覆盖该区",
        "未读取 A05 任何参考（该素材本身也无配对正常版），因此无法判断'差多少'，只能判断'有没有改善'",
        "自实现机制未与外部工具校验；Agent 未听音，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "h1h2", "h1h2_original", "rich_2_23", "rich_2_23_original",
                "centroid_hz", "centroid_original", "pitch", "lag", "raw_lufs", "raw_tp",
                "clipped", "hard", "flags", "matched_lufs", "matched_tp"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["h1h2"], round(tg_prof[1] - tg_prof[2], 2),
                    r["rich_2_23"], round(part_richness(tg_prof, 2, K_MAX_SELF), 2),
                    r["centroid_hz"], round(tg_centroid, 0), r["pitch_median_hz"], r["lag_samples"],
                    r["raw_lufs"], r["raw_true_peak_dbtp"], r["clipped"],
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"]),
                    m.get("lufs", ""), m.get("true_peak_dbtp", "")])

params = {
    "experiment": "F1-EXP-017", "phase": "F1-P4",
    "question": "无配对正常版参考时，X1+M2 能否运行并改善单薄感",
    "template_source": "4 对自录 Dev 中位数（固定，不再逐例拟合）",
    "no_new_modules": True, "no_reference_for_target": True,
    "target": {"id": sid, "dataset_id": sdid, "f0_hz": round(tg_f0, 1)},
    "sealed": SEALED,
    "listening_target_lufs": target_lufs,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-017", "phase": "F1-P4", "status": "pending_jade",
    "reference_dependency": {"X1": True, "M2": True, "overridden_by_template": True},
    "can_run_without_paired_reference": True,
    "target": {"id": sid, "f0_hz": round(tg_f0, 1)},
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(r.get("matched", {}).get("lufs", target_lufs + 99) - target_lufs) <= 0.3
                for r in records if r["deliverable"]),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-017\nphase: F1-P4\n"
    f"cross-singer transfer with fixed template from 4 self pairs; target {sid} ({sdid})\n"
    f"no reference for target; no new modules\nsealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    f"# F1-EXP-017 结果（pending_jade）\n\n"
    f"F1-P4 无配对参考迁移：从 4 对自录 Dev 提取固定模板（T_X1 谐波增益按绝对频率、T_M2 剩余包络），"
    f"原样应用到 {sid}（{sdisplay}），不读取目标任何参考、不按目标重拟合、不新增模块。\n\n"
    f"- 目标：{sid}，F0 中值 {tg_f0:.1f}Hz，原始 {tg_loud['integrated_lufs']:.2f} LUFS\n"
    f"- 交付：{', '.join(r['id'] for r in records if r['deliverable'])}（匿名 A/B）+ 原音\n"
    f"- 技术失败：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    f"- 已知盲区：250–500Hz 在该音高下只有 H1，无法增密\n\n"
    "听感结论待 Jade 试听后填写；本轮只生成材料，不做外推结论。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "T_X1_sample": {f"{float(g):.0f}": round(float(v), 2) for g, v in list(zip(grid, tpl_x1))[::8]},
    "T_M2_sample": {f"{int(c)}": round(float(v), 2) for c, v in list(zip(centers, tpl_m2))[::3]},
    "dev_template_quality": dev_check,
    "target_versions": [{k: r[k] for k in ("id", "h1h2", "rich_2_23", "centroid_hz",
                                             "pitch_median_hz", "lag_samples", "clipped",
                                             "hard_fail", "flags")} for r in records],
}, ensure_ascii=False, indent=1))
