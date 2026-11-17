"""F1-EXP-016 / F1-P4C: 频谱包络移植（spectral morphing）——论文方法的包络半部分。

论文：Perrotin & d'Alessandro, "Vocal effort modification for singing synthesis",
     Interspeech 2016, DOI 10.21437/Interspeech.2016-1096（HAL hal-01712564）。
论文要点：增强发声力度 = ①谐波增富（生成原信号没有的新谐波，声源时域整形）
     + ②频谱包络变换（morphing = 移植大声参考的包络；两者结合感知上接近真实大声唱。

本轮（在已认可的 X1 谐波对齐基线上）只做 morphing 这一半（每轮一个机制）：
  机制 = 逐频段测量"正常版 A02 相对 X1 的剩余包络差"（响度归一、250Hz 分辨率），
  曲线经 3 点滑动平均与 [-3,+5]dB 钳位，以 60% / 100% 两档移植到 X1。
  实测缺口：2500-3000Hz +3.8dB、3000-3500 +2.1、5-8k +1.5~+2.6（"芯/力量"区与高区），
  以及 X1 过冲区（1000-1500、3500-4500）的小幅负向回调。
  谐波增富（论文另一半）明确留作下一轮，本轮不实现。
应用方式：全段 FFT 零相位整形（静态频域滤波，无非线性、无限制器）。
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

OUT = ROOT / "experiments/F1-EXP-016"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-016 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; DISPLAY = "平凡之路 / 薄音版"
TARGET_ID = "A_self_02"; TARGET_DISPLAY = "平凡之路 / 正常版"
BASE_EXP015_X1 = ROOT / "experiments/F1-EXP-015/outputs/processed_raw/X1.wav"
EXP015_QA = ROOT / "experiments/F1-EXP-015/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
SEED = 20261105

# --- 机制参数 ---
MORPH_LO, MORPH_HI = 1000.0, 8000.0   # 包络移植作用区
MORPH_STEP = 250.0                     # 测量分辨率
CLAMP_DB = (-3.0, 5.0)                 # 单频段增益钳位（含小幅负向回调）
SMOOTH_WIN = 3                         # 曲线滑动平均窗（≈750Hz）
STRENGTHS = [("M1", 0.6), ("M2", 1.0)]
GATE_MAX_DEVIATION_DB = 1.2            # 实测增量与预期曲线的最大允许偏差
MIN_MORPH_PREMISE_DB = 2.0             # 前提门槛：M2 正向曲线峰值须 ≥2dB，否则缺口不足以支撑本轮


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


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


def band_levels(x, sr, nperseg=16384):
    f, psd = signal.welch(x, sr, nperseg=min(nperseg, len(x)))
    edges = np.arange(MORPH_LO, MORPH_HI + MORPH_STEP / 2, MORPH_STEP)
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (f >= lo) & (f < hi)
        out[(float(lo), float(hi))] = 10.0 * np.log10(max(float(psd[m].sum()), 1e-20))
    return out


def spectral_centroid(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    return float(np.sum(f[m] * p[m]) / max(float(np.sum(p[m])), 1e-20))


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    yseg = y[:len(seg)]
    c = signal.correlate(yseg, seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


def morph_curve(base, target, sr, lufs_ref, lufs_base, lufs_target):
    """剩余包络差曲线：target(正常版) 相对 base(X1)，响度归一到参考。返回 (xs, ys)。"""
    bt = band_levels(base, sr)
    tg = band_levels(target, sr)
    centers, gaps = [], []
    for band in sorted(bt.keys()):
        c = (band[0] + band[1]) / 2.0
        v = (tg[band] + (lufs_ref - lufs_target)) - (bt[band] + (lufs_ref - lufs_base))
        centers.append(c)
        gaps.append(v)
    gaps = np.array(gaps)
    # 3 点滑动平均 + 钳位
    sm = gaps.copy()
    for i in range(len(gaps)):
        lo, hi = max(0, i - SMOOTH_WIN // 2), min(len(gaps), i + SMOOTH_WIN // 2 + 1)
        sm[i] = float(np.mean(gaps[lo:hi]))
    sm = np.clip(sm, CLAMP_DB[0], CLAMP_DB[1])
    xs = [0.0, MORPH_LO] + centers + [MORPH_HI + 250.0]
    ys = [0.0, 0.0] + list(sm) + [0.0]
    return xs, ys, centers, sm, gaps


def apply_morph(x, sr, xs, ys, scale):
    """全段 FFT 零相位整形：静态频域包络移植，无非线性。"""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.interp(f, xs, [v * scale for v in ys], left=0.0, right=ys[-1] * scale)
    y = np.fft.irfft(X * 10.0 ** (g / 20.0), n)
    return y


# ---------------- 输入 ----------------
x_orig, sr = load(DATASET_ID)
x_target, _ = load(TARGET_ID)
if not BASE_EXP015_X1.exists():
    raise FileNotFoundError("EXP-015 X1 raw missing")
exp015 = json.loads(EXP015_QA.read_text(encoding="utf-8"))
x1_expected_sha = next(v for v in exp015["variants"] if v["id"] == "X1")["raw_sha256"]
if sha(BASE_EXP015_X1) != x1_expected_sha:
    raise ValueError("EXP-015 X1 raw changed; base invalid")
x_base, sr_b = read_pcm16(BASE_EXP015_X1)
if sr_b != sr:
    raise RuntimeError("sample rate mismatch between A01 and X1")

loud_orig = measure_loudness(x_orig, sr)
loud_target = measure_loudness(x_target, sr)
loud_base = measure_loudness(x_base, sr)

listening_contract = {
    "原音问题登记": "A01 自评薄音（蔫、轻、嗲）；响度已排除（EXP-013），谐波结构已对齐（EXP-015 X1，已获'更饱满/更清晰'弱正向）",
    "需求（用户明示）": "把已认可的效果从'认真听才可辨'调到'一听就能听出来'",
    "素材与片段": "A01 全长 13.14s，不截取；基线为上轮 X1",
    "本轮机制": "论文（Interspeech 2016 Perrotin）的包络移植半部分：把正常版实测剩余包络差（2.5-3.5kHz '芯'区 +3.8/+2.1dB、5-8kHz +1.5~2.6dB、X1 过冲区小幅回调）移植到 X1，60%/100% 两档",
    "预期可听变化": "更有力量、更实、一听可辨（本轮目标即'可辨'）",
    "否决副作用": "只亮不厚、嘶嘶声/齿音加重（5-8kHz 区注意）、金属感、闷、像别人",
    "预先声明": "若 100% 档仍不明显 → 下一轮做论文另一半（谐波增富/新谐波生成），不继续加大 EQ；若明显且好 → A05 复验",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-016", "phase": "F1-P4C",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "display": DISPLAY,
    "base": {"file": str(BASE_EXP015_X1.relative_to(ROOT)).replace("\\", "/"),
             "sha256": x1_expected_sha, "note": "EXP-015 X1（上轮已认可的谐波对齐基线），未改动"},
    "target_reference": {"dataset_id": TARGET_ID, "display": TARGET_DISPLAY,
                         "use": "只读包络目标，不处理不交付"},
    "sealed_not_processed": SEALED,
    "frames": len(x_orig), "sample_rate": sr,
    "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 曲线 ----------------
xs, ys, centers, curve, raw_gaps = morph_curve(x_base, x_target, sr,
                                              loud_orig["integrated_lufs"],
                                              loud_base["integrated_lufs"],
                                              loud_target["integrated_lufs"])
if max(ys) < MIN_MORPH_PREMISE_DB:
    raise RuntimeError(f"剩余包络缺口峰值仅 {max(ys):.2f}dB，不足以支撑本轮假设，需重新评估")

base_centroid = spectral_centroid(x_base, sr)
orig_centroid = spectral_centroid(x_orig, sr)
target_centroid = spectral_centroid(x_target, sr)
base_pitch = float(np.nanmedian(estimate_pitch(x_base, sr)))
base_stats = sample_stats(x_base, sr)
base_bands = band_levels(x_base, sr)
# 预期增量的能量加权基准：曲线在频段内有斜率时，必须按基线频谱加权，不能取中心点
f_w, psd_base = signal.welch(x_base, sr, nperseg=min(16384, len(x_base)))

# ---------------- 处理两档 ----------------
raw_root = OUT / "outputs/processed_raw"
records = []
for vid, scale in STRENGTHS:
    y = apply_morph(x_base, sr, xs, ys, scale)
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    y_bands = band_levels(y, sr)
    # 自检：实测增量 vs 能量加权预期（曲线路径按基线频谱能量加权；修正原因：
    # 曲线在频段内斜率不为零时，中心点预期与能量加权实测存在系统性偏差）
    devs, exp_devs = {}, {}
    hard, flags = [], []
    for i, band in enumerate(sorted(y_bands.keys())):
        lo, hi = band
        m = (f_w >= lo) & (f_w < hi)
        g_c = np.interp(f_w[m], xs, [v * scale for v in ys])
        expected = 10.0 * np.log10(float(np.sum(psd_base[m] * 10.0 ** (g_c / 10.0)))
                                   / max(float(np.sum(psd_base[m])), 1e-20))
        achieved = y_bands[band] - base_bands[band]
        devs[f"{int(lo)}_{int(hi)}"] = round(achieved, 2)
        exp_devs[f"{int(lo)}_{int(hi)}"] = round(expected, 2)
        if abs(achieved - expected) > GATE_MAX_DEVIATION_DB:
            hard.append(f"{int(lo)}-{int(hi)}Hz 增量偏差 {achieved - expected:+.2f}dB")
    centroid = spectral_centroid(y, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    lag = lag_between(y, x_base, sr)
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768)) if st["sample_peak_dbfs"] >= -0.01 else 0
    if abs(pitch - base_pitch) > 2.0:
        hard.append(f"音高异常 Δ={pitch - base_pitch:.2f}Hz")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    if clipped > 0:
        hard.append(f"削波 {clipped} samples")
    max_up = float(max(ys)) * scale
    max_down = float(min(ys)) * scale
    if scale >= 1.0 and max_up >= CLAMP_DB[1] - 0.1:
        flags.append(f"正向曲线触顶 +{max_up:.1f}dB，注意亮/刺感")
    if max_down <= -2.0:
        flags.append(f"负向回调最深 {max_down:.1f}dB，注意中频变薄")

    guard = min(0.0, -0.5 - st["sample_peak_dbfs"])
    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, guard, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    if int(np.count_nonzero(np.abs(ye) >= 32767 / 32768)) > 0:
        hard.append("导出削波")

    records.append({
        "id": vid, "scale": scale,
        "expected_curve_center_db": {f"{int(c)}Hz": round(curve[i] * scale, 2) for i, c in enumerate(centers)},
        "expected_band_gain_db": exp_devs,
        "achieved_band_gain_db": devs,
        "centroid_hz": round(centroid, 0), "base_centroid_hz": round(base_centroid, 0),
        "target_centroid_hz": round(target_centroid, 0),
        "max_up_db": round(max_up, 2), "max_down_db": round(max_down, 2),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2),
        "lag_samples": lag, "pitch_median_hz": round(pitch, 1),
        "raw_guard_db": guard, "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha(rp), "raw_lufs": ye_loud["integrated_lufs"],
        "raw_true_peak_dbtp": ye_loud["true_peak_dbtp"],
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    })

# ---------------- 统一响度 + 盲听 ----------------
loud_all = {"original": loud_orig, "base_X1": loud_base,
            **{r["id"]: {"integrated_lufs": r["raw_lufs"], "true_peak_dbtp": r["raw_true_peak_dbtp"]} for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loud_all.values()])

listen_root = OUT / "listening/audio"
exports = []
for tag, data_, rate, loud in (("original", x_orig, sr, loud_orig),
                               ("prev_best", x_base, sr, loud_base)):
    dest = listen_root / f"{tag}.wav"
    export_wav(data_, rate, target_lufs - loud["integrated_lufs"], dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{tag} loudness match failed")
    exports.append({"tag": tag, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                    "lufs": dm["integrated_lufs"], "true_peak_dbtp": dm["true_peak_dbtp"], "sha256": sha(dest)})

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-016")
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
    r = next(v for v in records if v["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P4C 包络移植盲听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:880px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P4C · 包络移植（仅 A01）</h1>
<div class="key"><b>本轮目标：把效果调到"一听就能听出来"。</b><br>
机制 = 论文（Interspeech 2016，Perrotin）的<b>频谱包络移植</b>半部分：把你的正常版实测的剩余包络差搬过来——
重点是 2.5–3.5kHz"芯/力量"区（+2~4dB）与 5–8kHz（+1.5~2.5dB），以及把上轮过冲的 1–1.5kHz 小幅收回。<br>
两档强度已隐藏。基线是上轮已认可的"上轮最佳版"（ harmonic 对齐版），A/B 都是在它之上加的。</div>
<div class="note"><b>本轮登记：</b>预期 = 更有力量、更实，一听可辨（不再需要认真对比）。<br>
否决副作用 = 只亮不厚、嘶嘶声/齿音加重（注意 s/是 等辅音）、金属感、闷、像别人。<br>
<b>请对照"上轮最佳版"听 A/B：差别是否一听可辨？哪个更好？</b></div>
<div class="card"><h3>试听（A/B 匿名）</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>上轮最佳版（基线）</label><audio controls preload="metadata" src="audio/prev_best.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。
若 100% 档仍不满足"一听可辨"，下一步是论文另一半（生成新谐波），不是继续加大 EQ。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA 与记录 ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-016", "phase": "F1-P4C", "status": "pending_jade",
    "paper": {
        "title": "Vocal effort modification for singing synthesis",
        "authors": "Perrotin & d'Alessandro, LIMSI/CNRS",
        "venue": "Interspeech 2016, pp.1235-1239",
        "doi": "10.21437/Interspeech.2016-1096",
        "url": "https://hal.science/hal-01712564",
        "adopted": "spectral morphing（包络移植）半部分：以同曲正常版为包络参考，按剩余差移植",
        "not_adopted_yet": "谐波增富（声源时域整形生成新谐波）——论文指出与 morphing 结合才接近真实大声唱，留作下一轮",
        "supporting": "arXiv:2203.10637（neural TTS 中用 spectral tilt 度量/控制 vocal effort，佐证包络/倾斜是 effort 主线索）",
    },
    "morph_curve": {
        "resolution_hz": MORPH_STEP, "range_hz": [MORPH_LO, MORPH_HI],
        "clamp_db": list(CLAMP_DB), "smooth_window": SMOOTH_WIN,
        "raw_gaps_db": {f"{int(c)}Hz": round(raw_gaps[i], 2) for i, c in enumerate(centers)},
        "smoothed_curve_db": {f"{int(c)}Hz": round(curve[i], 2) for i, c in enumerate(centers)},
        "anchor_points": [[round(a, 1), round(b, 2)] for a, b in zip(xs, ys)],
        "premise_check": f"M2 正向峰值 {max(ys):.2f}dB ≥ {MIN_MORPH_PREMISE_DB}dB",
    },
    "diagnosis_summary": {
        "centroid_hypothesis": "已排除：X1 重心 404.7 ≈ 正常版 404.0；X1/X2 重心差 43Hz 用户不可分辨",
        "remaining_gap": "2500-3000Hz +3.8dB、3000-3500 +2.1、5-8k +1.5~+2.6（宽带/包络层）；X1 过冲 1000-1500 −2.3、3500-4500 −1.3~−2.3",
        "conclusion": "剩余感知缺口在包络层（论文 morphing 的目标），不在谐波层与重心",
    },
    "base": {"id": "EXP-015 X1", "sha256": x1_expected_sha,
             "lufs": loud_base["integrated_lufs"], "centroid_hz": round(base_centroid, 1)},
    "original": {"lufs": loud_orig["integrated_lufs"], "centroid_hz": round(orig_centroid, 1)},
    "target_normal": {"lufs": loud_target["integrated_lufs"], "centroid_hz": round(target_centroid, 1)},
    "target_lufs": target_lufs, "exports": exports,
    "versions": records,
    "gate": {"band_deviation_max_db": GATE_MAX_DEVIATION_DB,
             "pitch_delta_max_hz": 2.0, "lag_max_samples": 1,
             "loudness_tolerance_lu": 0.3, "true_peak_max_dbtp": -2.9},
    "limitations": [
        "包络参考是 A02 单曲（同曲同段，F0 中值 171.8 vs 175.5），两次 take 元音/音区差异会混入包络差",
        "包络移植作用于整段混合信号（含辅音/噪声），非论文的声源域处理——齿音/嘶声风险已标注",
        "morphing 只能搬运已有能量形态；论文指出 effort 增强还需要生成新谐波（未做）",
        "自实现零相位 FFT 整形，未与外部工具校验；Agent 未听音，结论以 Jade 试听为准",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "scale", "max_up_db", "max_down_db", "centroid_hz", "base_centroid",
                "target_centroid", "pitch", "lag", "raw_lufs", "raw_tp", "hard", "flags",
                "matched_lufs", "matched_tp"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["scale"], r["max_up_db"], r["max_down_db"], r["centroid_hz"],
                    r["base_centroid_hz"], r["target_centroid_hz"], r["pitch_median_hz"],
                    r["lag_samples"], round(r["raw_lufs"], 2), round(r["raw_true_peak_dbtp"], 2),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"]),
                    m.get("lufs", ""), m.get("true_peak_dbtp", "")])

params = {
    "experiment": "F1-EXP-016", "phase": "F1-P4C",
    "question": "论文 morphing 半部分（包络移植）能否把已认可的效果调到一听可辨",
    "single_module": True, "no_stacking": True,
    "base": "EXP-015 X1 (harmonic alignment, approved)",
    "samples": [SAMPLE_ID], "sealed": SEALED,
    "mechanism": "FFT-domain zero-phase envelope morphing toward same-song normal take; 60%/100% two strengths",
    "paper": qa["paper"], "morph_curve": qa["morph_curve"],
    "listening_target_lufs": target_lufs,
    "stop_condition": "若 100% 档仍不'一听可辨' → 下一轮实现论文另一半（谐波增富/新谐波生成），不再加大 EQ；若明显且好 → A05 复验",
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-016", "phase": "F1-P4C", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "base": "EXP-015 X1",
    "versions": [r["id"] for r in records],
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "diag_marked": [r["id"] for r in records if r["diag_mark"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(r.get("matched", {}).get("lufs", target_lufs + 99) - target_lufs) <= 0.3
                for r in records if r["deliverable"]),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
    "goal": "把效果调到一听可辨（用户需求升级）",
    "next_rule": "100%仍不明显 → 实现论文另一半（谐波增富）；明显且好 → A05 复验",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-016\nphase: F1-P4C\n"
    f"envelope morphing (Perrotin Interspeech 2016, morphing half) on EXP-015 X1 base\n"
    f"sealed: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")
(OUT / "RESULT.md").write_text(
    "# F1-EXP-016 结果（pending_jade）\n\n"
    "F1-P4C 包络移植：论文（Perrotin, Interspeech 2016）morphing 半部分——把正常版实测剩余包络差"
    "（2.5–3.5kHz 芯区 +5–8kHz 高区 + 过冲区回调）移植到已认可的 X1 基线上，60%/100% 两档。\n\n"
    f"- 交付：{', '.join(r['id'] for r in records if r['deliverable'])}（匿名 A/B）+ 上轮最佳版对照\n"
    f"- 技术失败：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    f"- 工程标记：{', '.join(r['id'] for r in records if r['diag_mark']) or '无'}\n\n"
    "本轮目标为用户需求升级后的'一听可辨'。若 100% 档仍不满足，下一轮实现论文另一半（谐波增富）。\n"
    "听感结论待 Jade 试听后填写。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "morph_curve_smoothed": {f"{int(c)}Hz": round(curve[i], 2) for i, c in enumerate(centers)},
    "versions": [{"id": r["id"], "scale": r["scale"], "max_up": r["max_up_db"], "max_down": r["max_down_db"],
                  "centroid": r["centroid_hz"], "pitch": r["pitch_median_hz"],
                  "hard": r["hard_fail"], "flags": r["flags"]} for r in records],
}, ensure_ascii=False, indent=1))
