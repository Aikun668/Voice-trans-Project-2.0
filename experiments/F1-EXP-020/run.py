"""F1-EXP-020 / F1-P6: 第一共振峰（F1）位置调整。

只做 F1 位置上移 +20Hz/+45Hz，基线为 EXP-016 M2（原样引用）。
方法：LPC 极点旋转——保持极点模长（带宽不变）只改角度（频率上移），
由 A_old/A_new 得包络增益，施加在 STFT 幅度谱上（相位保留 ⇒ F0 不变）。
仅处理可靠浊音帧（r≥0.5），辅音不动；非变调、非全频 EQ。
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

OUT = ROOT / "experiments/F1-EXP-020"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-020 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

TARGET_ID = "A_self_01"
BASE_PATH = ROOT / "experiments/F1-EXP-016/outputs/processed_raw/M2.wav"
BASE_QA = ROOT / "experiments/F1-EXP-016/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
SHIFTS = [("S20", 20.0), ("S45", 45.0)]

STFT_N, STFT_H = 2048, 512
FMIN, FMAX = 90.0, 450.0
R_GATE_VOICED = 0.50
FLOOR_DB = 40.0
LPC_SR = 16000
LPC_ORDER = 14
F1_LO, F1_HI = 150.0, 500.0
BW_LO, BW_HI = 20.0, 600.0
SMOOTH_FRAMES = 2
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
SEED = 20261109


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


def levinson(r, order):
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


def lpc_of(x16):
    xp = np.append(x16[0], x16[1:] - 0.63 * x16[:-1])
    xp = xp * np.hanning(len(xp))
    r = np.correlate(xp, xp, mode="full")[len(xp) - 1:len(xp) + LPC_ORDER]
    if r[0] <= 0:
        return None
    return levinson(r, LPC_ORDER)


def poles(a, sr=LPC_SR):
    roots = np.roots(a)
    freqs = np.angle(roots) * sr / (2.0 * np.pi)
    bws = -np.log(np.maximum(np.abs(roots), 1e-12)) * sr / np.pi
    return roots, freqs, bws


def f1_index(roots, freqs, bws):
    cand = [i for i in range(len(roots))
            if np.imag(roots[i]) > 0 and F1_LO <= freqs[i] <= F1_HI and BW_LO <= bws[i] <= BW_HI]
    return min(cand, key=lambda i: freqs[i]) if cand else None


def shift_gain(x_frame, sr, shift_hz):
    gcd = math.gcd(sr, LPC_SR)
    x16 = signal.resample_poly(x_frame, LPC_SR // gcd, sr // gcd)
    if len(x16) < LPC_ORDER * 4:
        return None
    a_old = lpc_of(x16)
    if a_old is None or len(a_old) < 3:
        return None
    roots, freqs, bws = poles(a_old)
    idx = f1_index(roots, freqs, bws)
    if idx is None:
        return None
    conj = int(np.argmin(np.abs(roots - np.conj(roots[idx]))))
    z = roots[idx]
    dw = shift_hz * 2.0 * np.pi / LPC_SR
    nz = np.abs(z) * np.exp(1j * (np.angle(z) + dw))
    rn = roots.copy()
    rn[idx] = nz
    rn[conj] = np.conj(nz)
    a_new = np.poly(rn).real
    a_new = a_new / a_new[0]
    w = np.linspace(0, np.pi, 401)
    k = np.arange(len(a_old))
    E = np.exp(-1j * np.outer(w, k))
    A_old = np.abs(E @ a_old)
    A_new = np.abs(E @ a_new)
    g = 20.0 * np.log10(np.maximum(A_old, 1e-10) / np.maximum(A_new, 1e-10))
    return w, g, float(freqs[idx]), float(bws[idx])


def apply_shift(x, sr, shift_hz):
    N, H = STFT_N, STFT_H
    ww = np.hanning(N)
    pad = N
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - N) // H
    nb = N // 2 + 1
    f_bins = np.arange(nb) * (sr / N)
    f16 = np.linspace(0, np.pi, 401) * (LPC_SR / (2.0 * np.pi))

    rms_db, rs = [], []
    for i in range(n_frames):
        fr = xp[i * H:i * H + N]
        rms_db.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        _, r = frame_f0_ac(fr, sr)
        rs.append(r)
    rms_db = np.array(rms_db)
    gate = rms_db.max() - FLOOR_DB

    gains = np.zeros((n_frames, nb))
    f1s, n_proc, n_fail = [], 0, 0
    for i in range(n_frames):
        if rms_db[i] < gate or rs[i] < R_GATE_VOICED:
            continue
        res = shift_gain(xp[i * H:i * H + N], sr, shift_hz)
        if res is None:
            n_fail += 1
            continue
        wv, g, f1o, _ = res
        gains[i] = np.interp(f_bins, f16, g, left=0.0, right=0.0)
        f1s.append(f1o)
        n_proc += 1

    if n_proc < 50:
        raise RuntimeError(f"可靠浊音帧过少 ({n_proc})，F1 无法可靠修改")

    sm = np.copy(gains)
    for i in range(n_frames):
        lo = max(0, i - SMOOTH_FRAMES // 2)
        hi = min(n_frames, i + SMOOTH_FRAMES // 2 + 1)
        sm[i] = gains[lo:hi].mean(axis=0)
    gains = sm

    y_ola = np.zeros(len(xp))
    norm = np.zeros(len(xp))
    for i in range(n_frames):
        fr = xp[i * H:i * H + N] * ww
        y = np.fft.irfft(np.fft.rfft(fr) * 10.0 ** (gains[i] / 20.0), N)
        s = i * H
        y_ola[s:s + N] += y * ww
        norm[s:s + N] += ww * ww
    y = (y_ola / np.maximum(norm, 1e-8))[pad:pad + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))

    stats = {"frames_total": int(n_frames), "vowel_frames": int(n_proc),
             "vowel_fraction": round(n_proc / n_frames, 3), "lpc_failures": int(n_fail),
             "f1_source_median_hz": round(float(np.median(f1s)), 1),
             "max_gain_db": round(float(np.max(np.abs(gains))), 2)}
    return y, stats


def measure_f1(x, sr):
    win, hop = int(round(sr * ANA_WIN_S)), int(round(sr * ANA_HOP_S))
    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB
    vals = []
    gcd = math.gcd(sr, LPC_SR)
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        _, r = frame_f0_ac(seg, sr)
        if r < R_GATE_VOICED:
            continue
        x16 = signal.resample_poly(seg, LPC_SR // gcd, sr // gcd)
        a = lpc_of(x16)
        if a is None:
            continue
        roots, freqs, bws = poles(a)
        idx = f1_index(roots, freqs, bws)
        if idx is not None:
            vals.append(float(freqs[idx]))
    if len(vals) < 20:
        return None, len(vals)
    return float(np.median(vals)), len(vals)


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


x_orig, sr = load(TARGET_ID)
if not BASE_PATH.exists():
    raise FileNotFoundError("EXP-016 M2.wav missing")
exp016 = json.loads(BASE_QA.read_text(encoding="utf-8"))
base_sha = next(v for v in exp016["versions"] if v["id"] == "M2")["raw_sha256"]
if sha(BASE_PATH) != base_sha:
    raise ValueError("baseline changed")
x_base, sr_b = read_pcm16(BASE_PATH)
if sr_b != sr:
    raise RuntimeError("sample rate mismatch")

f1_orig, n_orig = measure_f1(x_orig, sr)
f1_base, n_base = measure_f1(x_base, sr)
if f1_base is None:
    raise RuntimeError("基线 F1 无法可靠估计，技术障碍")

raw_root = OUT / "outputs/processed_raw"
records = []
for vid, shift in SHIFTS:
    y, pstat = apply_shift(x_base, sr, shift)
    f1_after, n_after = measure_f1(y, sr)
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    pitch_base = float(np.nanmedian(estimate_pitch(x_base, sr)))
    hard, flags = [], []
    actual = (f1_after - f1_base) if f1_after is not None else float("nan")
    if f1_after is None:
        hard.append("处理后 F1 无法可靠估计")
    elif abs(actual - shift) > max(8.0, 0.35 * shift):
        hard.append(f"F1 位移未达预期 实测{actual:+.1f} 目标{shift:+.0f}Hz")
    if abs(pitch - pitch_base) > 2.0:
        hard.append(f"F0 偏移 {pitch - pitch_base:+.2f}Hz")
    if len(y) != len(x_base):
        hard.append(f"样本数变化 {len(y)} vs {len(x_base)}")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768))
    if clipped > 0:
        hard.append(f"削波 {clipped}")
    if pstat["vowel_fraction"] < 0.3:
        flags.append(f"覆盖偏低 {pstat['vowel_fraction']:.2f}")
    if pstat["lpc_failures"] > 0.2 * pstat["frames_total"]:
        flags.append(f"LPC 失败偏多 {pstat['lpc_failures']}")

    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, 0.0, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    records.append({
        "id": vid, "target_shift_hz": shift,
        "actual_f1_shift_hz": round(actual, 1) if f1_after else None,
        "f1_before_hz": round(f1_base, 1),
        "f1_after_hz": round(f1_after, 1) if f1_after else None,
        "proc": pstat, "pitch_hz": round(pitch, 2), "pitch_base_hz": round(pitch_base, 2),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "frames": len(y), "clipped": clipped,
        "raw_lufs": round(ye_loud["integrated_lufs"], 2),
        "raw_true_peak_dbtp": round(ye_loud["true_peak_dbtp"], 2),
        "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"),
        "hard_fail": hard, "flags": flags, "deliverable": not hard, "diag_mark": bool(flags),
    })

loud_orig = measure_loudness(x_orig, sr)
loud_base = measure_loudness(x_base, sr)
loud_all = {"original": loud_orig, "baseline": loud_base,
            **{r["id"]: {"integrated_lufs": r["raw_lufs"], "true_peak_dbtp": r["raw_true_peak_dbtp"]}
               for r in records}}
target_lufs = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loud_all.values()])

listen_root = OUT / "listening/audio"
exports = []
for tag, data_, loud in (("original", x_orig, loud_orig), ("baseline", x_base, loud_base)):
    dest = listen_root / f"{tag}.wav"
    export_wav(data_, sr, target_lufs - loud["integrated_lufs"], dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target_lufs) > 0.3:
        raise RuntimeError(f"{tag} loudness match failed")
    exports.append({"tag": tag, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                    "lufs": dm["integrated_lufs"], "true_peak_dbtp": dm["true_peak_dbtp"], "sha256": sha(dest)})

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-020")
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
                    "true_peak_dbtp": dm["true_peak_dbtp"], "sha256": sha(dest)}

rng = random.Random(SEED)
deliverables = [r["id"] for r in records if r["deliverable"]]
letters = ["A", "B", "C"][:len(deliverables)]
blind = {letter: vid for letter, vid in zip(letters, rng.sample(deliverables, len(deliverables)))}
for letter, vid in blind.items():
    shutil.copyfile(listen_root / f"{vid}.wav", listen_root / f"{letter}.wav")
    (listen_root / f"{vid}.wav").unlink()


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


rows = []
for letter in blind.keys():
    r = next(v for v in records if v["id"] == blind[letter])
    badge = ' <span class="badge">带工程标记</span>' if r["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter}{badge}</label>'
                f'<audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P6 F1位置调整（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:9px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P6 · 第一共振峰位置调整（A01）</h1>
<div class="key"><b>本轮补的是"从未实测就排除"的欠账。</b><br>
方法 = LPC 极点旋转：保持极点带宽不变、只把第一共振峰频率上移；相位与谐波位置不动 ⇒ <b>音高完全不变</b>。<br>
只在可靠浊音帧（元音）上处理，辅音不动。基线是你认可的最佳版 X1+M2。</div>
<div class="note"><b>请重点评价：</b>①饱满度是否明显提升？②是否更有"发声力度/主体感"？③元音、咬字、是否还像本人？④是否只是变亮、变尖或有失真？<br>
两档位移已隐藏。</div>
<div class="card"><h3>试听</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>基线（X1+M2 最佳版）</label><audio controls preload="metadata" src="audio/baseline.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">四版统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）；未改时长/节奏/音高；未新增其他模块；Holdout 未处理。<br>
客观：基线 F1 = {f1_base:.0f}Hz，两档实测位移见报告。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-020", "phase": "F1-P6", "status": "pending_jade",
    "question": "F1 位置上移能否带来可听正向改善（补'未测就排除'的欠账）",
    "method": "LPC 极点旋转；保持|z|不变只改角度；包络增益施于 STFT 幅度谱（相位保留）",
    "vowel_gate": f"仅 r≥{R_GATE_VOICED} 且能量≥最响帧-{FLOOR_DB}dB 的帧",
    "f0_invariance": "相位保留 + 谐波位置不动",
    "baseline": {"id": "EXP-016 M2", "sha256": base_sha, "f1_hz": round(f1_base, 1),
                 "f1_frames": n_base},
    "original": {"id": TARGET_ID, "f1_hz": round(f1_orig, 1) if f1_orig else None, "f1_frames": n_orig},
    "target_lufs": target_lufs, "exports": exports, "versions": records,
    "sealed_not_processed": SEALED,
    "no_new_modules": "只加 F1 位置；未做共振峰增益/逐帧包络轨迹/F0 活跃度",
    "stop_condition": "两档均无正向改善或伤咬字/自然度 → 停止，不加大位移",
    "limitations": [
        "A01 F0≈176Hz 与 F1≈230Hz 间隔较近，F1 极点可能受基频影响",
        "LPC(14)@16kHz 覆盖到 8kHz，F3+ 未单独控制",
        "仅处理浊音帧，浊/清边界已做帧平滑",
        "自实现机制未与外部工具校验；Agent 未听音",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "target_shift", "actual_shift", "f1_before", "f1_after", "pitch", "pitch_base",
                "frames", "vowel_fraction", "lpc_failures", "max_gain", "raw_lufs", "raw_tp",
                "clipped", "matched_lufs", "hard", "flags"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["target_shift_hz"], r["actual_f1_shift_hz"], r["f1_before_hz"],
                    r["f1_after_hz"], r["pitch_hz"], r["pitch_base_hz"], r["frames"],
                    r["proc"]["vowel_fraction"], r["proc"]["lpc_failures"], r["proc"]["max_gain_db"],
                    r["raw_lufs"], r["raw_true_peak_dbtp"], r["clipped"], m.get("lufs", ""),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"])])

params = {
    "experiment": "F1-EXP-020", "phase": "F1-P6",
    "approved_scope": "只做 F1 位置上移 +20/+45Hz",
    "baseline": "EXP-016 M2（原样引用）", "target": TARGET_ID, "sealed": SEALED,
    "listening_target_lufs": target_lufs,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-020", "phase": "F1-P6", "status": "pending_jade",
    "target": TARGET_ID, "baseline": "EXP-016 M2",
    "f1_original_hz": round(f1_orig, 1) if f1_orig else None,
    "f1_baseline_hz": round(f1_base, 1),
    "versions": [r["id"] for r in records],
    "actual_shifts_hz": {r["id"]: r["actual_f1_shift_hz"] for r in records},
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(e["lufs"] - target_lufs) <= 0.3 for e in exports),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
    "next_rule": "有正向改善则考虑 A02 复验；两档均无改善或伤咬字 → 停止本方向",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-020\nphase: F1-P6\n"
    f"F1 shift via LPC pole rotation on EXP-016 M2; +20/+45 Hz; F0/time untouched; vowels only\n"
    f"sealed untouched: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "f1_original_hz": round(f1_orig, 1) if f1_orig else None,
    "f1_baseline_hz": round(f1_base, 1),
    "versions": [{k: r[k] for k in ("id", "target_shift_hz", "actual_f1_shift_hz", "f1_after_hz",
                                     "pitch_hz", "proc", "hard_fail", "flags")} for r in records],
}, ensure_ascii=False, indent=1))
