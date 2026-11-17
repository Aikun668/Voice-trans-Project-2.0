"""F1-EXP-021 / F1-P7: 起音瞬态增强验证。

基线 = EXP-020 的 S45（X1+M2+F1 最大档，原样引用）。
只做"可靠元音起音"的短时动态增强，持续元音主体保持不变；
两档峰值目标 +1.5dB / +3.0dB；不改 F0/节奏/时长。

起音判据（三重，用于排除辅音、爆破音、噪声）：
  1) 该帧必须是"浊音"（归一化自相关峰值 r ≥ 0.45）→ 排除清辅音与噪声；
  2) 该帧能量必须 ≥ 最响帧 −35dB → 排除极弱瞬态；
  3) 相对前 3 帧（15ms）能量必须跃升 ≥ 6dB，或前一帧为清音 → 确认为起音沿；
  4) 其后 4 帧（20ms）必须保持浊音 → 排除孤立爆破/咔哒。
不满足任一条件则不处理，并把无法定位的情况如实报告。
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

OUT = ROOT / "experiments/F1-EXP-021"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-021 outputs already exist")
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

TARGET_ID = "A_self_01"
BASE_PATH = ROOT / "experiments/F1-EXP-020/outputs/processed_raw/S45.wav"
BASE_QA = ROOT / "experiments/F1-EXP-020/qa/qa_report.json"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]
BOOSTS = [("T15", 1.5), ("T30", 3.0)]

# 起音检测
ANA_WIN = 1200          # 25 ms @48kHz
ANA_HOP = 240           # 5 ms
FMIN, FMAX = 90.0, 450.0
R_VOICED = 0.45
ENERGY_FLOOR_DB = 35.0
RISE_DB = 6.0
CONFIRM_FRAMES = 4      # 其后 20ms 必须保持浊音
LOOKBACK = 3

# 增强包络（平滑，两端为 0）
RAMP_IN_S = 0.006
HOLD_S = 0.008
RAMP_OUT_S = 0.036
MAX_LEN_S = 0.050
MIN_GAP_S = 0.012       # 窗口不得越过下一个起音
SEED = 20261110


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


def detect_onsets(x, sr):
    starts = list(range(0, len(x) - ANA_WIN + 1, ANA_HOP))
    rms, rv, f0v = [], [], []
    for s in starts:
        fr = x[s:s + ANA_WIN]
        rms.append(20.0 * np.log10(max(float(np.sqrt(np.mean(fr ** 2))), 1e-12)))
        f0, r = frame_f0_ac(fr, sr)
        rv.append(r)
        f0v.append(f0)
    rms = np.array(rms)
    rv = np.array(rv)
    gate = rms.max() - ENERGY_FLOOR_DB
    voiced = (rv >= R_VOICED) & (rms >= gate)

    onsets, reasons = [], {"too_weak": 0, "no_confirm": 0, "no_rise": 0}
    for i in range(LOOKBACK, len(starts)):
        if not voiced[i]:
            continue
        # 3) 能量跃升或前一帧为清音
        rise = rms[i] - float(np.max(rms[i - LOOKBACK:i]))
        prev_clear = not voiced[i - 1]
        if rise < RISE_DB and not prev_clear:
            reasons["no_rise"] += 1
            continue
        # 4) 其后若干帧保持浊音
        j_end = min(i + CONFIRM_FRAMES + 1, len(starts))
        if not np.all(voiced[i:j_end]):
            reasons["no_confirm"] += 1
            continue
        # 2) 能量足够（voiced 已含 gate 判断，这里再核一次）
        if rms[i] < gate:
            reasons["too_weak"] += 1
            continue
        # 避免同一段连续浊音被重复标记
        if onsets and (starts[i] - onsets[-1]) < int(MIN_GAP_S * sr):
            continue
        onsets.append(starts[i])

    stats = {
        "frames": int(len(starts)),
        "voiced_fraction": round(float(np.mean(voiced)), 3),
        "onsets_detected": len(onsets),
        "onsets_per_second": round(len(onsets) / (len(x) / sr), 2),
        "rejected_no_rise": reasons["no_rise"],
        "rejected_no_confirm": reasons["no_confirm"],
        "gate_db": round(float(gate), 1),
    }
    return onsets, stats


def make_gain_series(n, sr, onsets):
    gdb = np.zeros(n)
    ai, ho, ro = int(RAMP_IN_S * sr), int(HOLD_S * sr), int(RAMP_OUT_S * sr)
    total = ai + ho + ro
    total = min(total, int(MAX_LEN_S * sr))
    shape = np.zeros(total)
    shape[:ai] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ai) / max(ai - 1, 1)))
    shape[ai:ai + ho] = 1.0
    tail = total - ai - ho
    if tail > 0:
        shape[ai + ho:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(tail) / max(tail - 1, 1)))
    shape[-1] = 0.0

    spans = []
    for k, o in enumerate(onsets):
        limit = n - o
        if k + 1 < len(onsets):
            limit = min(limit, onsets[k + 1] - o - int(MIN_GAP_S * sr))
        length = min(total, max(limit, 0))
        if length < int(0.012 * sr):
            continue
        spans.append((o, o + length))
        seg = shape[:length]
        # 取 max 避免重叠区域叠加过高
        gdb[o:o + length] = np.maximum(gdb[o:o + length], seg)
    return gdb, spans


def apply_boost(x, sr, onsets, peak_db):
    gdb, spans = make_gain_series(len(x), sr, onsets)
    gain_lin = 10.0 ** (gdb * peak_db / 20.0)
    y = x * gain_lin
    return y, gdb * peak_db, spans, gain_lin


def measure_region_delta(x_old, x_new, sr, spans, gdb_track=None, win_ms=25):
    """测量覆盖区与非覆盖区的实际增益（dB）。

    口径说明（本轮修正过一次，理由记录在案）：
      增益包络总长 50ms，而 RMS 窗口若用 25ms，会把两端衰减段一并平均，
      中位增益必然低于峰值（实测 P90 才接近峰值）。这属于测量口径偏差，
      不是处理失败。因此增加 gdb_track 后，对"包络接近峰值的窗口"单独统计，
      以反映起音峰实际被抬高的量；同时保留全窗口中位值作为对照。
    """
    n = min(len(x_old), len(x_new))
    mask = np.zeros(n, dtype=bool)
    for a, b in spans:
        mask[a:min(b, n)] = True
    w = int(win_ms * 1e-3 * sr)
    hop = w // 2
    peak_ref = float(np.max(np.abs(gdb_track))) if gdb_track is not None and np.any(gdb_track) else 0.0
    ins, outs, ins_peak = [], [], []
    for s in range(0, n - w + 1, hop):
        eo = float(np.sqrt(np.mean(x_old[s:s + w] ** 2)))
        en = float(np.sqrt(np.mean(x_new[s:s + w] ** 2)))
        if eo < 1e-8:
            continue
        d = 20.0 * np.log10(max(en, 1e-12) / max(eo, 1e-12))
        cov = float(np.mean(mask[s:s + w]))
        if cov > 0.6:
            ins.append(d)
            if peak_ref > 0 and gdb_track is not None:
                env = float(np.max(np.abs(gdb_track[s:s + w])))
                if env >= 0.8 * peak_ref:
                    ins_peak.append(d)
        elif cov < 0.1:
            outs.append(d)
    return {
        "onset_region_median_db": round(float(np.median(ins)), 2) if ins else None,
        "onset_region_p90_db": round(float(np.percentile(ins, 90)), 2) if ins else None,
        "onset_peakwin_median_db": round(float(np.median(ins_peak)), 2) if ins_peak else None,
        "onset_peakwin_p10_db": round(float(np.percentile(ins_peak, 10)), 2) if ins_peak else None,
        "sustain_region_median_db": round(float(np.median(outs)), 2) if outs else None,
        "n_onset_windows": len(ins), "n_onset_peakwin": len(ins_peak), "n_sustain_windows": len(outs),
    }


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
    raise FileNotFoundError("EXP-020 S45.wav missing")
exp020 = json.loads(BASE_QA.read_text(encoding="utf-8"))
base_rec = next(v for v in exp020["versions"] if v["id"] == "S45")
if not base_rec.get("deliverable", True):
    raise ValueError("EXP-020 S45 was not a deliverable")
base_sha = sha(BASE_PATH)
x_base, sr_b = read_pcm16(BASE_PATH)
if sr_b != sr:
    raise RuntimeError("sample rate mismatch")
if len(x_base) != base_rec["frames"]:
    raise ValueError(f"baseline frame count drift {len(x_base)} vs {base_rec['frames']}")

onsets, detect_stats = detect_onsets(x_base, sr)
if len(onsets) < 8:
    raise RuntimeError(f"可定位的元音起音过少（{len(onsets)}），无法可靠处理，技术障碍")
gaps = np.diff(onsets) / sr if len(onsets) > 1 else np.array([0.0])
detect_stats.update({
    "onset_interval_median_s": round(float(np.median(gaps)), 3),
    "onset_interval_p10_s": round(float(np.percentile(gaps, 10)), 3) if len(gaps) else None,
    "coverage_seconds": round(sum((b - a) for a, b in make_gain_series(len(x_base), sr, onsets)[1]) / sr, 2),
})

raw_root = OUT / "outputs/processed_raw"
records = []
for vid, peak_db in BOOSTS:
    y, gdb_track, spans, _ = apply_boost(x_base, sr, onsets, peak_db)
    delta = measure_region_delta(x_base, y, sr, spans, gdb_track=gdb_track)
    st = sample_stats(y, sr)
    loud = measure_loudness(y, sr)
    pitch = float(np.nanmedian(estimate_pitch(y, sr)))
    pitch_base = float(np.nanmedian(estimate_pitch(x_base, sr)))
    hard, flags = [], []
    # 验收用"包络峰值处"的实际增益（与 50ms 包络口径一致），避免 25ms 窗口平均造成的系统偏低
    measured = delta["onset_peakwin_median_db"]
    if measured is None:
        hard.append("覆盖区增益无法测量")
    elif abs(measured - peak_db) > max(0.6, 0.3 * peak_db):
        hard.append(f"起音增益未达目标 峰值窗口实测{measured:+.2f} 目标{peak_db:+.1f}dB")
    if delta["sustain_region_median_db"] is None or abs(delta["sustain_region_median_db"]) > 0.15:
        hard.append(f"持续段被误改 {delta['sustain_region_median_db']}")
    if abs(pitch - pitch_base) > 1.0:
        hard.append(f"F0 偏移 {pitch - pitch_base:+.2f}Hz")
    if len(y) != len(x_base):
        hard.append(f"样本数变化 {len(y)}")
    if not all(np.isfinite(y)):
        hard.append("非有限值")
    clipped = int(np.count_nonzero(np.abs(y) >= 32767 / 32768))
    if clipped > 0:
        hard.append(f"削波 {clipped}")
    if detect_stats["onsets_detected"] < 0.5 / detect_stats["onset_interval_median_s"] if detect_stats["onset_interval_median_s"] else False:
        flags.append("起音密度偏低")
    peak_env = float(np.max(np.abs(gdb_track)))
    if peak_env > 0 and abs(peak_env - peak_db) > 1e-6:
        flags.append("包络峰值与目标不一致")

    rp = raw_root / f"{vid}.wav"
    export_wav(y, sr, 0.0, rp)
    ye, esr = read_pcm16(rp)
    ye_loud = measure_loudness(ye, esr)
    records.append({
        "id": vid, "target_peak_db": peak_db,
        "onset_region_db": delta["onset_region_median_db"],
        "onset_region_p90_db": delta["onset_region_p90_db"],
        "onset_peakwin_db": delta["onset_peakwin_median_db"],
        "onset_peakwin_p10_db": delta["onset_peakwin_p10_db"],
        "sustain_region_db": delta["sustain_region_median_db"],
        "n_onset_windows": delta["n_onset_windows"],
        "n_onset_peakwin": delta["n_onset_peakwin"],
        "n_sustain_windows": delta["n_sustain_windows"],
        "envelope_peak_db": round(peak_env, 3),
        "boosted_seconds": detect_stats["coverage_seconds"],
        "pitch_hz": round(pitch, 2), "pitch_base_hz": round(pitch_base, 2),
        "peak_dbfs": round(st["sample_peak_dbfs"], 2), "crest_db": round(st["sample_peak_dbfs"] - st["rms_dbfs"], 2),
        "frames": len(y), "clipped": clipped,
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
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-021")
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

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P7 起音瞬态增强（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:170px 1fr;gap:10px;align-items:center;margin:9px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.key{{background:#eef4ff;border:1px solid #c4d6f0;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P7 · 起音瞬态增强（A01）</h1>
<div class="key"><b>本轮只改"每个元音开口的那一下"。</b><br>
起音由三重判据定位：浊音（排除清辅音/噪声）+ 能量跃升 ≥6dB + 其后 20ms 保持浊音（排除爆破咔哒）。<br>
增益包络两端为 0、最长 50ms、不越过下一个起音；<b>持续元音主体完全不动</b>。音高、节奏、时长不变。<br>
基线是你认可的最佳版（X1+M2+F1 最大档）。</div>
<div class="note"><b>请重点评价：</b>①发声是否更有力量、更自然？②饱满度是否真正进一步提高？③是否只是"声头更硬/局部更响"？④有无损伤音色、咬字、自然度？<br>
两档强度已隐藏（+1.5dB / +3.0dB）。</div>
<div class="card"><h3>试听</h3>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>
<div class="row"><label>基线（当前最佳，未改动）</label><audio controls preload="metadata" src="audio/baseline.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">四版统一响度 {target_lufs:.2f} LUFS（常数增益，无限制器）。<br>
客观：检测到 {detect_stats['onsets_detected']} 个元音起音（约 {detect_stats['onsets_per_second']} 个/秒），累计增强 {detect_stats['coverage_seconds']}s / {len(x_base)/sr:.1f}s；持续段增益实测应≈0。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-021", "phase": "F1-P7", "status": "pending_jade",
    "question": "元音起音瞬态增强能否增加自然发声力度感",
    "method": "浊音起音定位 + 短时平滑增益包络（两端为0，最长50ms，不越过下一��音）",
    "onset_detection": detect_stats,
    "onset_rules": {
        "voiced": f"r ≥ {R_VOICED}（排除清辅音与噪声）",
        "energy": f"≥ 最响帧 −{ENERGY_FLOOR_DB}dB",
        "rise_or_prev_clear": f"相对前 {LOOKBACK*5}ms 跃升 ≥{RISE_DB}dB，或前一帧为清音",
        "confirm": f"其后 {CONFIRM_FRAMES*5}ms 保持浊音（排除爆破/咔哒）",
        "min_gap": f"相邻起音至少间隔 {MIN_GAP_S*1000:.0f}ms",
    },
    "envelope": {"ramp_in_s": RAMP_IN_S, "hold_s": HOLD_S, "ramp_out_s": RAMP_OUT_S,
                 "max_len_s": MAX_LEN_S, "overlap_rule": "多窗口重叠取 max，不叠加"},
    "baseline": {"id": "EXP-020 S45", "sha256": base_sha},
    "target_lufs": target_lufs, "exports": exports,
    "versions": records, "sealed_not_processed": SEALED,
    "no_other_modules": "只做起音增益；未做全局扩展、F1 带宽、F0 修改",
    "stop_condition": "两档均仅表现为音量变化/声头突兀/无感 → 停止起音方向，不加量",
    "limitations": [
        "起音检测基于自相关周期性，气声很重的起音可能被误排除",
        "20ms 窗口在快速咬字处可能跨越音节边界（已用 MIN_GAP 与 max 规则限制）",
        "自实现机制未与外部工具校验；Agent 未听音",
    ],
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "target_peak_db", "onset_peakwin_db", "onset_peakwin_p10", "onset_region_median",
                "onset_p90", "sustain_region_db", "envelope_peak_db", "boosted_s", "pitch",
                "pitch_base", "crest", "clipped", "raw_lufs", "raw_tp", "matched_lufs", "hard", "flags"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["target_peak_db"], r["onset_peakwin_db"], r["onset_peakwin_p10_db"],
                    r["onset_region_db"], r["onset_region_p90_db"], r["sustain_region_db"],
                    r["envelope_peak_db"], r["boosted_seconds"],
                    r["pitch_hz"], r["pitch_base_hz"], r["crest_db"], r["clipped"],
                    r["raw_lufs"], r["raw_true_peak_dbtp"], m.get("lufs", ""),
                    "; ".join(r["hard_fail"]), "; ".join(r["flags"])])

params = {
    "experiment": "F1-EXP-021", "phase": "F1-P7",
    "approved_scope": "只做元音起音瞬态增强；不做全局扩展、F1 带宽、F0 修改",
    "baseline": "EXP-020 S45（原样引用）", "target": TARGET_ID, "sealed": SEALED,
    "listening_target_lufs": target_lufs,
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-021", "phase": "F1-P7", "status": "pending_jade",
    "target": TARGET_ID, "baseline": "EXP-020 S45",
    "onsets_detected": detect_stats["onsets_detected"],
    "onsets_per_second": detect_stats["onsets_per_second"],
    "onset_interval_median_s": detect_stats["onset_interval_median_s"],
    "boosted_seconds": detect_stats["coverage_seconds"],
    "versions": [r["id"] for r in records],
    "actual_onset_gain_db": {r["id"]: r["onset_peakwin_db"] for r in records},
    "onset_region_median_db": {r["id"]: r["onset_region_db"] for r in records},
    "sustain_leak_db": {r["id"]: r["sustain_region_db"] for r in records},
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "target_lufs": target_lufs,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(e["lufs"] - target_lufs) <= 0.3 for e in exports),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
    "next_rule": "有正向收益才扩大到其他 Dev 样本；仅音量变化/声头突兀/无感 → 停止本方向",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-021\nphase: F1-P7\n"
    f"vowel-onset transient boost on EXP-020 S45; peaks +1.5/+3.0 dB; sustain untouched\n"
    f"sealed untouched: {', '.join(SEALED)}\ntarget_lufs: {target_lufs}\n"
    f"technical_pass: {summary['technical_pass']}\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "onset_detection": detect_stats,
    "versions": [{k: r[k] for k in ("id", "target_peak_db", "onset_peakwin_db", "onset_peakwin_p10_db",
                                     "onset_region_db", "onset_region_p90_db", "sustain_region_db",
                                     "envelope_peak_db", "pitch_hz", "crest_db", "hard_fail", "flags")}
                 for r in records],
}, ensure_ascii=False, indent=1))
