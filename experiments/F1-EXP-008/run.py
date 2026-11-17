"""F1-EXP-008 / F1-P3B: 单一透明宽带向下压缩 + 并行混合，A01 机制验证。

只验证一个问题：薄音能否通过动态压缩与并行混合获得可听、自然的密度改善。
不叠加饱和/EQ/多段压缩；Ratio 4:1, Attack 15ms, Release 120ms, Soft Knee 6dB,
RMS 包络检测（双支路一阶均方平滑，attack/release 即检测器时间常数），
Threshold 依据有声片段电平自动校准，补偿增益使压缩分支混合前与干声可比响度；
Wet 50% / 100%。
"""
import csv, hashlib, json, math, os, pathlib, platform, random, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats, frame_rms, estimate_pitch


def ensure_ffmpeg_on_path():
    """当前会话 PATH 可能没有 ffmpeg（WinGet 安装）；找到后注入 PATH。"""
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

OUT = ROOT / "experiments/F1-EXP-008"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-008 outputs already exist")

DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}

SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; SOURCE = "Self"; DISPLAY = "平凡之路 / 薄音版"
SEALED = ["A04", "A06", "B03", "C01", "C02", "C03"]  # Holdout（沿用 EXP-001 dev 划分）
SEED = 20261101

# --- 压缩器固定参数（本轮唯一机制） ---
RATIO = 4.0
ATTACK_MS = 15.0
RELEASE_MS = 120.0
KNEE_DB = 6.0
SLOPE = 1.0 - 1.0 / RATIO
ANA_WIN_S = 0.04         # 客观分析窗 40ms（按任务书）
ACTIVE_FLOOR_DB = 35.0    # 有声帧判据：帧RBS 距最大帧 -35dB 以内
CAL_TOL_DB = 0.2
CAL_MAX_ITER = 6
VERSIONS = [("P1", 4.0, 0.5), ("P2", 8.0, 0.5), ("P3", 4.0, 1.0), ("P4", 8.0, 1.0)]
GR_TARGETS = [4.0, 8.0]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------- 压缩器实现（固定，全程记录） ----------------
def static_gr(level_db, thr_db):
    """软拐点静态曲线：拐点中心在阈值、宽 KNEE_DB，拐点内二次插值，上方 GR=SLOPE*(L-T)。"""
    over = level_db - thr_db
    gr = np.zeros_like(level_db)
    k2 = KNEE_DB / 2.0
    ink = (over > -k2) & (over <= k2)
    gr[ink] = SLOPE * (over[ink] + k2) ** 2 / (2.0 * KNEE_DB)
    ab = over > k2
    gr[ab] = SLOPE * over[ab]
    return gr


def compress(x, sr, thr_db):
    """前馈 RMS 向下压缩（固定实现，见 qa_report.compressor）：
    双支路一阶 RMS 检测（对均方信号各做 attack τ=15ms / release τ=120ms 一阶平滑，
    逐样本取 max，即经典球学包络），level=10log10(包络均方) -> 软拐点静态曲线 ->
    增益相乘。无前视、无额外窗口、无削峰/染色。63% 时间常数在功率域定义。"""
    ms = x * x

    def one_pole(t, tau_s):
        alpha = math.exp(-1.0 / (sr * tau_s))
        return signal.lfilter([1.0 - alpha], [1.0, -alpha], t)

    p = np.maximum(one_pole(ms, ATTACK_MS / 1000.0), one_pole(ms, RELEASE_MS / 1000.0))
    level = 10.0 * np.log10(np.maximum(p, 1e-24))
    sm = static_gr(level, thr_db)
    y = x * 10.0 ** (-sm / 20.0)
    return y, sm


# ---------------- 40ms 帧分析 ----------------
def frame_db(x, sr, win_s=ANA_WIN_S):
    return 20.0 * np.log10(np.maximum(frame_rms(x, sr, win_s), 1e-12))


def gr_frames(sm, sr, win_s=ANA_WIN_S):
    size = max(1, round(sr * win_s))
    return sm[:len(sm) // size * size].reshape(-1, size).mean(axis=1)


def spectral(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    m = (f >= 20) & (f < 8000)
    f, p = f[m], p[m]
    total = max(float(np.sum(p)), 1e-20)
    return {"centroid_hz": float(np.sum(f * p) / total),
            "flatness": float(np.exp(np.mean(np.log(np.maximum(p, 1e-20)))) / max(np.mean(p), 1e-20))}


def lag_between(y, x, sr):
    seg = x[:min(len(x), 2 * sr)]
    yseg = y[:len(seg)]
    c = signal.correlate(yseg, seg, mode="full")
    return int(np.argmax(np.abs(c)) - (len(seg) - 1))


# ---------------- 目录与输入 ----------------
for d in (OUT / "outputs/processed_raw", OUT / "listening/audio", OUT / "qa"):
    if d.exists():
        raise FileExistsError(d)
    d.mkdir(parents=True)

item = items[DATASET_ID]
wav = DATA / item["wav"]["path"]
raw = DATA / item["raw"]["path"]
if sha(raw) != item["raw"]["sha256"]:
    raise ValueError("Input raw changed")
if sha(wav) != item["wav"]["sha256"]:
    raise ValueError("Input wav changed")
x, sr = read_pcm16(wav)
if len(x) != item["wav"]["frames"]:
    raise ValueError("wav frames mismatch vs manifest")

listening_contract = {
    "原音问题登记": "A01 为用户自评薄音版（单薄、不够充实）；本轮只针对该问题验证压缩机制",
    "素材与片段": "A01 全长 13.14s（满足 8~15s 典型片段要求），不截取",
    "预期可听变化": "元音与整句更充实、小声部分更实；大声不突兀；轻重对比仍在（S1 饱满度方向）",
    "否决副作用": "每个字一样重、情绪消失、抽吸(pumping)、气声/尾音异常放大、闷糊、被压住、失真、只变响或只变闷不算通过",
    "若原音无此问题": "记 not_applicable，本轮机制验证作废，不强行制造差异",
}
(OUT / "inputs.json").write_text(json.dumps({
    "experiment": "F1-EXP-008", "phase": "F1-P3B",
    "sample_id": SAMPLE_ID, "dataset_id": DATASET_ID, "source": SOURCE, "display": DISPLAY,
    "wav_file": str(wav.relative_to(ROOT)).replace("\\", "/"),
    "raw_file": str(raw.relative_to(ROOT)).replace("\\", "/"),
    "wav_sha256": sha(wav), "raw_sha256": sha(raw),
    "frames": len(x), "sample_rate": sr,
    "sealed_not_processed": SEALED, "listening_contract": listening_contract,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---------------- 原音客观基线 ----------------
orig_stats = sample_stats(x, sr)
orig_loud = measure_loudness(x, sr)
orig_spec = spectral(x, sr)
orig_fdb = frame_db(x, sr)
active = orig_fdb > (orig_fdb.max() - ACTIVE_FLOOR_DB)
if int(active.sum()) < 10:
    raise RuntimeError("active frames too few")
orig_dyn = {
    "window_ms": ANA_WIN_S * 1000, "active_floor_db": ACTIVE_FLOOR_DB,
    "frames_total": int(len(orig_fdb)), "frames_active": int(active.sum()),
    "active_fraction": float(active.mean()),
    "frame_db_p10": float(np.percentile(orig_fdb[active], 10)),
    "frame_db_p50": float(np.percentile(orig_fdb[active], 50)),
    "frame_db_p90": float(np.percentile(orig_fdb[active], 90)),
    "short_term_dynamic_range_db_p90_p10": float(np.percentile(orig_fdb[active], 90) - np.percentile(orig_fdb[active], 10)),
}
quiet_mask = orig_fdb[active] <= np.percentile(orig_fdb[active], 10)  # 有声帧中最安静10%（气声/尾音区）
orig_pitch = float(np.nanmedian(estimate_pitch(x, sr)))

# ---------------- 阈值校准（每目标一次；P1/P3 与 P2/P4 共享压缩分支） ----------------
branches = {}
for target in GR_TARGETS:
    thr = orig_dyn["frame_db_p90"] - target / SLOPE  # 静态曲线初值
    hist = []
    converged = False
    for it in range(CAL_MAX_ITER):
        used = float(thr)
        y_c, sm = compress(x, sr, used)
        actual = float(np.percentile(gr_frames(sm, sr)[active], 90))
        hist.append({"threshold_db": used, "gr_p90_active": actual})
        if abs(actual - target) <= CAL_TOL_DB:
            converged = True
            break
        if it < CAL_MAX_ITER - 1:
            thr += (actual - target) / SLOPE
    g = gr_frames(sm, sr)[active]
    comp_dyn = frame_db(y_c, sr)
    comp_loud = measure_loudness(y_c, sr)
    makeup = orig_loud["integrated_lufs"] - comp_loud["integrated_lufs"]
    branches[target] = {
        "y": y_c, "sm": sm, "threshold_db": used, "converged": converged,
        "calibration_history": hist, "achieved_gr_p90": actual,
        "gr_median_active": float(np.median(g)),
        "gr_p90_active": actual,
        "gr_min_active": float(g.min()), "gr_max_active": float(g.max()),
        "gr_max_full": float(sm.max()),
        "frac_gr_gt_1db_active": float(np.mean(g > 1.0)),
        "gr_at_quiet_p10": float(np.median(g[quiet_mask])),
        "wet_dyn_db": float(np.percentile(comp_dyn[active], 90) - np.percentile(comp_dyn[active], 10)),
        "wet_lufs": comp_loud["integrated_lufs"], "makeup_db": makeup,
        "spectral": spectral(y_c, sr),
    }

# ---------------- 四个版本：混合 + 客观筛查 ----------------
raw_root = OUT / "outputs/processed_raw"
records = []
for vid, target, wet in VERSIONS:
    b = branches[target]
    mix = (1.0 - wet) * x + wet * (b["y"] * 10.0 ** (b["makeup_db"] / 20.0))
    st = sample_stats(mix, sr)
    loud = measure_loudness(mix, sr)
    fdb = frame_db(mix, sr)
    dyn_mix = float(np.percentile(fdb[active], 90) - np.percentile(fdb[active], 10))
    quiet_boost = float(np.median(fdb[active][quiet_mask] - orig_fdb[active][quiet_mask]))
    crest = st["sample_peak_dbfs"] - st["rms_dbfs"]
    crest_drop = (orig_stats["sample_peak_dbfs"] - orig_stats["rms_dbfs"]) - crest
    sp = spectral(mix, sr)
    centroid_ratio = sp["centroid_hz"] / orig_spec["centroid_hz"]
    lag = lag_between(mix, x, sr)
    jump_ratio = float(np.max(np.abs(np.diff(mix))) / max(np.max(np.abs(np.diff(x))), 1e-12))
    silent_in_active = int(np.count_nonzero(fdb[active] < -80.0))
    pitch = float(np.nanmedian(estimate_pitch(mix, sr)))
    peak_db = st["sample_peak_dbfs"]

    flags = []
    # 硬性故障（必须修复才交付）
    hard = []
    if abs(b["achieved_gr_p90"] - target) > 0.75:
        hard.append(f"GR校准未达标 target={target} achieved={b['achieved_gr_p90']:.2f}dB")
    if abs(lag) > 1:
        hard.append(f"时间对齐 {lag} samples")
    if abs(pitch - orig_pitch) > 2.0:
        hard.append(f"音高异常 Δmedian={pitch - orig_pitch:.2f}Hz")
    if not all(np.isfinite(mix)):
        hard.append("非有限值")
    if jump_ratio > 10.0 or jump_ratio < 0.1:
        hard.append(f"样本跳变异常 jump_ratio={jump_ratio:.2f}")
    if silent_in_active > 0:
        hard.append(f"有声区异常静音 {silent_in_active} 帧")
    # 软性诊断标记（保留交付，页面标注）
    if b["frac_gr_gt_1db_active"] > 0.95 and b["gr_at_quiet_p10"] > 3.0:
        flags.append("持续过压缩倾向（>95%有声时间GR>1dB且安静区GR>3dB）")
    dyn_change_wet = orig_dyn["short_term_dynamic_range_db_p90_p10"] - b["wet_dyn_db"]
    if dyn_change_wet < 1.0:
        flags.append(f"压缩分支短时动态变化极小 Δ={dyn_change_wet:.2f}dB")
    if quiet_boost > 10.0:
        flags.append(f"气声/尾音放大风险 quiet_boost={quiet_boost:+.1f}dB")
    if not (0.75 <= centroid_ratio <= 1.33):
        flags.append(f"频谱意外染色 centroid_ratio={centroid_ratio:.2f}x")

    # 导出原始混合（仅在峰值可能削波时加常数保护增益，非限制器）
    guard = min(0.0, -0.5 - peak_db)
    rp = raw_root / f"{vid}.wav"
    export_wav(mix, sr, guard, rp)
    ye, esr = read_pcm16(rp)
    clipped = int(np.count_nonzero(np.abs(ye) >= 32767 / 32768))
    if clipped > 0:
        hard.append(f"导出削波 {clipped} samples")
    ye_loud = measure_loudness(ye, esr)

    records.append({
        "id": vid, "target_gr_db": target, "wet": wet,
        "threshold_db": b["threshold_db"], "makeup_db": b["makeup_db"],
        "calibration_converged": b["converged"], "calibration_history": b["calibration_history"],
        "achieved_gr_p90": b["achieved_gr_p90"],
        "gr_median_active": b["gr_median_active"], "gr_min_active": b["gr_min_active"],
        "gr_max_active": b["gr_max_active"], "gr_max_full": b["gr_max_full"],
        "frac_gr_gt_1db_active": b["frac_gr_gt_1db_active"],
        "gr_at_quiet_p10": b["gr_at_quiet_p10"],
        "wet_dyn_db": b["wet_dyn_db"], "dyn_change_wet_db": dyn_change_wet,
        "mix_dyn_db": dyn_mix,
        "dyn_change_mix_db": orig_dyn["short_term_dynamic_range_db_p90_p10"] - dyn_mix,
        "quiet_boost_db": quiet_boost, "crest_drop_db": crest_drop,
        "centroid_ratio": centroid_ratio, "flatness": sp["flatness"],
        "lag_samples": lag, "jump_ratio": jump_ratio, "pitch_median_hz": pitch,
        "raw_export_guard_db": guard,
        "raw_file": str(rp.relative_to(ROOT)).replace("\\", "/"), "raw_sha256": sha(rp),
        "raw_frames": len(ye), "raw_peak_dbfs": st["sample_peak_dbfs"],
        "raw_rms_dbfs": st["rms_dbfs"], "raw_lufs": ye_loud["integrated_lufs"],
        "raw_true_peak_dbtp": ye_loud["true_peak_dbtp"], "raw_lra": ye_loud["loudness_range_lu"],
        "raw_clipped_samples": clipped,
        "hard_fail": hard, "flags": flags, "deliverable": not hard,
        "diag_mark": bool(flags),
    })

# ---------------- 统一响度试听版（常数增益，无限制器） ----------------
loudness_all = {"original": orig_loud, **{r["id"]: {"integrated_lufs": r["raw_lufs"], "true_peak_dbtp": r["raw_true_peak_dbtp"]} for r in records}}
target = min(-18.0, *[m["integrated_lufs"] - 3.2 - m["true_peak_dbtp"] for m in loudness_all.values()])

listen_root = OUT / "listening/audio"
orig_matched = listen_root / "original.wav"
export_wav(x, sr, target - orig_loud["integrated_lufs"], orig_matched)
om, osr = read_pcm16(orig_matched)
om_loud = measure_loudness(om, osr)
if abs(om_loud["integrated_lufs"] - target) > 0.3:
    raise RuntimeError(f"original match error {om_loud['integrated_lufs']} vs {target}")

for r in records:
    if not r["deliverable"]:
        continue
    src = OUT / pathlib.Path(r["raw_file"]).relative_to("experiments/F1-EXP-008")
    ym, ysr = read_pcm16(src)
    m = measure_loudness(ym, ysr)
    gain = target - m["integrated_lufs"]
    dest = listen_root / f"{r['id']}.wav"  # 先存 P-ID，盲听复制在下一步
    export_wav(ym, ysr, gain, dest)
    de, dsr = read_pcm16(dest)
    dm = measure_loudness(de, dsr)
    if abs(dm["integrated_lufs"] - target) > 0.3:
        raise RuntimeError(f"{r['id']} match error {dm['integrated_lufs']} vs {target}")
    if dm["true_peak_dbtp"] > -2.9:
        raise RuntimeError(f"{r['id']} matched true peak too hot {dm['true_peak_dbtp']}")
    r["matched"] = {"gain_db": gain, "lufs": dm["integrated_lufs"],
                    "true_peak_dbtp": dm["true_peak_dbtp"], "frames": len(de), "sha256": sha(dest)}

# ---------------- 盲听映射（映射仅存 qa/，页面与终端输出不出现） ----------------
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
    badge = '<span class="badge">带工程诊断标记</span>' if next(r for r in records if r["id"] == blind[letter])["diag_mark"] else ""
    rows.append(f'<div class="row"><label>版本 {letter} {badge}</label><audio controls preload="metadata" src="audio/{letter}.wav"></audio></div>')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P3B 压缩机制盲听（A01）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px}}.badge{{background:#e9edf3;color:#4a5568;border-radius:4px;font-size:12px;padding:1px 6px;margin-left:6px}}</style>
<h1>F1-P3B · 压缩机制盲听（仅 A01）</h1>
<div class="note"><b>本轮登记（先于生成音频）：</b>原音问题 = A01 自评薄音（单薄、不够充实）。<br>
预期可听变化 = 元音与整句更充实、小声部分更实，大声不突兀，轻重对比仍在。<br>
否决副作用 = 每个字一样重/情绪消失、抽吸、气声或尾音异常放大、闷糊、被压住、失真；只变响或只变闷不算通过。<br>
<b>若你认为 A01 原音没有“薄”的问题，请直接说明，本轮记 not_applicable。</b></div>
<div class="card"><h3>试听说明</h3><p class="meta">先听原音，再逐个听版本 A/B/C/D。四个版本参数已隐藏（映射存于 qa/blind_key.json，建议试听后再看）。请分别判断：①有无变化；②饱满度是否更好；③是否自然、像本人；④有无上面列出的否决副作用。</p>
<div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div>{''.join(rows)}</div>
<p class="meta">仅 A01（13.14s）；Holdout 未处理；全部文件已统一响度至 {target:.2f} LUFS（常数增益，无限制器）。带“工程诊断标记”的版本为高风险档，请留意气声/尾音与抽吸感。</p></html>'''
(OUT / "listening/index.html").write_text(page, encoding="utf-8")

# ---------------- QA 与记录 ----------------
(OUT / "qa/blind_key.json").write_text(json.dumps({
    "seed": SEED, "mapping_letter_to_version": blind,
    "note": "试听完成前请勿查看本文件，以免破坏盲听",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

qa = {
    "experiment": "F1-EXP-008", "phase": "F1-P3B", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "sample_display": DISPLAY,
    "original": {**orig_stats, "lufs": orig_loud, "spectral": orig_spec, "pitch_median_hz": orig_pitch},
    "original_dynamics": orig_dyn,
    "target_lufs": target,
    "compressor": {
        "topology": "feed-forward",
        "detector": "dual-branch one-pole RMS on mean-square: attack branch tau=15ms, release branch tau=120ms, per-sample max (classic RMS ballistics envelope); level=10*log10(envelope)",
        "curve": f"soft knee width {KNEE_DB}dB centered at threshold (quadratic), above: GR={SLOPE:.2f}*(L-T)",
        "ratio": RATIO, "attack_ms": ATTACK_MS, "release_ms": RELEASE_MS,
        "time_constant_convention": "63% step response in the power (mean-square) domain; no extra detector window; no lookahead",
        "gr_definition": "frame-mean smoothed GR over active 40ms frames; target=P90",
        "active_definition": f"40ms frame RMS within {ACTIVE_FLOOR_DB}dB of max frame",
        "mix": "out = (1-wet)*dry + wet*compressed_branch, branch loudness-matched to dry by constant makeup gain before mixing",
    },
    "branches": {str(t): {k: v for k, v in b.items() if k not in ("y", "sm")} for t, b in branches.items()},
    "versions": records,
    "screen_rules": {
        "hard": ["|GR_P90 - target| > 0.75dB", "|lag| > 1 sample", "|Δpitch median| > 2Hz", "nonfinite", "jump_ratio outside [0.1,10]", "silent active frames", "export clipping"],
        "diag_flags": {"persistent_over_compression": "GR>1dB on >95% active frames AND quiet-region GR>3dB",
                        "wet_dyn_change_lt_1db": "compressed branch short-term dynamics changed <1dB",
                        "quiet_boost_gt_10db": "median boost on quietest 10% active frames > +10dB",
                        "centroid_outside_0.75_1.33": "unexpected spectral coloration"},
        "listening_loudness": "constant gain only, no limiter; all files within 0.3 LU of target",
    },
    "note": "客观变化（动态、GR、LUFS）不代表听感改善；听感结论由 Jade 判定",
}
(OUT / "qa/qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with open(OUT / "qa/levels.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "target_gr_db", "wet", "threshold_db", "makeup_db", "achieved_gr_p90", "gr_median_active", "frac_gr_gt_1db", "gr_at_quiet_p10", "dyn_change_wet_db", "dyn_change_mix_db", "quiet_boost_db", "crest_drop_db", "centroid_ratio", "lag", "raw_peak_dbfs", "raw_lufs", "raw_tp_dbtp", "raw_clipped", "hard_fail", "flags", "matched_lufs", "matched_tp_dbtp"])
    for r in records:
        m = r.get("matched", {})
        w.writerow([r["id"], r["target_gr_db"], r["wet"], round(r["threshold_db"], 2), round(r["makeup_db"], 2), round(r["achieved_gr_p90"], 2), round(r["gr_median_active"], 2), round(r["frac_gr_gt_1db_active"], 3), round(r["gr_at_quiet_p10"], 2), round(r["dyn_change_wet_db"], 2), round(r["dyn_change_mix_db"], 2), round(r["quiet_boost_db"], 2), round(r["crest_drop_db"], 2), round(r["centroid_ratio"], 3), r["lag_samples"], round(r["raw_peak_dbfs"], 2), round(r["raw_lufs"], 2), round(r["raw_true_peak_dbtp"], 2), r["raw_clipped_samples"], "; ".join(r["hard_fail"]), "; ".join(r["flags"]), m.get("lufs", ""), m.get("true_peak_dbtp", "")])

params = {
    "experiment": "F1-EXP-008", "phase": "F1-P3B",
    "question": "薄音能否通过动态压缩与并行混合获得可听、自然的密度改善（不做广度随机扫描）",
    "single_module": True, "no_stacking": True, "samples": [SAMPLE_ID],
    "sealed": SEALED,
    "compressor": qa["compressor"],
    "versions": [{"id": v[0], "target_gr_p90_active_db": v[1], "wet": v[2]} for v in VERSIONS],
    "gr_target_definition": "P90 of frame-mean gain reduction over active 40ms frames; threshold auto-calibrated per target (max 6 iterations, tol 0.2dB)",
    "p4_purpose": "观察强压缩边界，不预设更好听；若触发持续过压缩标记仍保留供诊断",
    "listening_target_lufs": target,
    "loudness_tolerance_lu": 0.3,
    "references": {
        "A_AES2022_attack_release_audibility": "https://aes.org/publications/elibrary/elibrary-page/?id=21958 — 不把精力放在细小 Attack/Release 扫描",
        "B_MDPI2020_aggressive_descriptor": "https://www.mdpi.com/2076-3417/10/7/2350 — 强压缩可能带来攻击性/粗糙感，故设 8dB 上界与诊断标记",
        "C_JAES2014_perceptual_compression": "https://aes.org/publications/elibrary/elibrary-page/?id=17084 — 压缩未必改善主观音质，全部响度匹配后盲听",
        "D_SMM2019_glottal_source": "https://www.isca-archive.org/smm_2019/cabral19_smm.html — 若压缩失效，后续转向声源/发声特性诊断而非继续调参",
        "disclaimer": "以上参数为工程实验设计，并非上述论文验证过的最佳组合",
    },
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "ffmpeg": "winget links"},
    "blind_seed": SEED,
}
(OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "experiment": "F1-EXP-008", "phase": "F1-P3B", "status": "pending_jade",
    "sample_id": SAMPLE_ID, "versions": [r["id"] for r in records],
    "deliverable": [r["id"] for r in records if r["deliverable"]],
    "hard_failed": [r["id"] for r in records if not r["deliverable"]],
    "diag_marked": [r["id"] for r in records if r["diag_mark"]],
    "target_lufs": target,
    "technical_pass": all(r["deliverable"] for r in records)
        and all(abs(r["lag_samples"]) <= 1 for r in records)
        and not any(r["raw_clipped_samples"] > 0 for r in records)
        and all(abs(r.get("matched", {}).get("lufs", target + 99) - target) <= 0.3 for r in records if r["deliverable"]),
    "preview_page": "listening/index.html", "blind_key": "qa/blind_key.json",
    "raw_outputs": "outputs/processed_raw",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run_log.txt").write_text(
    f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-008\nphase: F1-P3B\n"
    f"sample: {SAMPLE_ID} ({DATASET_ID})\nsealed: {', '.join(SEALED)}\ntarget_lufs: {target}\n"
    f"technical_pass: {summary['technical_pass']}\n"
    f"note: original inputs not modified; holdout not processed; single compressor module only; no limiter\n",
    encoding="utf-8")

marked = [r["id"] for r in records if r["diag_mark"]]
(OUT / "RESULT.md").write_text(
    "# F1-EXP-008 结果（pending_jade）\n\n"
    "F1-P3B 压缩机制验证：A01 单素材，RMS 向下压缩 4:1/15ms/120ms/软拐点6dB，"
    "阈值按有声段 GR P90 自动校准（4dB/8dB 两档），并行混合 Wet 50%/100% 共四版，全部统一响度盲听。\n\n"
    f"- 交付版本：{', '.join(r['id'] for r in records if r['deliverable'])}（匿名 A/B/C/D，映射见 qa/blind_key.json）\n"
    f"- 工程诊断标记：{', '.join(marked) if marked else '无'}\n"
    f"- 硬性故障：{', '.join(r['id'] for r in records if not r['deliverable']) or '无'}\n"
    "- 听感结论待 Jade 试听后填写；技术通过不代表听感改善。\n",
    encoding="utf-8")

print(json.dumps({
    "summary": summary,
    "original_dynamics": orig_dyn,
    "versions": [{k: r[k] for k in ("id", "target_gr_db", "wet", "threshold_db", "makeup_db", "achieved_gr_p90",
                                     "gr_median_active", "frac_gr_gt_1db_active", "dyn_change_wet_db",
                                     "dyn_change_mix_db", "quiet_boost_db", "crest_drop_db", "hard_fail", "flags")}
                 for r in records],
}, ensure_ascii=False, indent=1))
