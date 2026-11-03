"""F2 Phase-2 PoC: 无参考保守修音决策层 + PSOLA 执行。

输入: 单声道 PCM16 WAV 清唱(存在音准问题的素材)
决策: Praat F0 逐帧 -> 声音符分段 -> 稳定音符(>=150ms, 帧间展开<=150音分)
      -> 估算整体调音偏移(圆均值, 不强迫唱在标准音高上)
      -> 对残差>=15音分的音符按 60% 强度拉向最近半音(颤音轮廓保留, 仅整体平移)
执行: 复用 Phase-1 已验证的 Manipulation(TD-PSOLA) 链, 逐音符 Multiply frequencies。
输出(新目录, 拒绝覆盖):
  00_original_untouched.wav / 01_original_matched.wav / 02_corrected.wav
  notes_table.csv (全部音符明细, 供核对) / qa_report.json
不含模型训练; 不宣称完整纠跑调, 只验证保守微修。听感验收由人完成。
"""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import (PEAK_HEADROOM_DB, band_energies_db, clip_count, peak_db,
                        read_wav_pcm16, rms_db, sha256_of, write_wav_pcm16)

PITCH_FLOOR = 75.0
PITCH_CEIL = 600.0
FRAME_STEP = 0.01
MIN_NOTE_DUR = 0.150      # 短于此的音符不处理
SPLIT_CENTS = 70.0         # 平滑轮廓偏离当前音符中位数超过此值 -> 切分新音符
MAX_SPREAD_CENTS = 150.0   # 段内 p90-p10 超过此值视为滑音/不稳定, 不处理
DEADBAND_CENTS = 15.0     # 残差小于此不动(避免过度处理)
STRENGTH = 0.60           # 收敛强度(默认; CLI 可覆盖)
BANDS_HZ = [(0, 300), (300, 1000), (1000, 3500), (3500, 8000), (8000, 22050)]


def wrap50(x):
    """把音分数折到 [-50, 50)。"""
    m = ((x % 100.0) + 100.0) % 100.0
    return m - 100.0 if m >= 50.0 else m


def hz_to_cents(f):
    return 1200.0 * math.log2(f / 440.0)


def pitch_series(x, sr):
    """Praat 逐帧 F0 -> (times, values)。value 为 nan 表示无声帧。"""
    p = parselmouth.praat.call(parselmouth.Sound(x, sr), "To Pitch", FRAME_STEP, PITCH_FLOOR, PITCH_CEIL)
    n = parselmouth.praat.call(p, "Get number of frames")
    ts, vs = [], []
    for i in range(1, n + 1):
        ts.append(parselmouth.praat.call(p, "Get time from frame number", i))
        v = parselmouth.praat.call(p, "Get value in frame", i, "Hertz")
        vs.append(v if v == v and v > 0 else math.nan)
    return np.array(ts), np.array(vs)


def analyze_notes(x, sr):
    """声音符切分: 先按有声段分组, 再在段内按平滑轮廓的音高跳变切音符。

    连唱时音符之间并不停声, 故不能只用有声/无声边界;
    用中值滤波(~1个颤音周期)压制颤音后, 轮廓偏离当前音符中位数超过阈值即切分。
    """
    times, vals = pitch_series(x, sr)
    notes = []
    idx = np.where(~np.isnan(vals))[0]
    if len(idx) == 0:
        return notes
    # 按无声间隙分组
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for run in runs:
        if len(run) < 3:
            continue
        cents = np.array([hz_to_cents(vals[i]) for i in run])
        # 平滑: 中值滤波窗口约 130ms, 压颤音
        w = max(3, int(round(0.13 / FRAME_STEP)) | 1)
        sm = np.array([np.median(cents[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(cents))])
        # 单遍生长: 平滑值偏离当前音符中位数 > SPLIT_CENTS 则切分
        start = 0
        for i in range(1, len(cents)):
            if i - start >= 3 and abs(sm[i] - np.median(cents[start:i + 1])) > SPLIT_CENTS:
                notes.append((times[run[start]], times[run[i]], vals[run[start:i]].copy()))
                start = i
        notes.append((times[run[start]], times[run[-1]], vals[run[start:]].copy()))
    return notes


def circular_mean_cents(values):
    a = [2.0 * math.pi * v / 100.0 for v in values]
    return math.atan2(sum(math.sin(v) for v in a) / len(a), sum(math.cos(v) for v in a) / len(a)) * 100.0 / (2.0 * math.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strength", type=float, default=STRENGTH, help="收敛强度 0~1, 默认 0.60")
    ap.add_argument("--deadband", type=float, default=DEADBAND_CENTS, help="死区音分, 默认 15")
    args = ap.parse_args()
    strength = max(0.0, min(1.0, args.strength))
    deadband = max(0.0, args.deadband)

    in_path, out_dir = Path(args.input), Path(args.output)
    if not in_path.is_file():
        raise SystemExit(f"输入不存在: {in_path}")
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out_dir}")

    x_in, sr = read_wav_pcm16(in_path)
    n_in = len(x_in)
    print(f"输入: {in_path}  sr={sr}  时长={n_in / sr:.3f}s")

    # ---- 决策层 ----
    notes = []
    for t0, t1, f0s in analyze_notes(x_in, sr):
        dur = t1 - t0
        cents = np.array([hz_to_cents(f) for f in f0s])
        spread = float(np.percentile(cents, 90) - np.percentile(cents, 10))
        stable = dur >= MIN_NOTE_DUR and spread <= MAX_SPREAD_CENTS
        notes.append({"t0": t0, "t1": t1, "dur": dur, "median_f0": float(np.median(f0s)),
                      "e": wrap50(hz_to_cents(float(np.median(f0s)))), "spread": spread, "stable": stable})
    stable_notes = [n for n in notes if n["stable"]]
    print(f"声音符段: {len(notes)} 段, 其中稳定音符: {len(stable_notes)} 段")

    if len(stable_notes) < 3:
        raise SystemExit(f"稳定音符不足({len(stable_notes)}), 偏移估计不可靠, 终止")

    offset = circular_mean_cents([n["e"] for n in stable_notes])
    for n in stable_notes:
        n["residual"] = wrap50(n["e"] - offset)
        n["shift"] = -strength * n["residual"] if abs(n["residual"]) >= deadband else 0.0
    corrected = [n for n in stable_notes if n["shift"] != 0.0]
    if not corrected:
        raise SystemExit(f"整体偏移 {offset:+.1f} 音分, 但无残差>={deadband} 音分的音符, 本素材不适用")
    print(f"整体调音偏移: {offset:+.1f} 音分; 修正音符: {len(corrected)}/{len(stable_notes)} "
          f"(强度 {strength:.0%}, 死区 {deadband:.0f} 音分)")

    # ---- 执行层: 复用 Phase-1 已验证的 PSOLA 链 ----
    sound = parselmouth.Sound(x_in, sr)
    manip = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    for n in corrected:
        parselmouth.praat.call(tier, "Multiply frequencies", n["t0"], n["t1"], 2.0 ** (n["shift"] / 1200.0))
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    x_corr = np.asarray(parselmouth.praat.call(manip, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)

    # 零修改对照(测量噪声抵消用, 不导出): 逐音符 F0 漂移在零修改重合成下即可达数音分,
    # 属跟踪器噪声而非真实泄漏(2026-11-01 shige 对照实验: 同一无修正音符漂 4.94 音分)。
    manip_pass = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    x_pass = np.asarray(parselmouth.praat.call(manip_pass, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)
    t_p, v_p = pitch_series(x_pass, sr)
    t_o, v_o = pitch_series(x_in, sr)

    # ---- 响度匹配(同 Phase-1 规则) + 导出 ----
    target_rms = rms_db(x_in)
    matched = {"01_original_matched.wav": x_in.copy(), "02_corrected.wav": x_corr}
    gains_db = {f: target_rms - rms_db(xv) for f, xv in matched.items()}
    peak_after = max(peak_db(xv * 10 ** (gains_db[f] / 20.0)) for f, xv in matched.items())
    common_trim_db = PEAK_HEADROOM_DB - peak_after if peak_after > PEAK_HEADROOM_DB else 0.0
    gains_db = {f: g + common_trim_db for f, g in gains_db.items()}
    out_dir.mkdir(parents=True)
    shutil.copyfile(in_path, out_dir / "00_original_untouched.wav")
    for f, xv in matched.items():
        write_wav_pcm16(out_dir / f, xv * 10 ** (gains_db[f] / 20.0), sr)

    # ---- 工程检查: 导出文件实测 + 零修改对照做差中差(抵消跟踪器逐音符噪声) ----
    x_out, _ = read_wav_pcm16(out_dir / "02_corrected.wav")
    t_out, v_out = pitch_series(x_out, sr)

    def window_median(times, vals, t0, t1):
        m = (times >= t0) & (times <= t1) & ~np.isnan(vals)
        return float(np.median(vals[m])) if m.any() else math.nan

    rows, err_c, err_u, pass_drift = [], [], [], []
    for n in stable_notes:
        med_old = window_median(t_o, v_o, n["t0"], n["t1"])
        med_new = window_median(t_out, v_out, n["t0"], n["t1"])
        med_pass = window_median(t_p, v_p, n["t0"], n["t1"])
        if math.isnan(med_old) or math.isnan(med_new) or math.isnan(med_pass):
            continue
        measured = hz_to_cents(med_new) - hz_to_cents(med_old)
        baseline = hz_to_cents(med_pass) - hz_to_cents(med_old)  # 零修改对照的固有漂移
        did = measured - baseline  # 差中差: 剔除跟踪噪声后的真实音高变化
        pass_drift.append(baseline)
        (err_c if n["shift"] != 0.0 else err_u).append(did - n["shift"] if n["shift"] != 0.0 else did)
        rows.append({
            "t0_s": round(n["t0"], 3), "t1_s": round(n["t1"], 3), "dur_s": round(n["dur"], 3),
            "median_f0_hz": round(n["median_f0"], 1), "cents_from_semitone": round(n["e"], 1),
            "residual_cents": round(n["residual"], 1), "applied_shift_cents": round(n["shift"], 1),
            "measured_shift_cents": round(measured, 1),
            "measured_shift_did_cents": round(did, 1),
            "corrected": n["shift"] != 0.0,
        })
    with open(out_dir / "notes_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x_orig, _ = read_wav_pcm16(out_dir / "01_original_matched.wav")
    bands_ref = band_energies_db(x_orig, sr)
    bands_out = band_energies_db(x_out, sr)
    band_diff = [round(b - r, 2) for b, r in zip(bands_out, bands_ref)]
    gates = {
        "corrected_notes_present": len(corrected) >= 1,
        "duration_dev_within_10ms": abs(len(x_out) - n_in) / sr * 1000.0 <= 10.0,
        "no_clipping": clip_count(x_out) == 0,
        "corrected_note_shift_error_mean_le_8c": len(err_c) > 0 and float(np.mean(np.abs(err_c))) <= 8.0,
        "corrected_note_shift_error_max_le_15c": len(err_c) > 0 and float(np.max(np.abs(err_c))) <= 15.0,
        "uncorrected_note_leakage_le_8c": len(err_u) == 0 or float(np.max(np.abs(err_u))) <= 8.0,
        "band_diff_vs_original_within_2dB": max(abs(d) for d in band_diff) <= 2.0,
    }
    report = {
        "phase": "F2-Phase2 reference-free conservative correction",
        "input": {"path": str(in_path), "sha256": sha256_of(in_path), "sample_rate": sr, "samples": n_in},
        "params": {"strength": strength, "deadband_cents": deadband, "min_note_dur_s": MIN_NOTE_DUR,
                   "max_spread_cents": MAX_SPREAD_CENTS, "pitch_floor_hz": PITCH_FLOOR, "pitch_ceiling_hz": PITCH_CEIL},
        "decision": {"total_notes": len(notes), "stable_notes": len(stable_notes), "corrected_notes": len(corrected),
                     "global_offset_cents": round(offset, 1),
                     "shift_applied_cents": [round(n["shift"], 1) for n in corrected]},
        "execution_check": {"duration_dev_ms": round((len(x_out) - n_in) / sr * 1000.0, 2),
                            "clipped_samples": clip_count(x_out),
                            "band_diff_vs_original_db": band_diff,
                            "methodology": "修正误差与泄漏均为差中差(修正版漂移-零修改对照漂移), 抵消跟踪器逐音符噪声",
                            "passthrough_control_drift_cents": {"mean": round(float(np.mean(np.abs(pass_drift))), 2),
                                                                "max": round(float(np.max(np.abs(pass_drift))), 2)},
                            "corrected_shift_error_cents": [round(e, 1) for e in err_c],
                            "uncorrected_shift_leak_cents": [round(e, 1) for e in err_u]},
        "gates": gates,
        "technical_pass": all(gates.values()),
        "listening": "pending_human (听感验收由用户完成)",
    }
    with open(out_dir / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"修正明细: {[(r['t0_s'], r['applied_shift_cents']) for r in rows if r['corrected']]}")
    print(f"修正误差DiD(音分): mean={float(np.mean(np.abs(err_c))):.2f} max={float(np.max(np.abs(err_c))):.2f}; "
          f"未动音符泄漏DiD max={float(np.max(np.abs(err_u))) if err_u else 0:.2f}; "
          f"对照固有漂移 max={float(np.max(np.abs(pass_drift))):.2f}; 带差={band_diff}")
    print("\n工程门槛:")
    for k, val in gates.items():
        print(f"  {'PASS' if val else 'FAIL'}  {k}")
    print(f"\n结论: {'technical_pass' if report['technical_pass'] else 'TECHNICAL_FAIL'} (听感 pending_human)")
    print(f"输出目录: {out_dir}")
    sys.exit(0 if report["technical_pass"] else 1)


if __name__ == "__main__":
    main()
