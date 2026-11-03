"""F2 诊断轮: 音高听感阈值 + 全局调音 + 旋律拼写检查。

背景: shige_medium 修音候选按规则暂停(60%/100% 两档均听不出差别)。
目的: 定位"听不出"的原因, 区分三个假设:
  H1 听感阈值: 逐音修正(<=50音分)本身低于用户感知阈
  H2 目标错位: 跑调主体是整体调音偏移(当前算法故意保留), 而非逐音偏差
  H3 旋律错误: 唱错音(旋律级), 无参考算法修不了
输出: 原音 + 整体下移(全局偏移量) + 整体+50c + 整体+100c 四版对照 + 旋律音名表。
纯诊断轮, 不含任何修音处理。听感判定由用户完成。
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

from poc_phase1 import (PEAK_HEADROOM_DB, clip_count, f0_stats, peak_db, read_wav_pcm16,
                        rms_db, sha256_of, shift_uniform, write_wav_pcm16)

NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def note_name(f):
    """440Hz 参考下的最近音名(如 B3)与偏差音分。"""
    n = round(12.0 * math.log2(f / 440.0))
    cents = 1200.0 * math.log2(f / 440.0) - 100.0 * n
    return f"{NOTE_NAMES[n % 12]}{4 + (n + 9) // 12}", cents


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--notes-csv", required=True, help="上一轮 notes_table.csv, 用于旋律音名表")
    ap.add_argument("--global-offset-cents", type=float, required=True, help="实测整体调音偏移(正=偏高)")
    args = ap.parse_args()

    in_path, out_dir = Path(args.input), Path(args.output)
    if not in_path.is_file():
        raise SystemExit(f"输入不存在: {in_path}")
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out_dir}")

    x_in, sr = read_wav_pcm16(in_path)
    n_in = len(x_in)
    print(f"输入: {in_path}  sr={sr}  时长={n_in / sr:.3f}s  全局偏移: +{args.global_offset_cents} 音分")

    sound = parselmouth.Sound(x_in, sr)
    versions = {
        f"02_shift_minus{abs(args.global_offset_cents):g}c.wav": ("整体下移(全局调音归一, 无逐音修正)",
                                                                 shift_uniform(sound, -args.global_offset_cents)),
        "03_shift_plus50c.wav": ("整体上移 50 音分(阈值探测·小)", shift_uniform(sound, 50.0)),
        "04_shift_plus100c.wav": ("整体上移 100 音分(阈值探测·大, 一个半音)", shift_uniform(sound, 100.0)),
    }

    # 响度: 与原音共目标, 统一余量修正
    target_rms = rms_db(x_in)
    matched = {"01_original_matched.wav": x_in.copy()}
    for f, (_, x) in versions.items():
        matched[f] = np.asarray(x.values[0], dtype=np.float64)
    gains_db = {f: target_rms - rms_db(xv) for f, xv in matched.items()}
    peak_after = max(peak_db(xv * 10 ** (gains_db[f] / 20.0)) for f, xv in matched.items())
    trim = PEAK_HEADROOM_DB - peak_after if peak_after > PEAK_HEADROOM_DB else 0.0

    out_dir.mkdir(parents=True)
    shutil.copyfile(in_path, out_dir / "00_original_untouched.wav")
    for f, xv in matched.items():
        write_wav_pcm16(out_dir / f, xv * 10 ** ((gains_db[f] + trim) / 20.0), sr)

    # ---- 检查: 重读导出文件, 验证移调量 ----
    f0_ref = f0_stats(x_in, sr)["median_f0_hz"]
    report = {"phase": "F2-diagnosis threshold/global/melody",
              "input": {"path": str(in_path), "sha256": sha256_of(in_path), "sample_rate": sr, "samples": n_in},
              "params": {"global_offset_cents": args.global_offset_cents},
              "versions": {}, "gates": {}}
    expect = {"02": -args.global_offset_cents, "03": 50.0, "04": 100.0}
    for f in matched:
        x_chk, _ = read_wav_pcm16(out_dir / f)
        med = f0_stats(x_chk, sr)["median_f0_hz"]
        shift = 1200.0 * math.log2(med / f0_ref)
        report["versions"][f] = {"dur_dev_ms": round((len(x_chk) - n_in) / sr * 1000.0, 2),
                                 "clipped": clip_count(x_chk),
                                 "rms_dbfs": round(rms_db(x_chk), 2),
                                 "measured_shift_cents": round(shift, 2)}
        print(f"{f}: 移调实测 {shift:+.2f} 音分, RMS={rms_db(x_chk):+.2f}dB, "
              f"时长偏差 {report['versions'][f]['dur_dev_ms']}ms, 削波 {clip_count(x_chk)}")
    rms_vals = [v["rms_dbfs"] for v in report["versions"].values()]
    gates = {
        "duration_dev_within_10ms": all(v["dur_dev_ms"] <= 10.0 for v in report["versions"].values()),
        "no_clipping": all(v["clipped"] == 0 for v in report["versions"].values()),
        "loudness_sane_between_-30_and_0db": all(-30.0 <= r <= 0.0 for r in rms_vals),
        "matched_rms_equal_within_0.5db": max(rms_vals) - min(rms_vals) <= 0.5,
        "shift_accuracy_within_5c": all(
            abs(report["versions"][f]["measured_shift_cents"] - e) <= 5.0
            for f, e in [(k, expect[k[0:2]]) for k in report["versions"] if k != "01_original_matched.wav"]),
    }
    report["gates"] = gates
    report["technical_pass"] = all(gates.values())

    # ---- 旋律音名表(来自上一轮 notes_table) ----
    rows = []
    with open(args.notes_csv, encoding="utf-8") as fcsv:
        for r in csv.DictReader(fcsv):
            f_hz = float(r["median_f0_hz"])
            nm, off = note_name(f_hz)
            rows.append({"time": f"{float(r['t0_s']):.2f}-{float(r['t1_s']):.2f}s", "f0_hz": r["median_f0_hz"],
                         "note": nm, "cents_off": round(off, 1),
                         "residual_vs_your_key": r["residual_cents"],
                         "corrected_last_round": r["corrected"]})
    with open(out_dir / "melody_table.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "qa_report.json", "w", encoding="utf-8") as fjson:
        json.dump(report, fjson, ensure_ascii=False, indent=2)

    print("\n工程门槛:")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"旋律音名表: {len(rows)} 音 -> {out_dir / 'melody_table.csv'}")
    sys.exit(0 if report["technical_pass"] else 1)


if __name__ == "__main__":
    main()
