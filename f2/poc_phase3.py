"""F2 Phase-3: 有参考修音(用自己"唱得好"的版本当参考)。

流程:
1. 严重版(目标) 与 正常版(参考) 做斜率约束 DTW 对齐
2. 逐音符算偏差: 目标音高 - 参考音高, 再减去整体调差 -> 相对偏差
3. 只修相对偏差 >100 音分(真正唱错的音), 修到参考音高
4. 执行: Praat Manipulation, 每个音的移调量拆成多段近似渐变, 避免接缝滑音
5. 导出 + 复测残留偏差 + 试听页
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from self_ref_align import (cents_track, fill, fix_oct, global_dtw, note_name,
                            notes_of)

HOP = 0.05
RAMP = 0.070          # 渐变时长(秒)
SEGMENTS = 9          # 每个音拆成的子区间数
FIX_THRESHOLD = 100.0 # 相对偏差超过此值才修(音分)
MAX_SHIFT = 600.0     # 单音最大修正量
PITCH_FLOOR, PITCH_CEIL = 75.0, 600.0


def build_mapping(n_frames, ref_frames, path):
    map_i = np.full(n_frames, np.nan)
    for i, j in path:
        map_i[i] = j
    ok = ~np.isnan(map_i)
    map_i[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], map_i[ok])
    return map_i


def apply_tapered_shift(sound, note_shifts):
    """对每个音符施加带渐变的移调。note_shifts: [(t0, t1, cents)]。"""
    manip = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    for t0, t1, cents in note_shifts:
        dur = t1 - t0
        ramp = min(RAMP, dur / 2.5) if dur > 0 else RAMP
        target = 2.0 ** (cents / 1200.0)
        n = max(3, SEGMENTS)
        for s in range(n):
            a = t0 + dur * s / n
            b = t0 + dur * (s + 1) / n
            mid = 0.5 * (a + b)
            # 三角窗: 首尾 0, 中间 1
            u = min(mid - t0, t1 - mid)
            w = min(1.0, u / ramp) if ramp > 0 else 1.0
            f = 1.0 + (target - 1.0) * w
            parselmouth.praat.call(tier, "Multiply frequencies", a, b, f)
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    out = parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values[0], dtype=np.float64)


def note_pitch_after(path_or_array, sr, notes):
    from poc_phase2 import hz_to_cents
    x = path_or_array
    res = []
    for n in notes:
        i0, i1 = int(n["t0"] * sr), int(n["t1"] * sr)
        seg = x[i0:i1]
        if len(seg) < 64:
            res.append(math.nan)
            continue
        s = parselmouth.Sound(seg, sr)
        p = parselmouth.praat.call(s, "To Pitch", 0.01, PITCH_FLOOR, PITCH_CEIL)
        v = parselmouth.praat.call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        res.append(hz_to_cents(v) if v and v > 0 else math.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="要修的版本(唱得差)")
    ap.add_argument("--reference", required=True, help="参考版本(唱得好)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    q = fill(fix_oct(cents_track(a.target)))
    r = fill(fix_oct(cents_track(a.reference)))
    path, cost = global_dtw(q, r)
    map_i = build_mapping(len(q), len(r), path)

    t_notes = notes_of(a.target)
    r_notes = notes_of(a.reference)

    rows = []
    for n in t_notes:
        i_c = int(round(0.5 * (n["t0"] + n["t1"]) / HOP))
        i_c = min(max(i_c, 0), len(map_i) - 1)
        rt_c = map_i[i_c] * HOP
        # 用中心时刻落点找"包含该时刻"的参考音符; 找不到退化为最近的音符中心
        hit = [m["cents"] for m in r_notes if m["t0"] <= rt_c <= m["t1"]]
        if hit:
            ref_c = float(np.median(hit))
        else:
            if not r_notes:
                continue
            nearest = min(r_notes, key=lambda m: abs(0.5 * (m["t0"] + m["t1"]) - rt_c))
            ref_c = float(nearest["cents"])
        rows.append({"t0": n["t0"], "t1": n["t1"], "hz": n["hz"], "cents": n["cents"],
                     "note": note_name(n["cents"]), "ref_cents": ref_c,
                     "ref_note": note_name(ref_c), "deviation": n["cents"] - ref_c})

    if not rows:
        raise SystemExit("无可对比音符")
    off = float(np.median([x["deviation"] for x in rows]))
    for x in rows:
        x["relative"] = x["deviation"] - off
    fix = [x for x in rows if abs(x["relative"]) > FIX_THRESHOLD]
    print(f"对齐代价 {cost:.0f} 音分/帧; 音符 {len(rows)}; 整体调差 {off:+.1f} 音分")
    print(f"待修 {len(fix)} 个音 (相对偏差 >{FIX_THRESHOLD:.0f} 音分)")

    shifts = []
    for x in fix:
        s = max(-MAX_SHIFT, min(MAX_SHIFT, -x["relative"]))
        shifts.append((x["t0"], x["t1"], s))
        x["applied"] = round(s, 1)
    for x in rows:
        x.setdefault("applied", 0.0)

    x_in, sr = read_wav_pcm16(Path(a.target))
    x_corr = apply_tapered_shift(parselmouth.Sound(x_in, sr), shifts)
    print(f"修正后样本数 {len(x_corr)} (原 {len(x_in)})")

    # 复测
    after = note_pitch_after(x_corr, sr, [{"t0": x["t0"], "t1": x["t1"]} for x in rows])
    for x, aft in zip(rows, after):
        x["after_cents"] = None if math.isnan(aft) else round(aft, 1)
        if x["applied"] != 0 and x["after_cents"] is not None:
            x["residual_after"] = round(x["after_cents"] - (x["cents"] + x["applied"]), 1)
        else:
            x["residual_after"] = None

    out.mkdir(parents=True)
    target_rms = rms_db(x_in)
    for name, arr in [("01_target_original.wav", x_in), ("02_corrected.wav", x_corr)]:
        g = 10 ** ((target_rms - rms_db(arr)) / 20.0)
        pk = peak_db(arr * g)
        if pk > -3.0:
            g *= 10 ** ((-3.0 - pk) / 20.0)
        write_wav_pcm16(out / name, arr * g, sr)

    # 参考版(独立导出, 供对照)
    x_ref, ref_sr = read_wav_pcm16(Path(a.reference))
    write_wav_pcm16(out / "03_reference_take.wav", x_ref, ref_sr)

    with open(out / "deviations.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["t0", "t1", "hz", "note", "cents", "ref_note", "ref_cents",
                  "deviation", "relative", "applied", "after_cents", "residual_after"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    res_vals = [abs(x["residual_after"]) for x in rows if x["residual_after"] is not None]
    summary = {
        "target": a.target, "reference": a.reference,
        "dtw_cost_cents": round(cost, 1),
        "global_offset_cents": round(off, 1),
        "notes": len(rows), "corrected_notes": len(fix),
        "correction_cents": [round(x["applied"], 1) for x in fix],
        "residual_after_cents": {"mean": round(float(np.mean(res_vals)), 1) if res_vals else None,
                                 "max": round(float(np.max(res_vals)), 1) if res_vals else None},
    }
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"复测残留(修正后实际音高 vs 意图音高): mean {summary['residual_after_cents']['mean']} "
          f"max {summary['residual_after_cents']['max']} 音分")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
