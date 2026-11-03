"""F2 Phase-3d: 乐句级平移修音。

与 3b 的区别: 不再逐个音单独修正, 而是把"时间连续 + 同方向偏差"的音归为一组,
整组按中位偏差平移, 保持组内音程关系不变, 只在组首尾渐变。
这样避免"旋律线上的台阶", 也不会产生音色突兀。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from poc_phase3b import align_notes, apply_tapered, note_name, notes_of

RAMP = 0.090
FIX_THRESHOLD = 90.0
GROUP_GAP = 0.40          # 相邻音间隔小于此值视为同一组
GROUP_SIGN_TOL = 0.5      # 同组要求偏差同号(允许低于阈值的成员)
MAX_SHIFT = 320.0


def build_groups(rows):
    """把连续、同方向的偏差归组; 返回 [(t0, t1, shift, members)]。"""
    rows = sorted(rows, key=lambda r: r["t0"])
    groups = []
    cur = []
    for r in rows:
        big = abs(r["relative"]) > FIX_THRESHOLD
        if not cur:
            if big:
                cur = [r]
            continue
        gap = r["t0"] - cur[-1]["t1"]
        same_sign = r["relative"] * cur[0]["relative"] > 0
        if big and same_sign and gap <= GROUP_GAP:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r] if big else []
    if cur:
        groups.append(cur)

    out = []
    for g in groups:
        # 组内成员: 从第一个大偏差到最后一个大偏差, 中间的小偏差音一并平移
        vals = [m["relative"] for m in g]
        shift = float(np.median(vals))
        shift = max(-MAX_SHIFT, min(MAX_SHIFT, shift))
        out.append({
            "t0": g[0]["t0"], "t1": g[-1]["t1"], "shift": -shift,
            "members": [(m["t0"], m["note"], m["ref_note"], round(m["relative"], 1)) for m in g],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    t = notes_of(a.target)
    r = notes_of(a.reference)
    pairs, unmatched = align_notes(t, r)
    rows = []
    for i, j in pairs:
        rows.append({"t0": t[i]["t0"], "t1": t[i]["t1"], "hz": t[i]["hz"],
                     "note": note_name(t[i]["cents"]), "ref_note": note_name(r[j]["cents"]),
                     "dev": t[i]["cents"] - r[j]["cents"]})
    if not rows:
        raise SystemExit("无配对")
    off = float(np.median([x["dev"] for x in rows]))
    for x in rows:
        x["relative"] = x["dev"] - off

    groups = build_groups(rows)
    print(f"配对 {len(rows)} 音 -> {len(groups)} 组")
    for g in groups:
        print(f"  {g['t0']:.2f}-{g['t1']:.2f}s 整组平移 {g['shift']:+.1f} 音分, "
              f"{len(g['members'])} 个音: " + ", ".join(f"{m[1]}@{m[0]:.2f}" for m in g["members"]))

    x_in, sr = read_wav_pcm16(Path(a.target))
    x_corr = apply_tapered(parselmouth.Sound(x_in, sr),
                           [(g["t0"], g["t1"], g["shift"]) for g in groups])

    out.mkdir(parents=True)
    tgt = rms_db(x_in)
    common = 10 ** ((-16.0 - tgt) / 20.0)
    for name, arr in [("01_target_original.wav", x_in), ("02_corrected.wav", x_corr)]:
        g = common
        if peak_db(arr * g) > -3.0:
            g *= 10 ** ((-3.0 - peak_db(arr * g)) / 20.0)
        write_wav_pcm16(out / name, arr * g, sr)
    x_ref, rsr = read_wav_pcm16(Path(a.reference))
    write_wav_pcm16(out / "03_reference_take.wav", x_ref, rsr)

    with open(out / "groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t0", "t1", "shift_cents", "n_notes", "notes"])
        for g in groups:
            w.writerow([round(g["t0"], 2), round(g["t1"], 2), round(g["shift"], 1),
                        len(g["members"]), ";".join(f"{m[1]}@{m[0]:.2f}" for m in g["members"])])
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "reference": a.reference, "paired": len(rows),
                   "groups": len(groups), "global_offset": round(off, 1),
                   "shifts": [round(g["shift"], 1) for g in groups]}, f, ensure_ascii=False, indent=2)
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
