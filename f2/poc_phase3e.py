"""F2 Phase-3e: 有参考修音（绝对调对准 + 发声强度匹配 + 乐句级平移）。

相对 Phase-3d 的改动:
1. 绝对对准: 先整段平移 -global_offset, 让整体调与原唱一致(不再是"保留你的调")
2. 发声强度: 每个修正组按原唱对应段的 RMS 匹配增益, 限幅 ±4dB, 边界 150ms 渐变
3. 修正粒度不变: 只修偏差大的音, 且整组平移
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from poc_phase2 import hz_to_cents
from poc_phase3b import align_notes, apply_tapered, note_name, notes_of

FIX_THRESHOLD = 90.0
GROUP_GAP = 0.40
MAX_SHIFT = 320.0
MAX_GAIN_DB = 3.0
GAIN_RAMP = 0.15


def build_groups(rows):
    rows = sorted(rows, key=lambda r: r["t0"])
    groups, cur = [], []
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
        shift = float(np.median([m["relative"] for m in g]))
        shift = max(-MAX_SHIFT, min(MAX_SHIFT, shift))
        out.append({"t0": g[0]["t0"], "t1": g[-1]["t1"], "shift": -shift,
                    "ref_t0": min(m["ref_t0"] for m in g),
                    "ref_t1": max(m["ref_t1"] for m in g),
                    "members": [(m["t0"], m["note"], m["ref_note"]) for m in g]})
    return out


def seg_rms(x, sr, t0, t1):
    a, b = int(t0 * sr), int(t1 * sr)
    a, b = max(0, a), min(len(x), b)
    if b - a < 64:
        return None
    return rms_db(x[a:b])


def build_gain_curve(n, sr, segments):
    """segments: [(t0, t1, gain_db)] -> 平滑增益曲线。"""
    g = np.ones(n)
    for t0, t1, gdb in segments:
        a, b = int(t0 * sr), int(t1 * sr)
        a, b = max(0, a), min(n, b)
        if b <= a:
            continue
        r = int(GAIN_RAMP * sr)
        f = 10 ** (gdb / 20.0)
        g[a:b] = f
        # 两端渐变
        if r > 0:
            pre = min(r, a)
            g[a - pre:a] = np.linspace(1.0, f, pre) if pre else g[a - pre:a]
            post = min(r, n - b)
            g[b:b + post] = np.linspace(f, 1.0, post) if post else g[b:b + post]
    return g


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
    rnotes = notes_of(a.reference)
    pairs, _ = align_notes(t, rnotes)
    rows = []
    for i, j in pairs:
        rows.append({"t0": t[i]["t0"], "t1": t[i]["t1"],
                     "note": note_name(t[i]["cents"]), "ref_note": note_name(rnotes[j]["cents"]),
                     "ref_t0": rnotes[j]["t0"], "ref_t1": rnotes[j]["t1"],
                     "dev": t[i]["cents"] - rnotes[j]["cents"]})
    if not rows:
        raise SystemExit("无配对")
    off = float(np.median([x["dev"] for x in rows]))
    for x in rows:
        x["relative"] = x["dev"] - off
    print(f"配对 {len(rows)} 音; 整体调差 {off:+.1f} 音分 -> 将整段平移 {-off:+.1f} 音分对齐原唱")

    groups = build_groups(rows)
    print(f"乐句组 {len(groups)} 个")

    x_in, sr = read_wav_pcm16(Path(a.target))
    x_ref, rsr = read_wav_pcm16(Path(a.reference))

    # 1) 绝对对准(整段) + 2) 组级相对修正
    all_shifts = [(0.0, len(x_in) / sr, -off)]
    all_shifts += [(g["t0"], g["t1"], g["shift"]) for g in groups]
    x_key = apply_tapered(parselmouth.Sound(x_in, sr), all_shifts)

    # 3) 发声强度匹配(仅修正组): 相对强弱, 先扣掉双方的全局电平差, 再限幅 ±3dB
    u_all = seg_rms(x_key, sr, 0.0, len(x_key) / sr)
    r_all = seg_rms(x_ref, rsr, 0.0, len(x_ref) / rsr)
    base = 0.0 if (u_all is None or r_all is None) else (r_all - u_all)
    gains = []
    for g in groups:
        u = seg_rms(x_key, sr, g["t0"], g["t1"])
        rr = seg_rms(x_ref, rsr, g["ref_t0"], g["ref_t1"])
        if u is None or rr is None:
            continue
        d = float(np.clip((rr - u) - base, -MAX_GAIN_DB, MAX_GAIN_DB))
        gains.append((g["t0"], g["t1"], d))
        print(f"  组 {g['t0']:.2f}-{g['t1']:.2f}s: 相对强度 {d:+.2f}dB")
    curve = build_gain_curve(len(x_key), sr, gains)
    x_corr = x_key * curve

    out.mkdir(parents=True)
    common = 10 ** ((-16.0 - rms_db(x_in)) / 20.0)
    for name, arr in [("01_target_original.wav", x_in), ("02_corrected.wav", x_corr)]:
        g = common
        if peak_db(arr * g) > -3.0:
            g *= 10 ** ((-3.0 - peak_db(arr * g)) / 20.0)
        write_wav_pcm16(out / name, arr * g, sr)
    g2 = common
    if peak_db(x_ref * g2) > -3.0:
        g2 *= 10 ** ((-3.0 - peak_db(x_ref * g2)) / 20.0)
    write_wav_pcm16(out / "03_reference_aligned.wav", x_ref * g2, rsr)

    with open(out / "corrections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t0", "t1", "pitch_shift_cents", "gain_db", "notes"])
        for g in groups:
            gd = next((d for t0, t1, d in gains if abs(t0 - g["t0"]) < 0.01), 0.0)
            w.writerow([round(g["t0"], 2), round(g["t1"], 2), round(g["shift"], 1),
                        round(gd, 2), ";".join(f"{m[1]}@{m[0]:.2f}" for m in g["members"])])
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "reference": a.reference,
                   "paired_notes": len(rows), "global_offset_cents": round(off, 1),
                   "key_aligned": True, "groups": len(groups),
                   "pitch_shifts": [round(g["shift"], 1) for g in groups],
                   "gain_db": [round(d, 2) for _, _, d in gains]}, f, ensure_ascii=False, indent=2)
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
