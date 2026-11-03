"""F2 Phase-4b: 在 phase4 正确对齐的基础上，减少"电音"并补上漏修的音。

相对 poc_phase4_realign.py 的改动（phase4 保持不变以便复现）：
1. 参考音高只取对齐区间中间 60% 的帧，离散度用 IQR（p75−p25）而非 p90−p10。
   原做法把相邻音/过渡帧算进参考，导致 zuixuan_mild 9.4–17.0s 一串约 −130 音分的偏低音全被判"参考不稳"。
2. 平滑移调：逐个音高点乘以 sin² 渐变权重（原 apply_tapered 分 9 段阶梯，边缘音高呈台阶，疑似电音来源之一）。
3. 局部替换：只在被修音符（含 40 ms 余量）处使用重合成音频，其余时间保留原始采样，20 ms 等功率交叉淡化。
4. 连续同向偏差的音合并为一组按组中位数平移（沿用 phase3d 的思路），避免相邻音一个修一个不修造成音程跳变。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from poc_phase4_realign import HOP, PITCH_CEIL, PITCH_FLOOR, align, cents_track, cost_matrix, subseq_dtw
from self_ref_align import note_name, notes_of

call = parselmouth.praat.call

FIX_THRESHOLD = 60.0
MAX_SHIFT = 200.0
REF_MAX_IQR = 80.0
CORE = 0.6
RAMP_S = 0.06
PAD_S = 0.04
XFADE_S = 0.02
GROUP_GAP_S = 0.25
KEY_TRIM = 70.0


def circ(v):
    return float((v + 600.0) % 1200.0 - 600.0)


def ref_pitch(r, js):
    if len(js) < 3:
        return None
    k = len(js)
    lo, hi = int(k * (1 - CORE) / 2), int(np.ceil(k * (1 + CORE) / 2))
    rc = r[js[lo:hi]]
    rc = rc[~np.isnan(rc)]
    if len(rc) < 3 or len(rc) < 0.6 * (hi - lo):
        return None
    seed = float(np.median(rc))
    rc = seed + (rc - seed + 600.0) % 1200.0 - 600.0
    return float(np.median(rc)), float(np.percentile(rc, 75) - np.percentile(rc, 25))


def group_fixes(rows):
    fixes = sorted([x for x in rows if x["status"] == "fix"], key=lambda x: x["t0"])
    groups = []
    for x in fixes:
        g = groups[-1] if groups else None
        if g and x["t0"] - g[-1]["t1"] <= GROUP_GAP_S and np.sign(x["deviation"]) == np.sign(g[0]["deviation"]):
            g.append(x)
        else:
            groups.append([x])
    out = []
    for g in groups:
        shift = -float(np.median([m["deviation"] for m in g]))
        for m in g:
            m["applied"] = round(shift, 1)
        out.append((g[0]["t0"], g[-1]["t1"], shift))
    return out


def smooth_shift(x, sr, spans):
    """spans: [(t0, t1, cents)]。逐点 sin² 渐变移调后返回整段重合成。"""
    sound = parselmouth.Sound(x, sr)
    manip = call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    pts = [(call(tier, "Get time from index", i), call(tier, "Get value at index", i)) for i in range(1, n + 1)]
    new = call("Create PitchTier", "smooth", 0.0, sound.get_total_duration())
    for t, f in pts:
        c = 0.0
        for t0, t1, cents in spans:
            if t0 <= t <= t1:
                ramp = min(RAMP_S, (t1 - t0) / 3)
                u = min(t - t0, t1 - t) / ramp if ramp > 0 else 1.0
                c = cents * np.sin(0.5 * np.pi * min(u, 1.0)) ** 2
        call(new, "Add point", t, f * 2.0 ** (c / 1200.0))
    call([manip, new], "Replace pitch tier")
    y = np.asarray(call(manip, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def splice(x, y, sr, spans):
    """只在 spans（加余量）内使用 y，等功率交叉淡化。"""
    w = np.zeros(len(x))
    fade = int(XFADE_S * sr)
    for t0, t1, _ in spans:
        a, b = max(0, int((t0 - PAD_S) * sr)), min(len(x), int((t1 + PAD_S) * sr))
        w[a:b] = 1.0
        ramp = np.sin(0.5 * np.pi * np.linspace(0, 1, fade)) ** 2
        a0 = max(0, a - fade)
        w[a0:a] = np.maximum(w[a0:a], ramp[fade - (a - a0):])
        b1 = min(len(x), b + fade)
        w[b:b1] = np.maximum(w[b:b1], ramp[::-1][: b1 - b])
    return np.sqrt(w) * y + np.sqrt(1 - w) * x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shift", type=float, default=MAX_SHIFT)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    q, r = cents_track(a.target), cents_track(a.reference)
    cost, path, key = align(q, r)
    base_cost, _ = subseq_dtw(cost_matrix(q, r[::-1].copy(), key))
    qmap = {}
    for i, j in path:
        qmap.setdefault(i, []).append(j)
    ref_t0, ref_t1 = path[0][1] * HOP, path[-1][1] * HOP

    rows = []
    for n in notes_of(a.target):
        i0, i1 = int(round(n["t0"] / HOP)), min(int(round(n["t1"] / HOP)), len(q) - 1)
        js = sorted({j for i in range(i0, i1 + 1) for j in qmap.get(i, [])})
        row = {"t0": n["t0"], "t1": n["t1"], "note": note_name(n["cents"]), "cents": n["cents"],
               "ref_t0": js[0] * HOP if js else None, "ref_t1": js[-1] * HOP if js else None}
        rp = ref_pitch(r, js)
        if rp is None:
            rows.append(row | {"status": "ref_unvoiced", "applied": 0.0})
            continue
        row |= {"ref_c": rp[0], "ref_iqr": round(rp[1], 1), "status": "pending"}
        rows.append(row)

    # 调差细化：截尾中位数迭代——跑调音本身会把中位数拉偏（受控测试实测偏 +18 音分），
    # 故第二轮起只用当前偏差 <KEY_TRIM 的音
    stable = [x for x in rows if x["status"] == "pending" and x["ref_iqr"] <= REF_MAX_IQR]
    for it in range(3):
        devs = np.array([circ(x["cents"] - x["ref_c"] - key) for x in stable])
        use = devs if it == 0 else devs[np.abs(devs) < KEY_TRIM]
        if len(use):
            key += float(np.median(use))
    for x in rows:
        if x["status"] != "pending":
            continue
        dev = circ(x["cents"] - x["ref_c"] - key)
        x |= {"ref_note": note_name(x["ref_c"] + key), "deviation": round(dev, 1), "applied": 0.0}
        if x["ref_iqr"] > REF_MAX_IQR:
            x["status"] = "ref_unstable"
        elif abs(dev) < FIX_THRESHOLD:
            x["status"] = "ok"
        elif abs(dev) > a.max_shift:
            x["status"] = "too_large_skip"
        else:
            x["status"] = "fix"
    spans = group_fixes(rows)
    print(f"对齐代价 {cost:.1f}/帧（反转基线 {base_cost:.1f}）; 调差 {key:+.1f} 音分; 参考 {ref_t0:.2f}-{ref_t1:.2f}s")
    for x in rows:
        print(f"  {x['t0']:5.2f}-{x['t1']:5.2f} {x['note']:>4} ref {x.get('ref_note', '-'):>4} "
              f"iqr {x.get('ref_iqr', float('nan')):6.1f} dev {x.get('deviation', float('nan')):+7.1f} {x['status']} {x['applied']:+.1f}")

    x_in, sr = read_wav_pcm16(Path(a.target))
    y = smooth_shift(x_in, sr, spans) if spans else x_in.copy()
    x_corr, w = splice(x_in, y, sr, spans)
    out.mkdir(parents=True)
    g = 10 ** ((-16.0 - rms_db(x_in)) / 20.0)
    g = min(g, 10 ** ((-3.0 - max(peak_db(x_in), peak_db(x_corr))) / 20.0))
    write_wav_pcm16(out / "01_target_original.wav", x_in * g, sr)
    write_wav_pcm16(out / "02_corrected.wav", x_corr * g, sr)
    x_ref, rsr = read_wav_pcm16(Path(a.reference))
    seg = x_ref[int(ref_t0 * rsr): int(ref_t1 * rsr) + 1]
    gr = min(10 ** ((-16.0 - rms_db(seg)) / 20.0), 10 ** ((-3.0 - peak_db(seg)) / 20.0))
    write_wav_pcm16(out / "03_reference_matched_segment.wav", seg * gr, rsr)

    with open(out / "notes.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["t0", "t1", "note", "ref_t0", "ref_t1", "ref_note", "ref_iqr", "deviation", "status", "applied"]
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "reference": a.reference, "align_cost_per_frame": round(cost, 1),
                   "reversed_ref_baseline_cost": round(base_cost, 1), "key_offset_cents": round(key, 1),
                   "ref_segment_s": [round(ref_t0, 2), round(ref_t1, 2)], "notes": len(rows),
                   "fixed_notes": sum(x["status"] == "fix" for x in rows),
                   "spans": [[round(t0, 2), round(t1, 2), round(c, 1)] for t0, t1, c in spans],
                   "resynth_fraction": round(float(np.mean(w > 0)), 3),
                   "params": {"fix_threshold": FIX_THRESHOLD, "max_shift": a.max_shift, "ref_max_iqr": REF_MAX_IQR,
                              "core": CORE, "ramp_s": RAMP_S, "pad_s": PAD_S, "xfade_s": XFADE_S,
                              "group_gap_s": GROUP_GAP_S, "key_trim": KEY_TRIM}}, f, ensure_ascii=False, indent=2)
    print(f"修正 {len(spans)} 组 / {sum(x['status'] == 'fix' for x in rows)} 音; 重合成占比 {np.mean(w > 0):.1%}; 输出: {out}")


if __name__ == "__main__":
    main()
