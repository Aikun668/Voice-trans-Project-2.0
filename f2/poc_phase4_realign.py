"""F2 Phase-4: 重新对齐的有参考修音（自然版）。

P3b/P3d/P3e 的问题（2026-11-02 诊断）：音符配对用"音高差 + 相对位置"的 Needleman-Wunsch，
目标 13.6s 与参考 25s 不是同一段歌词长度，配对横跨 1.9→8s 等不可能的位置，
90–230 音分的"偏差"大多是配错音，不是跑调。

本版：
1. 帧级子序列 DTW（参考起止自由），步型 (1,1)(1,2)(2,1) 限制语速比 0.5–2；
2. 音高距离按 1200 音分取模（容忍参考轨八度误判），同时搜索整体调差 k（不要求用户和原唱同调）；
3. 只对"稳定长音 + 参考对应段也稳定 + 偏差超过阈值"的音整音平移，音内颤音/滑音形状保留；
4. 不改整体调、不做强度匹配（P3e 的两项都可能造成不自然）。

--align pitch（默认）：用音高轮廓对齐，适合参考为原唱（演唱者不同、音色不同）。
--align mfcc：用音色/音素（MFCC）对齐，与音高无关，适合同一人同歌词的自参考；
  跑调严重的版本用音高对齐会自相矛盾（实测 shige/zuixuan severe 对齐代价接近反转基线）。
  调差 k 取对齐帧上音高差的中位数。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from poc_phase3b import apply_tapered
from self_ref_align import note_name, notes_of

HOP = 0.02
PITCH_FLOOR, PITCH_CEIL = 75.0, 600.0
UNVOICED_COST = 150.0
DIST_CAP = 400.0
KEY_STEP = 10
FIX_THRESHOLD = 60.0       # 偏差小于此不修（用户在旋律中的分辨阈约 100+ 音分）
MAX_SHIFT = 150.0          # 超过多半是对齐/八度问题，不修
REF_MAX_SPREAD = 120.0     # 参考对应段 p90-p10 超过此值视为滑音段，不作为目标
STRENGTH = 1.0


def cents_track(path):
    x, sr = read_wav_pcm16(Path(path))
    p = parselmouth.Sound(x, sr).to_pitch(HOP, PITCH_FLOOR, PITCH_CEIL).selected_array["frequency"]
    c = np.full(len(p), np.nan)
    c[p > 0] = 1200 * np.log2(p[p > 0] / 440.0)
    return c


def mfcc_track(path):
    x, sr = read_wav_pcm16(Path(path))
    m = parselmouth.Sound(x, sr).to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=HOP)
    return m.to_array()[1:].T          # 去掉 c0（能量），保留音色/音素


def mfcc_cost(a, b):
    both = np.vstack([a, b])
    mu, sd = both.mean(axis=0), both.std(axis=0) + 1e-9
    a, b = (a - mu) / sd, (b - mu) / sd
    d = np.sqrt(np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0.0))
    return d


def key_from_path(q, r, path):
    d = np.array([(q[i] - r[j] + 600.0) % 1200.0 - 600.0 for i, j in path
                  if not (np.isnan(q[i]) or np.isnan(r[j]))])
    return float(np.median(d))


def cost_matrix(q, r, k):
    d = (q[:, None] - r[None, :] - k + 600.0) % 1200.0 - 600.0
    c = np.minimum(np.abs(d), DIST_CAP)
    qv, rv = ~np.isnan(q)[:, None], ~np.isnan(r)[None, :]
    c = np.where(qv & rv, c, np.where(qv | rv, UNVOICED_COST, 0.0))
    return c


def subseq_dtw(C):
    """D[i,j] = C + min(D[i-1,j-1], D[i-1,j-2], D[i-2,j-1])；第 0 行任意起点。返回 (平均代价, 路径)。"""
    n, m = C.shape
    INF = 1e18
    D = np.full((n, m), INF)
    B = np.zeros((n, m), dtype=np.int8)
    D[0] = C[0]
    for i in range(1, n):
        cand = np.full((3, m), INF)
        cand[0, 1:] = D[i - 1, :-1]
        cand[1, 2:] = D[i - 1, :-2]
        if i >= 2:
            cand[2, 1:] = D[i - 2, :-1] + C[i - 1, 1:]
        b = np.argmin(cand, axis=0)
        D[i] = C[i] + cand[b, np.arange(m)]
        B[i] = b
    j = int(np.argmin(D[-1]))
    total = D[-1, j]
    i, path = n - 1, []
    while i > 0:
        path.append((i, j))
        b = B[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i, j = i - 1, j - 2
        else:
            path.append((i - 1, j - 1))
            i, j = i - 2, j - 1
        if j < 0:
            break
    path.append((0, max(j, 0)))
    path.reverse()
    return total / n, path


def align(q, r):
    best = None
    for k in range(-600, 600, KEY_STEP):
        cost, path = subseq_dtw(cost_matrix(q, r, float(k)))
        if best is None or cost < best[0]:
            best = (cost, path, float(k))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=STRENGTH)
    ap.add_argument("--align", choices=["pitch", "mfcc"], default="pitch")
    ap.add_argument("--max-shift", type=float, default=MAX_SHIFT)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    q, r = cents_track(a.target), cents_track(a.reference)
    if a.align == "pitch":
        cost, path, key = align(q, r)
        # 对照：把参考时间反转后的代价，作为"对不上"的基线
        base_cost, _ = subseq_dtw(cost_matrix(q, r[::-1].copy(), key))
    else:
        mq, mr = mfcc_track(a.target), mfcc_track(a.reference)
        n = min(len(q), len(mq))
        q, mq = q[:n], mq[:n]
        m = min(len(r), len(mr))
        r, mr = r[:m], mr[:m]
        cost, path = subseq_dtw(mfcc_cost(mq, mr))
        base_cost, _ = subseq_dtw(mfcc_cost(mq, mr[::-1].copy()))
        key = key_from_path(q, r, path)
    qmap = {}
    for i, j in path:
        qmap.setdefault(i, []).append(j)
    ref_t0, ref_t1 = path[0][1] * HOP, path[-1][1] * HOP
    print(f"对齐代价 {cost:.1f}/帧（反转参考基线 {base_cost:.1f}）; 调差 k={key:+.0f} 音分; "
          f"参考区间 {ref_t0:.2f}-{ref_t1:.2f}s")

    rows = []
    for n in notes_of(a.target):
        i0, i1 = int(round(n["t0"] / HOP)), min(int(round(n["t1"] / HOP)), len(q) - 1)
        js = sorted({j for i in range(i0, i1 + 1) for j in qmap.get(i, [])})
        rc = r[js] if js else np.array([])
        rc = rc[~np.isnan(rc)]
        row = {"t0": n["t0"], "t1": n["t1"], "note": note_name(n["cents"]),
               "ref_t0": js[0] * HOP if js else None, "ref_t1": js[-1] * HOP if js else None,
               "ref_voiced": round(len(rc) / max(len(js), 1), 2)}
        if len(rc) < 3 or row["ref_voiced"] < 0.6:
            rows.append(row | {"status": "ref_unvoiced", "applied": 0.0})
            continue
        # 参考轨常有八度误判：围绕中位数按 1200 音分折叠后再取中位数与离散度
        seed = float(np.median(rc))
        rcf = seed + (rc - seed + 600.0) % 1200.0 - 600.0
        ref_c = float(np.median(rcf))
        spread = float(np.percentile(rcf, 90) - np.percentile(rcf, 10))
        row |= {"ref_c": ref_c, "cents": n["cents"], "ref_spread": round(spread, 1)}
        rows.append(row | {"status": "pending"})

    def circ(v):
        return float((v + 600.0) % 1200.0 - 600.0)

    # 调差二次细化：用稳定音的音级偏差中位数修正 k（DTW 搜索步长 10 音分，且少数跑调音会把 k 带偏）
    stable = [x for x in rows if x["status"] == "pending" and x["ref_spread"] <= REF_MAX_SPREAD]
    if stable:
        key += float(np.median([circ(x["cents"] - x["ref_c"] - key) for x in stable]))
    print(f"细化调差 k={key:+.1f} 音分（{len(stable)} 个稳定音）")
    for x in rows:
        if x["status"] != "pending":
            continue
        dev = circ(x["cents"] - x["ref_c"] - key)
        x |= {"ref_note": note_name(x["ref_c"] + key), "deviation": round(dev, 1)}
        if x["ref_spread"] > REF_MAX_SPREAD:
            x |= {"status": "ref_unstable", "applied": 0.0}
        elif abs(dev) < FIX_THRESHOLD:
            x |= {"status": "ok", "applied": 0.0}
        elif abs(dev) > a.max_shift:
            x |= {"status": "too_large_skip", "applied": 0.0}
        else:
            x |= {"status": "fix", "applied": round(-dev * a.strength, 1)}

    for x in rows:
        print(f"  {x['t0']:5.2f}-{x['t1']:5.2f} {x['note']:>4} ref {x.get('ref_t0') or 0:5.2f} "
              f"{x.get('ref_note', '-'):>4} dev {x.get('deviation', float('nan')):+7.1f} {x['status']} {x['applied']:+.1f}")
    fix = [x for x in rows if x["applied"] != 0]

    x_in, sr = read_wav_pcm16(Path(a.target))
    x_corr = apply_tapered(parselmouth.Sound(x_in, sr), [(x["t0"], x["t1"], x["applied"]) for x in fix])[: len(x_in)]
    out.mkdir(parents=True)
    g = 10 ** ((-16.0 - rms_db(x_in)) / 20.0)
    g = min(g, 10 ** ((-3.0 - max(peak_db(x_in), peak_db(x_corr))) / 20.0))
    write_wav_pcm16(out / "01_target_original.wav", x_in * g, sr)
    write_wav_pcm16(out / "02_corrected.wav", x_corr * g, sr)
    x_ref, rsr = read_wav_pcm16(Path(a.reference))
    seg = x_ref[int(ref_t0 * rsr): int(ref_t1 * rsr) + 1]
    gr = 10 ** ((-16.0 - rms_db(seg)) / 20.0)
    gr = min(gr, 10 ** ((-3.0 - peak_db(seg)) / 20.0))
    write_wav_pcm16(out / "03_reference_matched_segment.wav", seg * gr, rsr)

    with open(out / "notes.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["t0", "t1", "note", "ref_t0", "ref_t1", "ref_note", "ref_voiced", "ref_spread", "deviation", "status", "applied"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "reference": a.reference, "align_cost_per_frame": round(cost, 1),
                   "reversed_ref_baseline_cost": round(base_cost, 1), "key_offset_cents": round(key, 1), "align": a.align,
                   "ref_segment_s": [round(ref_t0, 2), round(ref_t1, 2)], "notes": len(rows),
                   "fixed": len(fix), "strength": a.strength,
                   "params": {"fix_threshold": FIX_THRESHOLD, "max_shift": a.max_shift, "ref_max_spread": REF_MAX_SPREAD,
                              "hop": HOP, "unvoiced_cost": UNVOICED_COST, "dist_cap": DIST_CAP}},
                  f, ensure_ascii=False, indent=2)
    print(f"修正 {len(fix)}/{len(rows)} 音; 输出: {out}")


if __name__ == "__main__":
    main()
