"""F2 Phase-4c: 拖音（长音/装饰尾音）修音 + 连续衔接。

依据 demo v002 反馈：严重版里"一个字拖音拖得非常严重"、尾字故意变口型/吼；希望修拖音的音高并衔接好；
轻度版 16–19 s 有电音感。诊断：
- phase4/4b 只修"稳定音"（notes_of: 段内 p90−p10 ≤150 音分），三条跑调录音 37–49% 的演唱时间
  （正是拖音/滑音/变口型段）从未进入判定。
- 4b 每个修正音在边缘渐变回 0：相邻音反向修正时（轻度版 17.69/17.72 s：+130 → −114），
  音高在音与音之间先回到原位再跳走，形成"抖动"，与电音感吻合。
本版：
1. 子序列 DTW 增加"保持"步（参考停在同一帧、目标继续），代价 HOLD_PEN/帧：拖音比原唱长多少倍都能对齐。
2. 以"事件"为单位（按音高突变与清音切分的连续浊音段），事件内按 BLOCK_S 分块取偏差中位数，
   修正量随块平滑变化 → 拖音里慢慢跑偏的部分会被慢慢拉回，颤音等快速起伏保留（不被拉平）。
3. 修正曲线跨音连续：相邻两个被修事件间隔 ≤ JOIN_GAP_S 时在间隙内线性过渡，不回到 0；
   孤立事件在事件外的过渡区渐变，事件本身完整修正。
4. 只在修正曲线非零处拼接重合成音频（沿用 4b splice），其余原样。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import peak_db, read_wav_pcm16, rms_db, write_wav_pcm16
from poc_phase4_realign import DIST_CAP, HOP, PITCH_CEIL, PITCH_FLOOR, UNVOICED_COST, cents_track
from poc_phase4b import splice

call = parselmouth.praat.call

HOLD_PEN = 40.0
FIX_THRESHOLD = 60.0
MAX_SHIFT = 250.0
REF_MAX_IQR = 80.0
JUMP_CENTS = 70.0          # 相邻 20 ms 帧（平滑后）音高变化超过此值视为换音
MIN_EVENT_S = 0.12
BLOCK_S = 0.24
JOIN_GAP_S = 0.30
EDGE_RAMP_S = 0.08
KEY_TRIM = 70.0
WINDOW_PAD_S = 3.0
MAX_HOLD_FRACTION = 0.5
AGREE_CENTS = 50.0
MIN_FIX_EVENT_S = 0.25      # 短于此不修：句内短音 ~100 音分不可辨（F2 校准），短事件大修正 = 快速滑音 = 电音感


def median_filter(x, size):
    """.venv-f2 无 scipy：奇数窗中值滤波，边缘复制。"""
    h = size // 2
    xp = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, size), axis=1)


def uniform_filter1d(x, size, mode="nearest"):
    """滑动平均，边缘复制（等价于 scipy mode='nearest'）。"""
    if size <= 1:
        return np.asarray(x, dtype=float).copy()
    lo, hi = size // 2, size - 1 - size // 2
    xp = np.pad(np.asarray(x, dtype=float), (lo, hi), mode="edge")
    return np.convolve(xp, np.ones(size) / size, mode="valid")


def circ(v):
    return (np.asarray(v) + 600.0) % 1200.0 - 600.0


def cost_matrix(q, r, k):
    c = np.minimum(np.abs(circ(q[:, None] - r[None, :] - k)), DIST_CAP)
    qv, rv = ~np.isnan(q)[:, None], ~np.isnan(r)[None, :]
    return np.where(qv & rv, c, np.where(qv | rv, UNVOICED_COST, 0.0))


def dtw_hold(C):
    """步型 (1,1)(1,2)(2,1) + 保持 (1,0)+HOLD_PEN；第 0 行任意起点。返回 (平均代价, 路径)。"""
    n, m = C.shape
    INF = 1e18
    D = np.full((n, m), INF)
    B = np.zeros((n, m), dtype=np.int8)
    D[0] = C[0]
    for i in range(1, n):
        cand = np.full((4, m), INF)
        cand[0, 1:] = D[i - 1, :-1]
        cand[1, 2:] = D[i - 1, :-2]
        if i >= 2:
            cand[2, 1:] = D[i - 2, :-1] + C[i - 1, 1:]
        cand[3] = D[i - 1] + HOLD_PEN
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
        elif b == 2:
            path.append((i - 1, j - 1))
            i, j = i - 2, j - 1
        else:
            i -= 1
    path.append((0, j))
    path.reverse()
    return total / n, path


def align(q, r):
    """两阶段：① 无保持步的子序列 DTW（phase4 原样）定位参考区间与调差——保持步若参与全曲搜索，
    会靠"停在少数几帧"在错误段落拿到低代价（实测轻度版落到 239 s）；
    ② 只在该区间 ±WINDOW_PAD_S 内、调差 ±30 音分内，用含保持步的 DTW 细对齐。"""
    from poc_phase4_realign import align as align_nohold
    _, path0, k0 = align_nohold(q, r)
    pad = int(round(WINDOW_PAD_S / HOP))
    lo, hi = max(0, path0[0][1] - pad), min(len(r), path0[-1][1] + pad)
    rw = r[lo:hi]
    best = None
    for k in np.arange(k0 - 30, k0 + 31, 5):
        cost, path = dtw_hold(cost_matrix(q, rw, float(k)))
        if best is None or cost < best[0]:
            best = (cost, path, float(k))
    cost, path, k = best
    return cost, [(i, j + lo) for i, j in path], k, (path0, k0)


def frame_reference(q, r, path):
    """每个目标帧对应的参考音高（八度折叠后的中位数）与参考局部稳定度（±2 帧 IQR）。"""
    n = len(q)
    js = [[] for _ in range(n)]
    for i, j in path:
        js[i].append(j)
    ref = np.full(n, np.nan)
    iqr = np.full(n, np.nan)
    for i in range(n):
        if not js[i]:
            continue
        lo, hi = max(0, min(js[i]) - 2), min(len(r), max(js[i]) + 3)
        w = r[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        seed = np.median(w)
        w = seed + circ(w - seed)
        ref[i] = np.median(w)
        iqr[i] = np.percentile(w, 75) - np.percentile(w, 25)
    return ref, iqr


def events_of(q):
    """连续浊音段内按音高突变切分；返回 [(i0, i1)]（i1 不含）。"""
    v = ~np.isnan(q)
    qs = q.copy()
    qs[v] = median_filter(np.where(v, q, 0.0), size=5)[v]
    out, i, n = [], 0, len(q)
    min_len = int(round(MIN_EVENT_S / HOP))
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i + 1
        while j < n and v[j] and abs(qs[j] - qs[j - 1]) <= JUMP_CENTS:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def correction_curve(q, ref, iqr, key, events, max_shift, ref0=None):
    n = len(q)
    dev = circ(q - ref - key)
    ok = ~np.isnan(q) & ~np.isnan(ref) & (iqr <= REF_MAX_IQR)
    blk = int(round(BLOCK_S / HOP))
    rows, fixed = [], []
    for e0, e1 in events:
        good = ok[e0:e1]
        if good.mean() < 0.5:
            rows.append({"t0": e0 * HOP, "t1": e1 * HOP, "status": "ref_unreliable"})
            continue
        med = float(np.median(dev[e0:e1][good]))
        row = {"t0": e0 * HOP, "t1": e1 * HOP, "deviation": round(med, 1), "reliable": round(float(good.mean()), 2)}
        if abs(med) < FIX_THRESHOLD:
            rows.append(row | {"status": "ok"})
            continue
        if (e1 - e0) * HOP < MIN_FIX_EVENT_S:
            rows.append(row | {"status": "short_skip"})
            continue
        if abs(med) > max_shift:
            rows.append(row | {"status": "too_large_skip"})
            continue
        if ref0 is not None:                                  # 两种对齐必须给出一致的偏差，否则不修
            d0 = circ(q[e0:e1] - ref0[e0:e1] - key)
            d0 = d0[~np.isnan(d0)]
            med0 = float(np.median(d0)) if len(d0) >= 3 else np.nan
            row["deviation_stage1"] = None if np.isnan(med0) else round(med0, 1)
            if np.isnan(med0) or abs(med0 - med) > AGREE_CENTS:
                rows.append(row | {"status": "align_disagree"})
                continue
        # 事件内分块：每块偏差中位数（不可靠块用事件中位数），块间平滑
        c = np.full(e1 - e0, -med)
        for b0 in range(e0, e1, blk):
            b1 = min(e1, b0 + blk)
            if e1 - b1 < blk // 2:
                b1 = e1
            g = ok[b0:b1]
            if g.sum() >= max(3, 0.5 * (b1 - b0)):
                c[b0 - e0:b1 - e0] = -np.clip(np.median(dev[b0:b1][g]), -max_shift, max_shift)
            if b1 == e1:
                break
        c = uniform_filter1d(c, size=min(len(c), 7), mode="nearest")
        fixed.append((e0, e1, c))
        rows.append(row | {"status": "fix", "applied_mean": round(float(c.mean()), 1),
                           "applied_range": [round(float(c.min()), 1), round(float(c.max()), 1)]})

    curve = np.zeros(n)
    join = int(round(JOIN_GAP_S / HOP))
    ramp = int(round(EDGE_RAMP_S / HOP))
    for k, (e0, e1, c) in enumerate(fixed):
        curve[e0:e1] = c
        prev = fixed[k - 1] if k > 0 else None
        nxt = fixed[k + 1] if k + 1 < len(fixed) else None
        if nxt and nxt[0] - e1 <= join:           # 与下一个被修事件之间线性过渡（不回到 0）
            g = nxt[0] - e1
            curve[e1:nxt[0]] = np.linspace(c[-1], nxt[2][0], g + 2)[1:-1]
        else:
            a = min(ramp, n - e1)
            curve[e1:e1 + a] = c[-1] * np.cos(0.5 * np.pi * np.arange(1, a + 1) / (a + 1)) ** 2
        if not (prev and e0 - prev[1] <= join):
            a = min(ramp, e0)
            curve[e0 - a:e0] = c[0] * np.sin(0.5 * np.pi * np.arange(1, a + 1) / (a + 1)) ** 2
    return uniform_filter1d(curve, size=3), rows, dev, ok


def shift_by_curve(x, sr, curve):
    sound = parselmouth.Sound(x, sr)
    manip = call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    tc = (np.arange(len(curve)) + 0.5) * HOP
    new = call("Create PitchTier", "curve", 0.0, sound.get_total_duration())
    for idx in range(1, n + 1):
        t = call(tier, "Get time from index", idx)
        f = call(tier, "Get value at index", idx)
        call(new, "Add point", t, f * 2.0 ** (float(np.interp(t, tc, curve)) / 1200.0))
    call([manip, new], "Replace pitch tier")
    y = np.asarray(call(manip, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def spans_of(curve, eps=1.0):
    on = np.abs(curve) > eps
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append((i * HOP, j * HOP, float(np.mean(curve[i:j]))))
            i = j
        else:
            i += 1
    return out


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
    cost, path, key, (path0, k0) = align(q, r)
    lo, hi = path[0][1], path[-1][1] + 1
    base_cost, _ = dtw_hold(cost_matrix(q, r[lo:hi][::-1].copy(), key))
    ref, iqr = frame_reference(q, r, path)
    ok = ~np.isnan(q) & ~np.isnan(ref) & (iqr <= REF_MAX_IQR)
    for it in range(3):                               # 截尾中位数细化调差
        d = circ(q[ok] - ref[ok] - key)
        key += float(np.median(d if it == 0 else d[np.abs(d) < KEY_TRIM]))
    events = events_of(q)
    ref0, _ = frame_reference(q, r, path0)
    curve, rows, dev, ok = correction_curve(q, ref, iqr, key, events, a.max_shift, ref0=ref0)
    holds = sum(1 for k in range(1, len(path)) if path[k][1] == path[k - 1][1])
    if holds > MAX_HOLD_FRACTION * len(q):
        raise SystemExit(f"保持步占比 {holds / len(q):.0%} 过高，对齐退化，拒绝输出")
    print(f"对齐代价 {cost:.1f}/帧（反转基线 {base_cost:.1f}）; 调差 {key:+.1f}; 保持步 {holds} 帧; "
          f"参考 {path[0][1] * HOP:.2f}-{path[-1][1] * HOP:.2f}s; 事件 {len(events)}")
    for x in rows:
        if x["status"] == "fix":
            print(f"  {x['t0']:5.2f}-{x['t1']:5.2f} dev {x['deviation']:+7.1f} -> 修 {x['applied_range']}")

    x_in, sr = read_wav_pcm16(Path(a.target))
    spans = spans_of(curve)
    y = shift_by_curve(x_in, sr, curve) if spans else x_in.copy()
    x_corr, w = splice(x_in, y, sr, spans)

    # 修正后残差：重测输出音高，对可靠帧比较与参考的偏差
    q2 = cents_track_array(x_corr, sr)[: len(q)]
    m = ok[: len(q2)] & ~np.isnan(q2)
    before = np.abs(dev[: len(q2)][m])
    after = np.abs(circ(q2[m] - ref[: len(q2)][m] - key))
    voiced_time = float(np.sum(~np.isnan(q)) * HOP)
    fixed_time = float(sum(x["t1"] - x["t0"] for x in rows if x["status"] == "fix"))

    out.mkdir(parents=True)
    g = 10 ** ((-16.0 - rms_db(x_in)) / 20.0)
    g = min(g, 10 ** ((-3.0 - max(peak_db(x_in), peak_db(x_corr))) / 20.0))
    write_wav_pcm16(out / "01_target_original.wav", x_in * g, sr)
    write_wav_pcm16(out / "02_corrected.wav", x_corr * g, sr)
    x_ref, rsr = read_wav_pcm16(Path(a.reference))
    seg = x_ref[int(path[0][1] * HOP * rsr): int(path[-1][1] * HOP * rsr) + 1]
    gr = min(10 ** ((-16.0 - rms_db(seg)) / 20.0), 10 ** ((-3.0 - peak_db(seg)) / 20.0))
    write_wav_pcm16(out / "03_reference_matched_segment.wav", seg * gr, rsr)
    np.savetxt(out / "curve_cents.csv", np.c_[(np.arange(len(curve)) + 0.5) * HOP, curve], delimiter=",",
               header="t,cents", comments="", fmt="%.3f")

    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["t0", "t1", "deviation", "deviation_stage1", "reliable", "status", "applied_mean", "applied_range"]
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    qa = {"target": a.target, "reference": a.reference, "align_cost_per_frame": round(cost, 1),
          "reversed_ref_baseline_cost": round(base_cost, 1), "key_offset_cents": round(key, 1), "hold_frames": holds,
          "ref_segment_s": [round(path[0][1] * HOP, 2), round(path[-1][1] * HOP, 2)],
          "events": len(events), "fixed_events": sum(x["status"] == "fix" for x in rows),
          "disagree_events": sum(x["status"] == "align_disagree" for x in rows),
          "fixed_time_s": round(fixed_time, 2), "voiced_time_s": round(voiced_time, 2),
          "resynth_fraction": round(float(np.mean(w > 0)), 3),
          "reliable_frame_abs_dev_median_before": round(float(np.median(before)), 1),
          "reliable_frame_abs_dev_median_after": round(float(np.median(after)), 1),
          "reliable_frames_over_100c_before": round(float(np.mean(before > 100)), 3),
          "reliable_frames_over_100c_after": round(float(np.mean(after > 100)), 3),
          "spans": [[round(t0, 2), round(t1, 2), round(c, 1)] for t0, t1, c in spans],
          "stage1_ref_segment_s": [round(path0[0][1] * HOP, 2), round(path0[-1][1] * HOP, 2)], "stage1_key": k0,
          "params": {"hold_pen": HOLD_PEN, "window_pad_s": WINDOW_PAD_S, "fix_threshold": FIX_THRESHOLD, "max_shift": a.max_shift,
                     "ref_max_iqr": REF_MAX_IQR, "jump_cents": JUMP_CENTS, "min_event_s": MIN_EVENT_S,
                     "block_s": BLOCK_S, "join_gap_s": JOIN_GAP_S, "edge_ramp_s": EDGE_RAMP_S, "key_trim": KEY_TRIM, "agree_cents": AGREE_CENTS, "min_fix_event_s": MIN_FIX_EVENT_S}}
    (out / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"修 {qa['fixed_events']} 个事件 {fixed_time:.1f}s / 浊音 {voiced_time:.1f}s; 重合成 {qa['resynth_fraction']:.0%}; "
          f"可靠帧偏差中位 {qa['reliable_frame_abs_dev_median_before']} -> {qa['reliable_frame_abs_dev_median_after']} 音分; "
          f">100 音分帧 {qa['reliable_frames_over_100c_before']:.0%} -> {qa['reliable_frames_over_100c_after']:.0%}")


def cents_track_array(x, sr):
    p = parselmouth.Sound(x, sr).to_pitch(HOP, PITCH_FLOOR, PITCH_CEIL).selected_array["frequency"]
    c = np.full(len(p), np.nan)
    c[p > 0] = 1200 * np.log2(p[p > 0] / 440.0)
    return c


if __name__ == "__main__":
    main()
