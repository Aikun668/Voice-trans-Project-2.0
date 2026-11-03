"""检查 pair7 在 ~3s 处的接缝伪影: 单音 +247 音分移调的边界。

检查项:
1. F0 轮廓在边界处是否突变(帧间跳变)
2. 是否存在"两个同时存在的高音峰"(双音/双重基频)
3. 波形是否有采样级跳变(爆音)
4. 与未修改的第 1 遍对照
"""

import math
import sys

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16

FLOOR, CEIL = 75.0, 600.0


def f0_track(x, sr, t0, t1, step=0.005):
    p = parselmouth.praat.call(parselmouth.Sound(x, sr), "To Pitch", step, FLOOR, CEIL)
    n = parselmouth.praat.call(p, "Get number of frames")
    ts, vs = [], []
    for i in range(1, n + 1):
        t = parselmouth.praat.call(p, "Get time from frame number", i)
        if t0 <= t <= t1:
            v = parselmouth.praat.call(p, "Get value in frame", i, "Hertz")
            ts.append(t)
            vs.append(v if (v == v and v > 0) else math.nan)
    return np.array(ts), np.array(vs)


def spectral_peaks(x, sr, center, width=0.030, topn=4):
    a = int((center - width / 2) * sr)
    b = int((center + width / 2) * sr)
    seg = x[a:b]
    if len(seg) < 64:
        return []
    n = 1 << 18
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    m = (fr > 100) & (fr < 1200)
    fr, sp = fr[m], sp[m]
    # 局部极大
    pk = [(fr[i], sp[i]) for i in range(1, len(sp) - 1) if sp[i] > sp[i - 1] and sp[i] > sp[i + 1]]
    pk.sort(key=lambda p: -p[1])
    if not pk:
        return []
    ref = pk[0][1]
    return [(round(f, 1), round(20 * math.log10(v / ref), 1)) for f, v in pk[:topn]]


def main():
    path = sys.argv[1]
    x, sr = read_wav_pcm16(path)
    print(f"{path}  sr={sr}  时长={len(x) / sr:.3f}s")

    # 第1遍: 0.30-2.30; 第2遍: 2.80-4.80; 被改音在第2遍内 0.181-0.791 -> 2.981-3.591
    for label, off in [("第1遍(未改动)", 0.30), ("第2遍(已移调)", 2.80)]:
        t0, t1 = off + 0.15, off + 0.85
        ts, vs = f0_track(x, sr, t0, t1)
        voiced = ~np.isnan(vs)
        print(f"\n=== {label} F0 轮廓 {t0:.2f}-{t1:.2f}s (帧步长 5ms) ===")
        prev = None
        maxjump = 0.0
        for t, v in zip(ts, vs):
            mark = ""
            if not math.isnan(v) and prev is not None and not math.isnan(prev):
                j = 1200 * math.log2(v / prev)
                if abs(j) > 60:
                    mark = f"  <-- 跳变 {j:+.0f}c"
                maxjump = max(maxjump, abs(j))
            if (t - t0) % 0.05 < 0.005:
                print(f"  {t:.3f}s  {'nan(无声)' if math.isnan(v) else f'{v:7.1f}Hz':>12s}  相对{off:.2f}s={t - off:.3f}s{mark}")
            prev = v
        print(f"  有声帧 {voiced.sum()}/{len(vs)}, 最大帧间跳变 {maxjump:.1f} 音分")

    # 双峰检查: 边界处 2.98-3.05s
    print("\n=== 频谱峰(30ms 窗), 查是否存在两个相当强度的高峰 ===")
    for center in [2.92, 2.97, 3.00, 3.03, 3.10]:
        peaks = spectral_peaks(x, sr, center)
        print(f"  t={center:.2f}s: {peaks}")

    # 波形跳变
    d = np.abs(np.diff(x))
    print(f"\n波形最大采样级跳变: {d.max():.4f} (满量程 1.0); 边界区 2.95-3.05s: {d[int(2.95*sr):int(3.05*sr)].max():.4f}")
    print(f"全文件 99.9 分位跳变: {np.percentile(d, 99.9):.4f}")


if __name__ == "__main__":
    main()
