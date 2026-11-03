"""F2 正控制测试: 用"绝对该听得出"的大幅改动验证测试方式本身是否有效。

背景: 校准测试中单音 ±100 音分可辨, 但句内局部 +103 音分与多音 29~44 音分均不可辨。
需排除"这个测试方式本身测不出"的可能, 故加入正控制:
  Pair 7: 只把一个 0.61s 长音抬高 250 音分(两个半音)——旋律上明显错音
  Pair 8: 整句抬高 250 音分——大幅整体移调, 正常情况下绝不可能听不出
若 Pair 8 也听不出, 则该听法/听感通道无法用于判断音高, 需换评估方式(如音高曲线图)。
"""

import math
import sys
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16, rms_db, write_wav_pcm16

FADE = 0.025
GAP = 0.50
PAD = 0.30
TARGET_RMS = -16.0
PITCH_FLOOR = 75.0
PITCH_CEIL = 600.0


def fade(x, sr):
    n = int(FADE * sr)
    y = x.copy()
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def norm(x, sr):
    return fade(x * 10 ** ((TARGET_RMS - rms_db(x)) / 20.0), sr)


def pair(a, b, sr):
    return np.concatenate([np.zeros(int(PAD * sr)), a, np.zeros(int(GAP * sr)), b, np.zeros(int(PAD * sr))])


def med_f0(x, sr, t0=None, t1=None):
    p = parselmouth.praat.call(parselmouth.Sound(x, sr), "To Pitch", 0.01, PITCH_FLOOR, PITCH_CEIL)
    return parselmouth.praat.call(p, "Get quantile", 0.0 if t0 is None else t0,
                                  0.0 if t1 is None else t1, 0.5, "Hertz")


def shift_range(x, sr, t0, t1, cents):
    sound = parselmouth.Sound(x, sr)
    manip = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    parselmouth.praat.call(tier, "Multiply frequencies", t0, t1, 2.0 ** (cents / 1200.0))
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    return np.asarray(parselmouth.praat.call(manip, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")
    x, sr = read_wav_pcm16(src)

    P0, P1 = 4.70, 6.70          # 2s 短句
    N0, N1 = 4.881 - P0, 5.491 - P0  # 句内 0.61s 长音(相对时间)
    CENTS = 250.0

    phrase = norm(x[int(P0 * sr):int(P1 * sr)].copy(), sr)
    one = norm(shift_range(x[int(P0 * sr):int(P1 * sr)].copy(), sr, N0, N1, CENTS), sr)
    whole = norm(shift_range(x[int(P0 * sr):int(P1 * sr)].copy(), sr, 0.0, P1 - P0, CENTS), sr)

    out.mkdir(parents=True)
    for name, second in [("pair7_one_note_plus250.wav", one), ("pair8_whole_plus250.wav", whole)]:
        write_wav_pcm16(out / name, pair(phrase, second, sr), sr)
        fa, fb = med_f0(phrase, sr, N0 + 0.02, N1 - 0.02), med_f0(second, sr, N0 + 0.02, N1 - 0.02)
        print(f"{name}: 该长音 {fa:.1f} -> {fb:.1f}Hz 差 {1200 * math.log2(fb / fa):+6.1f}c | "
              f"整句中位 {med_f0(phrase, sr):.1f} -> {med_f0(second, sr):.1f}Hz "
              f"差 {1200 * math.log2(med_f0(second, sr) / med_f0(phrase, sr)):+6.1f}c")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
