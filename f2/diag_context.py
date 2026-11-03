"""F2 上下文感知测试: 旋律里的个别音改动能否被听出。

背景: 校准测试已确认——单音升半音可听出, 但整句整体升半音听不出(旋律归一化)。
本测试回答关键问题: 真正修音那种"个别音被改动"的变化, 在句子上下文里能被听出吗?

Pair E: 原句 vs 全量化版(40/40 音都修, 强度100%, 死区0) —— 大量小修正是否可感知
Pair F: 原句 vs 只把一个音挪半音(+) —— 单点大修正是否可感知
全部 RMS 统一, 25ms 淡入淡出。
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
    g = 10 ** ((TARGET_RMS - rms_db(x)) / 20.0)
    return fade(x * g, sr)


def pair(a, b, sr):
    return np.concatenate([np.zeros(int(PAD * sr)), a, np.zeros(int(GAP * sr)), b, np.zeros(int(PAD * sr))])


def med_f0(x, sr, t0=None, t1=None):
    p = parselmouth.praat.call(parselmouth.Sound(x, sr), "To Pitch", 0.01, PITCH_FLOOR, PITCH_CEIL)
    if t0 is None:
        return parselmouth.praat.call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    return parselmouth.praat.call(p, "Get quantile", t0, t1, 0.5, "Hertz")


def shift_range(x, sr, t0, t1, cents):
    """只把 [t0,t1] 区间移调, 其余保持不变。"""
    sound = parselmouth.Sound(x, sr)
    manip = parselmouth.praat.call(sound, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    parselmouth.praat.call(tier, "Multiply frequencies", t0, t1, 2.0 ** (cents / 1200.0))
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    out = parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(out.values[0], dtype=np.float64)


def main():
    src = Path(sys.argv[1])
    quant = Path(sys.argv[2])
    out = Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    x, sr = read_wav_pcm16(src)
    xq, _ = read_wav_pcm16(quant)
    # 取修正最密集段 6.0-8.0s(该段 5 个音被挪 29~44 音分)
    P0, P1 = 6.00, 8.00
    N0, N1 = 7.391 - P0, 7.601 - P0   # 段内稳定音 7.391-7.601s
    phrase = x[int(P0 * sr):int(P1 * sr)].copy()
    phrase_q = xq[int(P0 * sr):int(P1 * sr)].copy()
    phrase_one = shift_range(phrase.copy(), sr, N0, N1, 100.0)

    phrase = norm(phrase, sr)
    phrase_q = norm(phrase_q, sr)
    phrase_one = norm(phrase_one, sr)

    out.mkdir(parents=True)
    files = {
        "pair_e_fullquant.wav": ("全量化版(40音全修)", phrase, phrase_q),
        "pair_f_single_note.wav": ("只把一个音挪高半音", phrase, phrase_one),
    }
    print(f"源: {src}  全量化: {quant}  sr={sr}")
    for name, (desc, a, b) in files.items():
        write_wav_pcm16(out / name, pair(a, b, sr), sr)
        # 核验: 句内该音的实测差
        fa = med_f0(a, sr, N0 + 0.02, N1 - 0.02)
        fb = med_f0(b, sr, N0 + 0.02, N1 - 0.02)
        print(f"{name}: 句内稳定音 {fa:.1f} -> {fb:.1f}Hz 差 {1200 * math.log2(fb / fa):+6.1f} 音分  ({desc})")
        whole_a, whole_b = med_f0(a, sr), med_f0(b, sr)
        print(f"    整句中位: {whole_a:.1f} -> {whole_b:.1f}Hz 差 {1200 * math.log2(whole_b / whole_a):+6.1f} 音分")

    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
