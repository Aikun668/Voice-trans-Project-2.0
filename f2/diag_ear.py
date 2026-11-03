"""F2 人耳校准测试: 判断"听不出音高变化"是播放链路还是听法/感知阈值问题。

做法: 从 shige_medium 取一个稳定长音(4.88-5.49s)和一段 2 秒短句,
      每条测试 WAV 内放"两声紧邻", 中间留 0.5s。
      A 控制组: 两声完全相同(判"一样"才是正确)
      B/C: 第二声分别高 100/200 音分
      D: 短句第二声高 100 音分
全部 RMS 统一, 边缘 25ms 淡入淡出防爆音。
"""

import math
import sys
from pathlib import Path

import numpy as np
import parselmouth

from poc_phase1 import read_wav_pcm16, rms_db, shift_uniform, to_array, write_wav_pcm16

FADE = 0.025
GAP = 0.50
PAD = 0.30
TARGET_RMS = -16.0


def fade(x, sr):
    n = int(FADE * sr)
    y = x.copy()
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def norm(x, sr):
    g = 10 ** ((TARGET_RMS - rms_db(x)) / 20.0)
    return fade(x * g, sr)


def seg_f0(x, sr):
    """FFT 谐波峰估计 F0(独立于 Praat 跟踪器)。"""
    n = 1 << 18
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    m = (fr > 120) & (fr < 600)
    f0c = fr[m][np.argmax(sp[m])]
    mh = (fr > 5 * f0c - 40) & (fr < 5 * f0c + 40)
    return fr[mh][np.argmax(sp[mh])] / 5.0


def pair(item1, item2, sr):
    gap = np.zeros(int(GAP * sr))
    pad = np.zeros(int(PAD * sr))
    return np.concatenate([pad, item1, gap, item2, pad])


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_dataset/my_singing/shige_medium.wav")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("f2/outputs/diag_ear_v001")
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")

    x, sr = read_wav_pcm16(src)
    note = x[int(4.88 * sr):int(5.49 * sr)].copy()   # A#3, 0.61s, 稳定
    phrase = x[int(4.70 * sr):int(6.70 * sr)].copy()  # 2s

    note = norm(note, sr)
    phrase = norm(phrase, sr)
    note_up100 = norm(to_array(shift_uniform(parselmouth.Sound(note, sr), 100.0)), sr)
    note_up200 = norm(to_array(shift_uniform(parselmouth.Sound(note, sr), 200.0)), sr)
    phrase_up100 = norm(to_array(shift_uniform(parselmouth.Sound(phrase, sr), 100.0)), sr)

    out.mkdir(parents=True)
    files = {
        "pair_a_control.wav": ("控制组: 两声完全相同", note, note),
        "pair_b_note_plus100.wav": ("单音: 第二声高 100 音分(一个半音)", note, note_up100),
        "pair_c_note_plus200.wav": ("单音: 第二声高 200 音分(两个半音)", note, note_up200),
        "pair_d_phrase_plus100.wav": ("短句: 第二声高 100 音分", phrase, phrase_up100),
    }
    print(f"源: {src}  sr={sr}")
    for name, (desc, a, b) in files.items():
        y = pair(a, b, sr)
        write_wav_pcm16(out / name, y, sr)
        fa, fb = seg_f0(a, sr), seg_f0(b, sr)
        print(f"{name}: 第1声 {fa:6.1f}Hz  第2声 {fb:6.1f}Hz  "
              f"差 {1200 * math.log2(fb / fa):+6.1f} 音分  ({desc})")

    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
