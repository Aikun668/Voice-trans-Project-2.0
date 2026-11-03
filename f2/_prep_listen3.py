"""准备 Phase-3 试听材料: 复制音频 + 生成改动最集中段落的 ABA 拼接。"""
import shutil
from pathlib import Path

import numpy as np

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

SRC = Path("f2/outputs/phase3_shige_v001")
LST = SRC / "listening" / "audio"
LST.mkdir(parents=True, exist_ok=True)

for src, dst in [("01_target_original.wav", "before.wav"),
                 ("02_corrected.wav", "after.wav")]:
    shutil.copyfile(SRC / src, LST / dst)
    x, sr = read_wav_pcm16(LST / dst)
    print(f"{dst:14s} {len(x)/sr:6.2f}s  RMS {rms_db(x):+6.2f}  Peak {peak_db(x):+6.2f}  削波 {clip_count(x)}")

# 参考版统一响度到 before
x_ref, sr_ref = read_wav_pcm16(SRC / "03_reference_take.wav")
tgt = rms_db(read_wav_pcm16(LST / "before.wav")[0])
g = 10 ** ((tgt - rms_db(x_ref)) / 20.0)
pk = peak_db(x_ref * g)
if pk > -3.0:
    g *= 10 ** ((-3.0 - pk) / 20.0)
write_wav_pcm16(LST / "reference.wav", x_ref * g, sr_ref)
x, _ = read_wav_pcm16(LST / "reference.wav")
print(f"{'reference.wav':14s} {len(x)/sr_ref:6.2f}s  RMS {rms_db(x):+6.2f}  Peak {peak_db(x):+6.2f}")

# ABA: 8.0-9.6s 段(含 A4->D4 的大修正)
before, sr = read_wav_pcm16(SRC / "01_target_original.wav")
after, _ = read_wav_pcm16(SRC / "02_corrected.wav")
t0, t1 = int(7.8 * sr), int(9.7 * sr)
gap = np.zeros(int(0.4 * sr))
aba = np.concatenate([before[t0:t1], gap, after[t0:t1], gap, before[t0:t1]])
write_wav_pcm16(LST / "aba_7.8-9.7.wav", aba, sr)
print(f"aba_7.8-9.7.wav  {len(aba)/sr:.2f}s")
