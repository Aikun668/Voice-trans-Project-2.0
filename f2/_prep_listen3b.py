"""准备 Phase-3b 试听材料: 统一响度 + ABA 拼接。"""
import shutil
from pathlib import Path

import numpy as np

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

SRC = Path("f2/outputs/phase3b_shige_v001")
LST = SRC / "listening" / "audio"
LST.mkdir(parents=True, exist_ok=True)

for src, dst in [("01_target_original.wav", "before.wav"),
                 ("02_corrected.wav", "after.wav"),
                 ("03_reference_take.wav", "reference.wav")]:
    shutil.copyfile(SRC / src, LST / dst)
    x, sr = read_wav_pcm16(LST / dst)
    print(f"{dst:14s} {len(x)/sr:6.2f}s  RMS {rms_db(x):+6.2f}  Peak {peak_db(x):+6.2f}  削波 {clip_count(x)}")

before, sr = read_wav_pcm16(LST / "before.wav")
after, _ = read_wav_pcm16(LST / "after.wav")
t0, t1 = int(1.8 * sr), int(3.4 * sr)
gap = np.zeros(int(0.4 * sr))
aba = np.concatenate([before[t0:t1], gap, after[t0:t1], gap, before[t0:t1]])
write_wav_pcm16(LST / "aba_1.8-3.4.wav", aba, sr)
print(f"aba_1.8-3.4.wav {len(aba)/sr:.2f}s")
