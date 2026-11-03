"""Phase-3d 试听材料: ABA + 分组试听页数据。"""
import shutil
from pathlib import Path

import numpy as np

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

D = Path("f2/outputs/phase3d_gyjyn_v001")
LST = D / "listening" / "audio"
LST.mkdir(parents=True, exist_ok=True)

before, sr = read_wav_pcm16(D / "01_target_original.wav")
after, _ = read_wav_pcm16(D / "02_corrected.wav")
shutil.copyfile(D / "01_target_original.wav", LST / "before.wav")
shutil.copyfile(D / "02_corrected.wav", LST / "after.wav")

for f in ["before.wav", "after.wav"]:
    x, _ = read_wav_pcm16(LST / f)
    print(f"{f:12s} RMS {rms_db(x):+.2f} Peak {peak_db(x):+.2f} 削波 {clip_count(x)}")

t0, t1 = int(1.9 * sr), int(3.9 * sr)
gap = np.zeros(int(0.45 * sr))
aba = np.concatenate([before[t0:t1], gap, after[t0:t1], gap, before[t0:t1]])
write_wav_pcm16(LST / "aba_1.9-3.9.wav", aba, sr)
print(f"aba_1.9-3.9.wav {len(aba)/sr:.2f}s")
