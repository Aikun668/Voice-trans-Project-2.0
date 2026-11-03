"""Phase-3e 试听材料。"""
import shutil
from pathlib import Path

import numpy as np

from poc_phase1 import clip_count, peak_db, read_wav_pcm16, rms_db, write_wav_pcm16

D = Path("f2/outputs/phase3e_gyjyn_v002")
LST = D / "listening" / "audio"
LST.mkdir(parents=True, exist_ok=True)

for src, dst in [("01_target_original.wav", "before.wav"),
                 ("02_corrected.wav", "after.wav"),
                 ("03_reference_aligned.wav", "reference.wav")]:
    shutil.copyfile(D / src, LST / dst)
    x, _ = read_wav_pcm16(LST / dst)
    print(f"{dst:13s} {len(x)/44100:6.2f}s RMS {rms_db(x):+.2f} Peak {peak_db(x):+.2f} 削波 {clip_count(x)}")

before, sr = read_wav_pcm16(LST / "before.wav")
after, _ = read_wav_pcm16(LST / "after.wav")
t0, t1 = int(1.9 * sr), int(3.9 * sr)
gap = np.zeros(int(0.45 * sr))
aba = np.concatenate([before[t0:t1], gap, after[t0:t1], gap, before[t0:t1]])
write_wav_pcm16(LST / "aba_1.9-3.9.wav", aba, sr)
print(f"aba_1.9-3.9.wav {len(aba)/sr:.2f}s")
