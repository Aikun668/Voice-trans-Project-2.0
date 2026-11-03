"""(.venv-f2) 候选2输入：薄音版整体升 310 音分（Praat PSOLA，共振峰保持，复用 f2/poc_phase1.shift_uniform）；
并导出两者 10 ms 的 F0 轨迹，供 harmonic_tilt.py 使用。"""
import sys
from pathlib import Path
import numpy as np
import parselmouth
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "f2"))
from poc_phase1 import read_wav_pcm16, shift_uniform, to_array, write_wav_pcm16  # noqa: E402
HERE = Path(__file__).parent
SRC = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌-薄音版.wav"
KEY_UP = 310.0
x, sr = read_wav_pcm16(SRC)
y = to_array(shift_uniform(parselmouth.Sound(x, sr), KEY_UP))[: len(x)]
y = np.pad(y, (0, len(x) - len(y)))
(HERE / "work").mkdir(exist_ok=True)
write_wav_pcm16(HERE / "work/thin_keyup.wav", y, sr)
for name, sig in [("thin", x), ("thin_keyup", y)]:
    p = parselmouth.Sound(sig, sr).to_pitch(0.01, 75, 600)
    np.savetxt(HERE / f"work/f0_{name}.csv", np.c_[p.xs(), p.selected_array["frequency"]], delimiter=",", fmt="%.5f")
    f = p.selected_array["frequency"]; print(name, "F0 median", round(float(np.median(f[f > 0])), 1))
