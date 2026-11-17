# 步骤3 试听页准备：把 sanity 的两条音频用固定增益匹配到 -18 LUFS。
# 响度测量与导出沿用项目 f1_poc.py 的函数，不重写算法。
# 运行（系统 Python，需 ffmpeg 在 PATH）：
#   python sanity_prep.py
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))  # 项目根，导入 f1_poc
import f1_poc  # noqa: E402

TARGET_LUFS = -18.0

PAIRS = [
    ("downloads/bench_noisy_0001_singing.wav", "sanity/01_noisy_matched.wav"),
    ("sanity/bench_noisy_0001_enhanced.wav", "sanity/02_smule_enhanced_matched.wav"),
]

for src_rel, dst_rel in PAIRS:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    rate, data = wavfile.read(src)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        raise ValueError(f"{src_rel} 不是单声道，先查原因")
    measured = f1_poc.measure_loudness(data, rate)
    gain = TARGET_LUFS - measured["integrated_lufs"]
    f1_poc.export_wav(data, rate, gain, dst)
    print(f"{src_rel} -> {dst_rel}: {measured['integrated_lufs']:.2f} LUFS, "
          f"TP {measured['true_peak_dbtp']:.2f} dBTP, gain {gain:+.2f} dB")
print("done")
