"""V3 额外检查：齿音段相对能量、句间底噪、动态起伏（.venv-f1-adaptive）。"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent


def frames(x, sr, n=0.02):
    k = int(sr * n)
    m = len(x) // k
    return 10 * np.log10((x[: m * k].reshape(m, k) ** 2).mean(1) + 1e-12)


rows = []
for f in sorted((HERE / "audio").glob("*.wav")):
    x, sr = sf.read(str(f))
    full = frames(x, sr)
    hi = frames(sosfilt(butter(4, [5000, 9000], "bandpass", fs=sr, output="sos"), x), sr)
    loud = full > np.percentile(full, 95) - 30
    sib = hi - full
    rows.append({"file": f.name,
                 "sibilance_p99_db": round(float(np.percentile(sib[loud], 99)), 1),   # 5–9 kHz 相对全带，最刺的 1% 帧
                 "gap_floor_rel_db": round(float(np.percentile(full, 5) - np.percentile(full, 95)), 1),
                 "dynamics_p95_p30_db": round(float(np.percentile(full[loud], 95) - np.percentile(full[loud], 30)), 1)})
(HERE / "qa_extra.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
for r in rows:
    print(r)
