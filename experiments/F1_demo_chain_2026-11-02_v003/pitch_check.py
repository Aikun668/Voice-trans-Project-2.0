"""音高粗检（.venv-f2, Praat）：处理版逐帧 F0 相对原音的偏差。只覆盖 75–600 Hz。"""
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "f2"))
from poc_phase1 import read_wav_pcm16  # noqa: E402


def f0(path: Path) -> np.ndarray:
    x, sr = read_wav_pcm16(path)
    return parselmouth.Sound(x, sr).to_pitch(0.01, 75.0, 600.0).selected_array["frequency"]


rows = []
for orig in sorted((HERE / "audio").glob("*_orig.wav")):
    a = f0(orig)
    for v in ("a", "b"):
        b = f0(orig.with_name(orig.name.replace("_orig", f"_{v}")))
        n = min(len(a), len(b))
        m = (a[:n] > 0) & (b[:n] > 0)
        c = 1200 * np.log2(b[:n][m] / a[:n][m])
        rows.append({"file": f"{orig.stem.split('_')[0]}_{v}", "voiced_overlap": round(float(m.sum() / max((a[:n] > 0).sum(), 1)), 3),
                     "median_cents": round(float(np.median(c)), 2), "p95_abs_cents": round(float(np.percentile(np.abs(c), 95)), 1)})
(HERE / "pitch_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
for r in rows:
    print(r)
