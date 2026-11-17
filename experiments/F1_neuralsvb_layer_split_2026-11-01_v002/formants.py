"""Read-only formant check on the layer-split outputs (A01/A03).

Measures median F1/F2/F3 on voiced frames for original / vocoder_only / a2a / a2p.
Descriptive only: singing formants are not a validated nasality score.
"""
import json
import math
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "F1_neuralsvb_layer_split_2026-11-01_v002"
SR = 22050
CLIPS = {"A01 平凡之路薄音版": "平凡之路薄音版", "A03 我们的歌薄音版": "我们的歌-薄音版"}
FILES = {
    "original": ("candidate", "__original.wav"),
    "vocoder_only": ("candidate", "__vocoder_only.wav"),
    "a2a": ("diagnostics", "__a2a.wav"),
    "a2p": ("candidate", "__a2p.wav"),
}


def read(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float64")
    if sr != SR:
        g = math.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def formants(x: np.ndarray):
    snd = parselmouth.Sound(x, sampling_frequency=SR)
    form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                               maximum_formant=5500, window_length=0.025)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=80, pitch_ceiling=750,
                            voicing_threshold=0.6)
    times = np.arange(0.02, snd.duration - 0.02, 0.01)
    vals = {1: [], 2: [], 3: []}
    for t in times:
        f0 = pitch.get_value_at_time(t)
        if not np.isfinite(f0) or f0 <= 0:
            continue
        f1 = form.get_value_at_time(1, t)
        f2 = form.get_value_at_time(2, t)
        f3 = form.get_value_at_time(3, t)
        if np.isfinite(f1) and 200 <= f1 <= 900:
            vals[1].append(f1)
        if np.isfinite(f2) and 700 <= f2 <= 2500:
            vals[2].append(f2)
        if np.isfinite(f3) and 1500 <= f3 <= 4000:
            vals[3].append(f3)
    return {k: (round(float(np.median(v)), 1) if len(v) >= 30 else None) for k, v in vals.items()} | \
        {f"n{k}": len(v) for k, v in vals.items()}


out = {}
for label, stem in CLIPS.items():
    out[label] = {}
    for name, (folder, suf) in FILES.items():
        x = read(EXP / folder / f"{stem}{suf}")
        out[label][name] = formants(x)
    print(label)
    for name in ("original", "vocoder_only", "a2a", "a2p"):
        print(" ", name, out[label][name])

(EXP / "formants.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
