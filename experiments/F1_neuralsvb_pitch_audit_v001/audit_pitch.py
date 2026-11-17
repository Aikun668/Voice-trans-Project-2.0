"""Pitch-input audit (read-only): Praat (pipeline params) vs pYIN vs harmonic-comb evidence on the ORIGINAL wav.
Usage: python audit_pitch.py IN.wav OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa, parselmouth

wav_path, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
y, sr = sf.read(wav_path, dtype="float32")
if y.ndim > 1: y = y.mean(1)
if sr != 22050: y = librosa.resample(y, orig_sr=sr, target_sr=22050); sr = 22050
hop = 128; step = hop / sr
pr = parselmouth.Sound(y.astype(np.float64), sr).to_pitch_ac(time_step=step, voicing_threshold=0.6, pitch_floor=80, pitch_ceiling=750)
pt = pr.xs(); pf = pr.selected_array["frequency"]
yp, vf, vp = librosa.pyin(y, fmin=60, fmax=800, sr=sr, frame_length=2048, hop_length=hop)
pyt = librosa.times_like(yp, sr=sr, hop_length=hop); pyf = np.nan_to_num(yp)

def at(t, ts, fs):
    return float(fs[np.argmin(abs(ts - t))])

def comb(f0, t, win=0.04):
    """harmonic-comb score: mean log-magnitude at k*f0 minus at (k-0.5)*f0, k=1..8, from a Hann spectrum."""
    i = int(t * sr); n = int(win * sr); seg = y[max(i - n // 2, 0):i + n // 2]
    seg = seg * np.hanning(len(seg)); N = 1 << 15
    S = np.abs(np.fft.rfft(seg, N)); fr = np.fft.rfftfreq(N, 1 / sr)
    g = lambda f: np.log(S[(abs(fr - f) < 0.03 * f)].max() + 1e-9)
    return float(np.mean([g(k * f0) - g((k - .5) * f0) for k in range(1, 9) if k * f0 < 8000]))

# 1) octave jumps in Praat track between adjacent voiced frames
jumps = []
for i in range(1, len(pf)):
    a, b = pf[i - 1], pf[i]
    if a > 0 and b > 0:
        r = b / a
        if r < 0.6 or r > 1.7:
            t = float(pt[i])
            jumps.append({"t": round(t, 3), "praat_from": round(float(a), 1), "praat_to": round(float(b), 1),
                          "pyin_at_t": round(at(t, pyt, pyf), 1),
                          "comb_from_f0": round(comb(a, t), 2), "comb_to_f0": round(comb(b, t), 2)})
# 2) dense look at user-flagged windows
wins = {}
for c in [float(x) for x in sys.argv[3:]]:
    rows = []
    for t in np.arange(c - 0.15, c + 0.15, 0.02):
        p = at(t, pt, pf); q = at(t, pyt, pyf)
        rows.append({"t": round(float(t), 3), "praat": round(p, 1), "pyin": round(q, 1),
                     "comb_praat": round(comb(p, t), 2) if p > 0 else None,
                     "comb_half": round(comb(p / 2, t), 2) if p > 0 else None,
                     "comb_double": round(comb(p * 2, t), 2) if p > 0 else None,
                     "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(y[int(t * sr) - 256:int(t * sr) + 256] ** 2)) + 1e-9)), 1)})
    wins[str(c)] = rows
json.dump({"file": str(wav_path), "sr": sr, "octave_jumps": jumps, "windows": wins}, open(out / "pitch_audit.json", "w"), indent=1)
print(len(jumps), "octave-jump candidates"); [print(j) for j in jumps]
for c, rows in wins.items():
    print("== window", c); [print(r) for r in rows]
