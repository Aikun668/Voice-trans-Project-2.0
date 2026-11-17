"""Read-only descriptive voice-quality metrics (Praat): HNR, jitter, shimmer on voiced stretches. NOT listening scores."""
import sys, json, numpy as np, parselmouth
from parselmouth.praat import call
def metrics(path):
    s = parselmouth.Sound(path)
    if s.sampling_frequency != 22050: s = s.resample(22050)
    hnr = s.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
    v = hnr.values[0]; v = v[v > -200]            # -200 = unvoiced
    pp = call(s, "To PointProcess (periodic, cc)", 80, 750)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shi = call([s, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {"hnr_median_db": round(float(np.median(v)), 1), "hnr_p10_db": round(float(np.percentile(v, 10)), 1),
            "voiced_frames": int(len(v)), "jitter_local": round(float(jit), 4), "shimmer_local": round(float(shi), 4)}
R = {}
base = "F1_neuralsvb_v002_restored_2026-10-26/audio/"
for c in ["molihua", "shige", "zuixuan"]:
    for m in ["original", "vocoder_only", "a2p"]:
        R[f"v002/{c}/{m}"] = metrics(f"{base}{c}_normal__{m}.wav")
b = "F1_neuralsvb_pitch_fix_v001/shige_control/shige_normal__"
for m in ["original", "base_vocoder", "base_a2a", "base_a2p"]:
    R[f"contract_v001/shige/{m}"] = metrics(b + m + ".wav")
json.dump(R, open("F1_neuralsvb_vocoder_audit_v001/voice_quality.json", "w"), ensure_ascii=False, indent=1)
for k, v in R.items(): print(f"{k:34s}", v)
