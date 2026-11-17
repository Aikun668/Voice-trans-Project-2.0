"""Read-only: where Praat(pipeline) says unvoiced but pYIN says voiced with real energy, and Praat-vs-pYIN octave disagreements."""
import sys, json, numpy as np, soundfile as sf, librosa, parselmouth
y, sr = sf.read(sys.argv[1], dtype="float32"); hop = 128
if y.ndim > 1: y = y.mean(1)
if sr != 22050: y = librosa.resample(y, orig_sr=sr, target_sr=22050); sr = 22050
pr = parselmouth.Sound(y.astype(np.float64), sr).to_pitch_ac(time_step=hop/sr, voicing_threshold=0.6, pitch_floor=80, pitch_ceiling=750)
pt, pf = pr.xs(), pr.selected_array["frequency"]
yp, vf, _ = librosa.pyin(y, fmin=60, fmax=800, sr=sr, frame_length=2048, hop_length=hop)
n = min(len(pf), len(yp)); pyt = librosa.times_like(yp, sr=sr, hop_length=hop)
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
def runs(mask, ts):
    out=[]; i=0
    while i < len(mask):
        if mask[i]:
            j=i
            while j+1 < len(mask) and mask[j+1]: j+=1
            out.append((round(float(ts[i]),2), round(float(ts[j]),2), j-i+1)); i=j+1
        else: i+=1
    return out
pf_i = np.interp(pyt[:n], pt, pf)  # praat on pyin grid
loud = 20*np.log10(rms[:n]+1e-9) > -40
drop = loud & vf[:n] & (pf_i < 1)
octv = (pf_i > 1) & vf[:n] & (np.abs(1200*np.log2(np.maximum(pf_i,1)/np.nan_to_num(yp[:n],nan=1))) > 600)
print("frames", n, "| praat-unvoiced but pyin-voiced & loud:", int(drop.sum()), "| octave-ish disagreement:", int(octv.sum()))
print("dropout runs (start,end,frames) len>=3:", [r for r in runs(drop, pyt) if r[2] >= 3])
print("octave-disagree runs len>=2:", [r for r in runs(octv, pyt) if r[2] >= 2])
