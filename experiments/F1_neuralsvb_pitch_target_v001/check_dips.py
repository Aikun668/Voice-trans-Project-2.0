"""What is in the ORIGINAL at the frames where the a2p output collapses? level, zero-crossing, spectral centroid, Praat/pYIN pitch, and a2p level change. Measurements only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa, parselmouth
base = Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001/full_normal/shige_normal'
sr = 22050
def L(n):
    y, s = sf.read(str(base / f'shige_normal__{n}.wav'), dtype='float32'); return librosa.resample(y, orig_sr=s, target_sr=sr)
o, a = L('original'), L('a2p'); voc = L('vocoder')
yp, vf, vp = librosa.pyin(o, fmin=75, fmax=800, sr=sr, frame_length=2048, hop_length=128)
f0 = np.load(base / 'f0_aligned.npy')
def db(x): return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
for lo, hi in [(0.74, 1.06), (16.78, 17.04)]:
    print(f'--- {lo}-{hi}s   t | f0_given | pyin(vprob) | orig dB  zcr  centroid | a2p-orig dB | vocoder-orig dB')
    for t in np.arange(lo, hi, 0.02):
        i, j = int(t * sr), int(t * sr) + 441; k = int(t * sr / 128)
        seg = o[i:j]; zc = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)); cen = float(librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=441, hop_length=441)[0][0])
        print(f'{t:5.2f} | {f0[k]:5.0f} | {np.nan_to_num(yp[k]):5.0f} ({vp[k]:.2f}) | {db(seg):6.1f} {zc:.2f} {cen:6.0f} | {db(a[i:j]) - db(seg):6.1f} | {db(voc[i:j]) - db(seg):6.1f}')
