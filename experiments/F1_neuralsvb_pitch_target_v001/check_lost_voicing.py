"""Where do voiced frames get lost in a2p outputs (contract_v001 run), and was the vocoder told 'voiced' there?
Frames: grid i*128/22050 s. Voiced(original/output) = Praat 80-750 Hz, th 0.6, on the 44.1 kHz files. F = f0_aligned.npy (what the vocoder received)."""
import numpy as np, parselmouth, soundfile as sf
from pathlib import Path
base = Path(__file__).resolve().parents[1] / 'F1_neuralsvb_contract_v001/full_normal'
def voiced_on_grid(path, n):
    y, s = sf.read(str(path)); p = parselmouth.Sound(y.astype(float), s).to_pitch_ac(time_step=128 / 22050, voicing_threshold=.6, pitch_floor=80, pitch_ceiling=750)
    t = np.arange(n) * 128 / 22050; idx = np.rint((t - p.x1) / p.dx).astype(int); f = np.zeros(n); ok = (idx >= 0) & (idx < len(p.xs()))
    f[ok] = p.selected_array['frequency'][idx[ok]]; return f > 0
for clip in ('shige_normal', 'zuixuan_normal', 'molihua_normal'):
    d = base / clip; F = np.load(d / 'f0_aligned.npy') > 0; n = len(F)
    Vo = voiced_on_grid(d / f'{clip}__original.wav', n)
    for v in ('vocoder', 'a2a', 'a2p'):
        Vu = voiced_on_grid(d / f'{clip}__{v}.wav', n); lost = Vo & ~Vu; gained = ~Vo & Vu
        print(f'{clip:15s} {v:8s} voiced orig {Vo.sum():5d} out {Vu.sum():5d} | lost {lost.sum():4d} (vocoder told voiced: {(lost & F).sum():4d}, told unvoiced: {(lost & ~F).sum():4d}) | gained {gained.sum():4d}')
    # dropouts in the f0 handed to the vocoder: unvoiced frames sitting inside voiced stretches (gap <= 12 frames)
    gaps = 0; i = 0
    while i < n:
        if not F[i]:
            j = i
            while j + 1 < n and not F[j + 1]: j += 1
            if i > 0 and j + 1 < n and j - i + 1 <= 12: gaps += j - i + 1
            i = j + 1
        else: i += 1
    print(f'{clip:15s} f0 given to vocoder: voiced {F.sum()}, inner dropout frames (<=12) {gaps}')
