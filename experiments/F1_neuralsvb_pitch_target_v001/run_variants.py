"""All remaining single-variable candidates on SHIGE and zuixuan, each vs the whole-file candidate A (Q_full). Resumable per file.
 quiet   : input -10 dB (official clips are about -22 dBFS RMS, user recordings about -12)
 lp8k    : input low-passed at 8 kHz (official takes carry little energy above 8 kHz)
 nomap   : a2p with the latent mapping disabled (amateur latent decoded with the target pitch conditions)
 a2a     : plain VAE reconstruction (no mapping, own conditions)
 combo   : quiet + lp8k + nomap
Same pitch repair and same output calibration (level_match + voiced-only colour_match 1.0) in every variant."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa, torch
from scipy.signal import butter, sosfiltfilt
import common as c, pipeline as p
sr = 22050; E = c.HERE / 'eval_out' / 'audio'; b = p.Beautifier()
def synth(a, tgt, way, disable_map):
    pad = (-a.count) % 4; pe = lambda x: np.pad(x, (0, pad), mode='edge')
    ids_self, _ = b._pitch_ids_and_vocoder_f0(pe(a.f0)); ids_tgt, f0v = b._pitch_ids_and_vocoder_f0(pe(tgt))
    mel = torch.tensor(np.pad(a.mel, ((0, pad), (0, 0)), mode='edge'), dtype=torch.float32)[None]; spk = torch.tensor(a.spk, dtype=torch.float32)[None]
    torch.manual_seed(0)
    with torch.no_grad():
        o = b.eng.model(amateur_mel=mel, prof_mel=mel, amateur_pitch=torch.tensor(ids_self, dtype=torch.long)[None], prof_pitch=torch.tensor(ids_tgt, dtype=torch.long)[None],
                        amateur_spk_id=spk, prof_spk_id=spk, a2p_alignment=torch.arange(mel.shape[1], dtype=torch.long)[None], p2a_alignment=None, infer=False,
                        concurrent_ways=['a2a', 'p2p', 'a2p'], disable_map=disable_map)
    m = o[way]['mel_out'][0].detach().numpy()[:a.count]; torch.manual_seed(0)
    w = b.eng.vocoder.spec2wav(m, f0=f0v[:a.count]); return np.pad(w, (0, max(0, len(a.wav) - len(w))))[:len(a.wav)]
sos = butter(8, 8000, btype='low', fs=sr, output='sos')
VARS = {'quiet': dict(gain=-10, lp=False, way='a2p', nomap=False), 'lp8k': dict(gain=0, lp=True, way='a2p', nomap=False), 'nomap': dict(gain=0, lp=False, way='a2p', nomap=True),
        'a2a': dict(gain=0, lp=False, way='a2a', nomap=False), 'combo': dict(gain=-10, lp=True, way='a2p', nomap=True)}
for nm in ('shige', 'zuixuan'):
    y0 = librosa.resample(sf.read(str(c.ROOT / f'test_dataset/my_singing/{nm}_normal.wav'), dtype='float32')[0], orig_sr=44100, target_sr=22050)
    for vn, o in VARS.items():
        f = E / f'my_{nm}__V_{vn}.wav'
        if f.exists(): continue
        y = sosfiltfilt(sos, y0).astype(np.float32) if o['lp'] else y0
        y = (y * 10 ** (o['gain'] / 20)).astype(np.float32)
        a = b.analyse(y, 'aligned'); tgt, _ = p.build_target(a, True); raw = synth(a, tgt, o['way'], o['nomap'])
        out, _ = p.postprocess(a, {'a2p': raw}, True, 1.0); out = out * 10 ** (-o['gain'] / 20)
        sf.write(f, out.astype(np.float32), sr, subtype='FLOAT'); print(nm, vn, 'done', flush=True)
print('DONE', flush=True)
