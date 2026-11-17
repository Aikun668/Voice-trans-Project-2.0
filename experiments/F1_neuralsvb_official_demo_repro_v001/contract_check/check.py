"""Objective contract check (read-only, no listening claims):
 (1) vocoder round trip: mel -> vocoder -> audio -> mel' ; L1(mel', mel) and signed per-band bias
 (2) acoustic model a2a: L1(a2a_mel, mel) and per-band bias, with our F0/pitch inputs
Uses the same official feature path as the F1 pipeline. Tells whether the vocoder / model reproduce the feature domain."""
import sys, json
from pathlib import Path
import numpy as np, soundfile as sf, torch
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; ROOT = EXP.parents[0]
sys.path.insert(0, str(ROOT.parent / 'neuralsvb_f1')) if False else sys.path.insert(0, str(Path('D:/My_Grand_Projects2026/Voice-trans-Project-2.0/neuralsvb_f1')))
import neuralsvb_infer as n
clips = {f'en{i}': str(EXP / 'official' / f'English_gt_a_{i}.wav') for i in (1, 2, 3)}
clips.update({f'zh{i}': str(EXP / 'official' / f'Chinese_gt_a_{i}.wav') for i in (1, 2)})
clips['shige_0_8s'] = str(HERE / 'shige_first8s.wav')
y, s = sf.read('D:/My_Grand_Projects2026/Voice-trans-Project-2.0/test_dataset/my_singing/shige_normal.wav'); sf.write(clips['shige_0_8s'], y[:s * 8], s)
eng = n.NeuralSVB(); groups = {'0.05-0.5k': (50, 500), '0.5-1.5k': (500, 1500), '1.5-4k': (1500, 4000), '4-11k': (4000, 11025)}
import librosa
fr = librosa.mel_frequencies(80, fmin=50, fmax=11025)
def bias(d):  # d: [T,80] signed diff; mean per band
    return {k: round(float(d[:, (fr >= a) & (fr < b)].mean()), 3) for k, (a, b) in groups.items()}
res = {}
for name, p in clips.items():
    wav, mel, f0, spk = eng.features(p); T = len(mel) - len(mel) % 4
    mel, f0 = mel[:T], f0[:T]
    out = {}
    # (1) vocoder round trip with raw F0
    torch.manual_seed(0); w = eng.vocoder.spec2wav(mel, f0=f0)
    tmp = HERE / f'_{name}_voc.wav'; sf.write(tmp, w, 22050); _, mel2, _, _ = eng.features(str(tmp)); tmp.unlink()
    L = min(len(mel2), T); d = mel2[:L] - mel[:L]
    out['vocoder_roundtrip'] = {'L1_log10': round(float(np.abs(d).mean()), 3), 'bias': bias(d)}
    # (2) a2a with own pitch (identity f0 standardisation as in single-WAV F1)
    pitch, _ = eng._pitch_inputs(f0)
    mt = torch.FloatTensor(mel)[None]; pt = torch.LongTensor(pitch)[None]; st = torch.FloatTensor(spk)[None]; al = torch.arange(T, dtype=torch.long)[None]
    torch.manual_seed(0)
    with torch.no_grad():
        o = eng.model(amateur_mel=mt, prof_mel=mt, amateur_pitch=pt, prof_pitch=pt, amateur_spk_id=st, prof_spk_id=st, a2p_alignment=al, p2a_alignment=None, infer=False, concurrent_ways=['a2a', 'p2p', 'a2p'], disable_map=False)
    for way in ('a2a', 'a2p'):
        m = o[way]['mel_out'][0].numpy(); d = m[:T] - mel[:T]
        out[way + '_mel_vs_input'] = {'L1_log10': round(float(np.abs(d).mean()), 3), 'bias': bias(d)}
    res[name] = out; print(name, json.dumps(out, ensure_ascii=False), flush=True)
json.dump(res, open(HERE / 'contract_check.json', 'w'), ensure_ascii=False, indent=1)
