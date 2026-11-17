"""Descriptive tremble/stability metrics on the same sentences: gt_a (amateur input), gt_p (pro target), official NSVB, ours pair, ours single.
fast_f0: std (cents) of F0 minus its 60ms moving average (fast pitch wobble). fast_amp: std (dB) of 10ms-frame level minus 100ms moving average.
follow_gtp: median |cents| to gt_p F0 (index-aligned, frames voiced in both). Measurements only, not listening scores."""
import json, numpy as np, soundfile as sf, librosa
from pathlib import Path
E = Path(__file__).resolve().parents[1]; sr = 22050
def load(p): return sf.read(p, dtype='float32')[0]
def f0(y):
    f, v, _ = librosa.pyin(y, fmin=75, fmax=800, sr=sr, frame_length=2048, hop_length=128); return np.where(v, f, np.nan)
def fast_f0(f):
    c = 1200*np.log2(f/np.nanmedian(f)); ok = ~np.isnan(c); out = []
    i = 0
    while i < len(c):                       # per voiced run >= 15 frames
        if ok[i]:
            j = i
            while j+1 < len(c) and ok[j+1]: j += 1
            if j-i+1 >= 15:
                s = c[i:j+1]; k = 11; sm = np.convolve(np.pad(s, k//2, mode='edge'), np.ones(k)/k, 'valid'); out.append((s-sm)[k:-k])
            i = j+1
        else: i += 1
    return float(np.concatenate(out).std()) if out else float('nan')
def fast_amp(y):
    h = 220; r = 20*np.log10(librosa.feature.rms(y=y, frame_length=512, hop_length=h)[0]+1e-6); act = r > r.max()-35
    sm = np.convolve(np.pad(r, 5, mode='edge'), np.ones(11)/11, 'valid')[:len(r)]; return float((r-sm)[act].std())
rows = {}
sets = [('en', 'English', [1, 2, 3]), ('zh', 'Chinese', [1, 2, 3, 4, 5])]
for tag, lang, ids in sets:
    for i in ids:
        Y = {'gt_a': load(E/'official'/f'{lang}_gt_a_{i}.wav'), 'gt_p': load(E/'official'/f'{lang}_gt_p_{i}.wav'), 'official': load(E/'official'/f'{lang}_NSVB_{i}.wav'),
             'ours_pair': load(E/'ours_fixed'/f'{tag}{i}_P_a2p_pair.wav'), 'ours_old_pair': load(E/('ours_en' if tag == 'en' else 'ours')/f'{tag}{i}_P_a2p_pair.wav')}
        F = {k: f0(v) for k, v in Y.items()}; r = {}
        for k in Y:
            n = min(len(F[k]), len(F['gt_p'])); m = ~np.isnan(F[k][:n]) & ~np.isnan(F['gt_p'][:n])
            r[k] = {'fast_f0_cents': round(fast_f0(F[k]), 1), 'fast_amp_dB': round(fast_amp(Y[k]), 2),
                    'follow_gtp_cents': round(float(np.median(np.abs(1200*np.log2(F[k][:n][m]/F['gt_p'][:n][m])))), 0) if m.sum() > 20 else None}
        rows[f'{tag}{i}'] = r; print(f'{tag}{i}', ' | '.join(f"{k}: f0 {v['fast_f0_cents']} amp {v['fast_amp_dB']} follow {v['follow_gtp_cents']}" for k, v in r.items()), flush=True)
agg = {k: {m: round(float(np.nanmean([rows[c][k][m] for c in rows if rows[c][k][m] is not None])), 2) for m in ('fast_f0_cents', 'fast_amp_dB', 'follow_gtp_cents')} for k in ('gt_a', 'gt_p', 'official', 'ours_pair', 'ours_old_pair')}
print('MEAN', json.dumps(agg)); json.dump({'rows': rows, 'mean': agg}, open(E/'contract_check'/'tremble_fixed.json', 'w'), indent=1)
