"""One variable: process the recording as short sentences (cut in quiet gaps, 3-7 s, cuts on 128-sample multiples) instead of one long take.
Everything else identical to the whole-file candidate A (Q_full): same pipeline per chunk (aligned F0, octave+target repair, a2p), chunk outputs
concatenated, then the SAME whole-file level_match + voiced-only colour_match(1.0)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, soundfile as sf, librosa
import common as c, pipeline as p, eval_metrics as em
from dip_repair import frame_db
sr, HOP = 22050, 128
E = c.HERE / 'eval_out' / 'audio'
b = p.Beautifier(); report = {}
def cuts_for(y, lo_s=3.0, hi_s=7.0, goal_s=5.0):
    L = frame_db(y); thr = np.percentile(L, 99) - 28; quiet = L < thr; cuts = [0]; n = len(y)
    while n - cuts[-1] > hi_s * sr:
        a, z = cuts[-1] + int(lo_s * sr), cuts[-1] + int(hi_s * sr); fa, fz = a // HOP, min(z // HOP, len(L) - 1)
        runs, i = [], fa
        while i <= fz:
            if quiet[i]:
                j = i
                while j + 1 <= fz and quiet[j + 1]: j += 1
                if j - i + 1 >= 5: runs.append((i + j) // 2)
                i = j + 1
            else: i += 1
        f = min(runs, key=lambda q: abs(q * HOP - (cuts[-1] + int(goal_s * sr)))) if runs else fa + int(np.argmin(L[fa:fz + 1]))
        cuts.append(f * HOP)
    return cuts + [n]
for nm in ('shige', 'zuixuan'):
    y = librosa.resample(sf.read(str(c.ROOT / f'test_dataset/my_singing/{nm}_normal.wav'), dtype='float32')[0], orig_sr=44100, target_sr=22050)
    a = b.analyse(y, 'aligned'); cuts = cuts_for(a.wav); print(nm, 'cuts (s):', [round(q / sr, 2) for q in cuts], flush=True)
    pieces = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        ac = b.analyse(a.wav[s:e], 'aligned'); tgt, rep = p.build_target(ac, True); base = b.synthesise(ac, tgt)
        pieces.append(base['a2p'][:e - s]); print('  chunk %.2f-%.2f s ok, octave frames %d' % (s / sr, e / sr, rep['octave_evidence']['frames_changed']), flush=True)
    raw = np.concatenate(pieces); raw = np.pad(raw, (0, max(0, len(a.wav) - len(raw))))[:len(a.wav)]
    out, rep = p.postprocess(a, {'a2p': raw}, True, 1.0)
    sf.write(E / f'my_{nm}__chunk_raw.wav', raw.astype(np.float32), sr, subtype='FLOAT'); sf.write(E / f'my_{nm}__S_chunked.wav', out.astype(np.float32), sr, subtype='FLOAT')
    whole = sf.read(str(E / f'my_{nm}__Q_full.wav'), dtype='float64')[0]; k = min(len(a.wav), len(whole), len(out))
    report[nm] = {'cuts_s': [round(q / sr, 2) for q in cuts], 'chunked': em.all_metrics(a.wav[:k], out[:k]), 'whole': em.all_metrics(a.wav[:k], whole[:k])}
    print(nm, 'whole  ', json.dumps({q: report[nm]['whole'][q] for q in ('worst_drop_db', 'hnr_median_db', 'shimmer', 'p95_dev_cents')}), '\n   chunked', json.dumps({q: report[nm]['chunked'][q] for q in ('worst_drop_db', 'hnr_median_db', 'shimmer', 'p95_dev_cents')}), flush=True)
json.dump(report, open('chunked_report.json', 'w'), indent=1); print('DONE', flush=True)
