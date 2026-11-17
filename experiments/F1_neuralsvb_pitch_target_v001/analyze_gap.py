"""How does the professional F0 differ from the amateur F0 (same song, DTW-aligned)? Measurements only.
Decomposition of d = cents(pro_warped) - cents(amateur) on frames voiced in both:
  key offset (median), slow part (145 ms moving average of the remainder), fast part (the rest).
Also: dropouts (unvoiced frames inside voiced stretches) in amateur vs pro, and fast-variation level of each."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as c

def inner_gaps(f0, max_gap=12):
    """Count unvoiced frames that sit in gaps <= max_gap between voiced frames."""
    v = f0 > 0; n = 0; i = 0
    while i < len(v):
        if not v[i]:
            j = i
            while j + 1 < len(v) and not v[j + 1]: j += 1
            if i > 0 and j + 1 < len(v) and (j - i + 1) <= max_gap: n += j - i + 1
            i = j + 1
        else: i += 1
    return n, int(v.sum())

def fast_level(f0, width=11):
    ct = c.cents(f0); ok = ~np.isnan(ct); vals = []; i = 0
    while i < len(ct):
        if ok[i]:
            j = i
            while j + 1 < len(ct) and ok[j + 1]: j += 1
            if j - i + 1 >= 15:
                s = ct[i:j + 1]; vals.append((s - c.moving_average(s, width))[width:-width])
            i = j + 1
        else: i += 1
    return float(np.concatenate(vals).std()) if vals else float('nan')

rows = {}
for tag, cid in c.all_clips():
    d = c.LANGS[tag]
    _, ma, fa = c.features(c.clip_path(d, 'gt_a', cid)); _, mp, fp = c.features(c.clip_path(d, 'gt_p', cid))
    al = c.dtw_alignment(fa, fp); fpw = c.warp_pro_to_amateur(fp, al, len(fa))
    both = (fa > 0) & (fpw > 0); diff = c.cents(fpw)[both] - c.cents(fa)[both]
    g = float(np.median(diff)); r = diff - g
    slow = c.moving_average(r, 25); fast = r - slow
    ga, va = inner_gaps(fa); gp, vp = inner_gaps(fp)
    rows[f'{tag}{cid}'] = {
        'split': c.split_of(tag, cid), 'frames_a': len(fa), 'frames_p': len(fp), 'both_voiced': int(both.sum()),
        'key_offset_cents': round(g, 1), 'median_abs_after_offset': round(float(np.median(np.abs(r))), 1),
        'slow_std': round(float(slow.std()), 1), 'fast_std': round(float(fast.std()), 1),
        'amateur_fast_level': round(fast_level(fa), 1), 'pro_fast_level': round(fast_level(fp), 1),
        'amateur_inner_gap_frames': ga, 'amateur_voiced': va, 'pro_inner_gap_frames': gp, 'pro_voiced': vp}
    print(f'{tag}{cid:>2} off {g:7.1f} | resid med|.| {rows[f"{tag}{cid}"]["median_abs_after_offset"]:6.1f} slow {slow.std():6.1f} fast {fast.std():5.1f} | vib-ish a/p {rows[f"{tag}{cid}"]["amateur_fast_level"]:5.1f}/{rows[f"{tag}{cid}"]["pro_fast_level"]:5.1f} | gaps a {ga}/{va} p {gp}/{vp}', flush=True)
M = lambda k: round(float(np.nanmean([r[k] for r in rows.values()])), 1)
summary = {k: M(k) for k in ['key_offset_cents', 'median_abs_after_offset', 'slow_std', 'fast_std', 'amateur_fast_level', 'pro_fast_level']}
summary['amateur_gap_rate'] = round(sum(r['amateur_inner_gap_frames'] for r in rows.values()) / sum(r['amateur_voiced'] for r in rows.values()), 4)
summary['pro_gap_rate'] = round(sum(r['pro_inner_gap_frames'] for r in rows.values()) / sum(r['pro_voiced'] for r in rows.values()), 4)
print('SUMMARY', json.dumps(summary)); json.dump({'rows': rows, 'summary': summary}, open('gap_analysis.json', 'w'), indent=1)
