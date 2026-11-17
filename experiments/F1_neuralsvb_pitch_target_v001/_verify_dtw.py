import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, common as c
for tag, cid in [('zh', 1), ('zh', 4), ('en', 1), ('en', 2)]:
    d = c.LANGS[tag]; _, _, fa = c.features(c.clip_path(d, 'gt_a', cid)); _, _, fp = c.features(c.clip_path(d, 'gt_p', cid))
    a, b = c.dtw_alignment(fa, fp), c.dtw_alignment_official(fa, fp)
    print(tag, cid, len(fa), len(fp), 'identical:', bool(np.array_equal(a, b)), flush=True)
a = c.dtw_alignment(*[c.features(c.clip_path('Chinese', k, 3))[2] for k in ('gt_a', 'gt_p')]); print('zh3 chunked ok', len(a))
