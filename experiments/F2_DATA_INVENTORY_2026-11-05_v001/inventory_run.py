"""Read-only inventory and tentative first-song reference alignment.

Writes only into a new results/ directory; never changes source audio.
Run: .venv-f1-adaptive/Scripts/python.exe experiments/F2_DATA_INVENTORY_2026-11-05_v001/inventory_run.py
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import soundfile as sf
sys.path.append(str(Path(__file__).resolve().parents[2] / '.venv-f2/Lib/site-packages'))
import parselmouth

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent / 'results'
if DEST.exists():
    raise SystemExit(f'输出目录已存在，拒绝覆盖: {DEST}')
DEST.mkdir()
sys.path.insert(0, str(ROOT / 'f2'))
from poc_phase4_realign import cost_matrix, subseq_dtw

BASE = ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2'
targets = sorted(p for p in (BASE / 'wav').glob('*.wav') if '严重' not in p.name)
references = sorted((BASE / 'ref_vocals').glob('*.wav'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inspect(p):
    info = sf.info(str(p))
    # Scan in blocks to avoid loading the full reference into memory.
    peak = 0.0
    clipped = 0
    for x in sf.blocks(str(p), blocksize=262144, dtype='float32', always_2d=True):
        peak = max(peak, float(np.max(np.abs(x))))
        clipped += int(np.count_nonzero(np.abs(x) >= 0.999969))
    return {'path': str(p.relative_to(ROOT)), 'seconds': round(info.duration, 3),
            'rate': info.samplerate, 'channels': info.channels, 'subtype': info.subtype,
            'sample_peak': round(peak, 6), 'near_full_scale_samples': clipped, 'sha256': sha(p)}

items = [inspect(p) for p in targets + references]

def cents(p):
    y, sr = sf.read(str(p), dtype='float64', always_2d=True)
    y = y.mean(axis=1)
    hz = parselmouth.Sound(y, sr).to_pitch(time_step=0.10, pitch_floor=75, pitch_ceiling=600).selected_array['frequency']
    result = np.full(len(hz), np.nan)
    valid = hz > 0
    result[valid] = 1200 * np.log2(hz[valid] / 440)
    return result

# Candidate location only; the existing pitch-DTW approach is not lyric verification.
reference = BASE / 'ref_vocals/心墙_郭静_vocals.wav'
ref = cents(reference)
alignments = []
for target in targets:
    if not target.name.startswith('心墙'):
        continue
    q = cents(target)
    best = None
    for key in range(-600, 600, 100):
        cost, path = subseq_dtw(cost_matrix(q, ref, float(key)))
        if best is None or cost < best[0]:
            best = (cost, path, key)
    cost, path, key = best
    reverse, _ = subseq_dtw(cost_matrix(q, ref[::-1].copy(), float(key)))
    alignments.append({'target': str(target.relative_to(ROOT)), 'reference': str(reference.relative_to(ROOT)),
                       'reference_start_s': round(path[0][1] * .1, 2),
                       'reference_end_s': round(path[-1][1] * .1, 2),
                       'coarse_key_cents': key, 'cost': round(cost, 2), 'reversed_cost': round(reverse, 2),
                       'status': '自动候选，歌词和唱法未确认；不可作为训练标签'})

unchanged = all(sha(ROOT / i['path']) == i['sha256'] for i in items)
report = {'items': items, 'first_song_alignment_candidates': alignments,
          'source_hashes_unchanged': unchanged,
          'notes': ['只扫描 WAV 文件，不修改音频；没有生成音频。', '数值与自动对齐不能证明唱准、同字或听感。', '排除文件名含严重的版本。']}
(DEST / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'items': len(items), 'source_hashes_unchanged': unchanged, 'alignments': alignments}, ensure_ascii=False, indent=2))
