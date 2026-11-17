"""Review one Jade-confirmed pause and the current A->B behavior, in memory."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from f1_adaptive_v0.analyzer import analyze, read_wav
from f1_adaptive_v0.decision_engine import decide, load_rules

SOURCE = ROOT / 'test_dataset/my_singing/shige_normal.wav'
OUTPUT = HERE / 'review_v001.json'
START, END = 5.74, 6.29


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def review(x, sr):
    f = analyze(x, sr)
    d = decide(f, load_rules())
    return {'gate_dbfs': f['activity_gate_dbfs'], 'dynamic_db': f['dynamic_range_db'],
            'active_seconds': f['active_seconds'], 'low_level_seconds': f['low_level']['seconds'],
            'nearby_active': [a for a in f['active_intervals'] if a['end_seconds'] >= 5.5 and a['start_seconds'] <= 6.5],
            'nearby_unknown': [a for a in f['low_level']['candidate_intervals'] if a['end_seconds'] >= 5.5 and a['start_seconds'] <= 6.5],
            'decision': d, 'uncertainty_status': f['activity_assessment']['status']}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    before = digest(SOURCE)
    x, sr, _ = read_wav(SOURCE)
    baseline = review(x, sr)
    start, end = round(START * sr), round(END * sr)
    modified = x.copy()
    modified[start:end] = 0
    zero_pause = review(modified, sr)
    # Approximate boundaries reported by Jade: exclude 20ms at either edge in an independent check.
    inside = x.copy()
    inside[round((START + .02) * sr):round((END - .02) * sr)] = 0
    zero_inside = review(inside, sr)
    if digest(SOURCE) != before:
        raise RuntimeError('Original source changed')
    result = {
        'source': str(SOURCE), 'source_sha256': before,
        'source_annotation': {'start_seconds': START, 'end_seconds': END,
            'user_words': '就是一个简单音乐中间的一个正常的停顿',
            'scope': 'one approximate passage only; not a frame-by-frame segmentation or other samples'},
        'method': 'Original versus in-memory zeroing as counterfactual diagnostic; no audio or other source modified.',
        'baseline': baseline, 'zero_full_approximate_interval': zero_pause, 'zero_inner_interval': zero_inside,
        'dynamic_delta_full_db': zero_pause['dynamic_db'] - baseline['dynamic_db'],
        'dynamic_delta_inner_db': zero_inside['dynamic_db'] - baseline['dynamic_db'],
        'analyzer_code_sha256': digest(ROOT / 'f1_adaptive_v0/analyzer.py'),
        'decision_code_sha256': digest(ROOT / 'f1_adaptive_v0/decision_engine.py'),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
