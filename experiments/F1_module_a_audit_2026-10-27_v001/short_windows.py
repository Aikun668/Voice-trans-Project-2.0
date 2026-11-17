"""Check short crop behavior without modifying any source audio."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_adaptive_v0.analyzer import read_wav, analyze

SRC = ROOT / 'test_dataset/web_singing/gtsinger_alto_control/告白气球_normal.wav'
DEST = Path(__file__).parent / 'results_v001/short_window_check.json'

def mask(intervals, times):
    out = np.zeros(len(times), dtype=bool)
    for interval in intervals:
        out |= (times >= interval['start_seconds']) & (times < interval['end_seconds'])
    return out

def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    source_hash = hashlib.sha256(SRC.read_bytes()).hexdigest()
    x, sr, _ = read_wav(SRC)
    full = analyze(x, sr)
    result = {'file': str(SRC), 'sha256': source_hash, 'windows': [],
              'note': 'Statistics on a shorter excerpt need not equal the full recording. Only compare corresponding time labels; no human-listened ground truth.'}
    for start in (0.5, 2.0):
        end = start + 3.5
        cropped = analyze(x[round(start * sr):round(end * sr)], sr)
        # Compare candidate labels only inside the window, excluding 20ms boundaries.
        times = np.arange(.03, 3.47, .01)
        fullmask = mask(full['sibilance']['candidate_intervals'], times + start)
        cropmask = mask(cropped['sibilance']['candidate_intervals'], times)
        result['windows'].append({'source_start_seconds': start, 'source_end_seconds': end,
            'full_recording_gate_dbfs': full['activity_gate_dbfs'], 'crop_gate_dbfs': cropped['activity_gate_dbfs'],
            'full_dynamic_db': full['dynamic_range_db'], 'crop_dynamic_db': cropped['dynamic_range_db'],
            'candidate_label_mismatch_fraction_in_overlap': float(np.mean(fullmask != cropmask)),
            'note': 'Different summary values can be expected: the clip contains a different distribution of singing.'})
    if source_hash != hashlib.sha256(SRC.read_bytes()).hexdigest():
        raise RuntimeError('Source changed')
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
