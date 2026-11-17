"""复核：同段两版的中位音高差是否恒定（只读，输出仅写本目录）。

背景：F2_R1_ALIGN_AUDIT 的判据写成 `gap - pair_key`，符号反了（pair_key 是
"对照 − 轻度"，gap 是 "轻度 − 对照"），把三个 take 里两个"一致"误判成"矛盾"。
本探针完全不看 DTW，直接比较两版浊音帧的中位音高（每 10 秒一段），
回答一个更基本的问题：两版之间约 500 音分的差，是"整段换了调"还是"局部某几句"。

运行: .venv-f1-adaptive/Scripts/python.exe experiments/F2_R1_AUDIT_FIX_2026-11-07_v001/probe_key_drift.py
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
if (DEST / 'results').exists():
    raise SystemExit('输出已存在，拒绝覆盖')
(DEST / 'results').mkdir()

sys.path.insert(0, str(ROOT / 'experiments/F2_R1_TRAIN_2026-11-07_v003'))
import r1_pipeline as R  # noqa: E402

PAIRS = [('心墙', '01'), ('离开我的依赖', '01'), ('离开我的依赖', '02')]
WIN = 10.0


def circ(v):
    return float((v + 600.0) % 1200.0 - 600.0)


def profile(c):
    """每 WIN 秒的浊音中位音高（音分）。"""
    n = int(WIN / R.HOP)
    out = []
    for i in range(0, len(c), n):
        w = c[i:i + n]
        w = w[~np.isnan(w)]
        out.append(round(float(np.median(w)), 1) if len(w) > 20 else None)
    return out


def main():
    rows = []
    for song, take in PAIRS:
        seg_a = f'{song}{take}-对照微微跑调'
        seg_b = f'{song}{take}-轻度跑调'
        ca, cb = R.cents_track(str(R.seg_path(seg_a))), R.cents_track(str(R.seg_path(seg_b)))
        pa, pb = profile(ca), profile(cb)
        wa, wb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
        diffs = [round(circ(y - x), 1) for x, y in zip(pa, pb) if x is not None and y is not None]
        row = {'song': song, 'take': take,
               'median_对照': round(float(np.median(wa)), 1),
               'median_轻度': round(float(np.median(wb)), 1),
               'median_gap_轻度减对照': round(circ(float(np.median(wb)) - float(np.median(wa))), 1),
               'dur_对照_s': round(len(ca) * R.HOP, 1), 'dur_轻度_s': round(len(cb) * R.HOP, 1),
               'win_gaps': diffs,
               'win_gap_spread': round(float(max(diffs) - min(diffs)), 1) if diffs else None}
        rows.append(row)
        print(f"[{song}{take}] 中位差 {row['median_gap_轻度减对照']:+.0f} 音分"
              f"（长度 {row['dur_对照_s']}s / {row['dur_轻度_s']}s）；"
              f"逐 10s 差 {diffs}，极差 {row['win_gap_spread']}")
    (DEST / 'results' / 'probe.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：results/probe.json')


if __name__ == '__main__':
    main()
