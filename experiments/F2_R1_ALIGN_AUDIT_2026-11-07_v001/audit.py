"""P1 排查 · 同段两版调差一致性审计（只读，输出仅写入本目录）。

事实：同一段歌词唱两遍（对照/轻度），相对同一原唱的调差应几乎一致。
实测（v003）：心墙01 两版 -431 vs +125（差 556）；离开01 两版 +548 vs +47（差 501）。
本脚本做两件事：
1. 两版互相对齐（同一人同一段，最可靠的对齐），量出两版之间的真实调差；
2. 结合各自对原唱的调差，判断哪一版对原唱的对齐是错的。

运行: .venv-f1-adaptive/Scripts/python.exe experiments/F2_R1_ALIGN_AUDIT_2026-11-07_v001/audit.py
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
import r1_pipeline as R  # noqa: E402  (含 f2 路径与常量)
from poc_phase4_realign import cost_matrix, subseq_dtw, HOP  # noqa: E402

# v003 实测的各版相对原唱调差（音分）
REF_KEYS = {
    '心墙01-对照微微跑调': -431, '心墙01-轻度跑调': 125,
    '离开我的依赖01-对照微微跑调': 548, '离开我的依赖01-轻度跑调': 47,
    '离开我的依赖02-对照微微跑调': 381, '离开我的依赖02-轻度跑调': 396,
    '莫问归期01-轻度跑调': 570, '莫问归期02-轻度跑调': 123,
}


def circ(v):
    return float((v + 600.0) % 1200.0 - 600.0)


def align_key(path_a, path_b):
    """A 对 B 的子序列 DTW 调差（A 相对 B 唱高/唱低多少音分）。"""
    qa, qb = R.cents_track(str(path_a)), R.cents_track(str(path_b))
    best = None
    for k in range(-600, 600, 10):
        cost, path = subseq_dtw(cost_matrix(qa, qb, float(k)))
        if best is None or cost < best[0]:
            best = (cost, path, float(k))
    cost, path, key = best
    qv = qa[~np.isnan(qa)]
    rev, _ = subseq_dtw(cost_matrix(qa, qb[::-1].copy(), key))
    return {'key_cents': round(circ(key), 1), 'cost': round(cost, 1), 'reversed_cost': round(rev, 1),
            'match_frames': len(path)}


def main():
    out = []
    takes = [('心墙', '01'), ('离开我的依赖', '01'), ('离开我的依赖', '02'), ('莫问归期', '01'), ('莫问归期', '02')]
    for song, take in takes:
        seg_a = f'{song}{take}-对照微微跑调'
        seg_b = f'{song}{take}-轻度跑调'
        if seg_a not in REF_KEYS or seg_b not in REF_KEYS:
            continue
        pair = align_key(R.seg_path(seg_a), R.seg_path(seg_b))
        gap = circ(REF_KEYS[seg_b] - REF_KEYS[seg_a])
        verdict = '一致' if abs(gap - pair['key_cents']) < 60 else '矛盾：两版对原唱的调差之差 ≠ 两版互相对齐的调差'
        row = {'song': song, 'take': take,
               'pair_key_cents': pair['key_cents'], 'pair_cost': pair['cost'],
               'pair_reversed_cost': pair['reversed_cost'],
               'ref_key_对照': REF_KEYS[seg_a], 'ref_key_轻度': REF_KEYS[seg_b],
               'ref_key_gap': round(gap, 1), 'verdict': verdict}
        out.append(row)
        print(f"[{song}{take}] 两版互对齐调差 {pair['key_cents']:+.0f}（代价 {pair['cost']:.0f} vs 反转 {pair['reversed_cost']:.0f}）；"
              f"对原唱调差 对照 {REF_KEYS[seg_a]:+d} / 轻度 {REF_KEYS[seg_b]:+d}，差 {gap:+.0f} → {verdict}")
    (DEST / 'results' / 'audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')
    print('审计完成：results/audit.json（源文件只读，未修改任何音频）')


if __name__ == '__main__':
    main()
