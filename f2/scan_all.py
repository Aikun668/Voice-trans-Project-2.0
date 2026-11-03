"""统一验证: 每条本人录音 vs 对应原唱人声轨的匹配位置与置信度。"""
import sys

import numpy as np

from scan2 import cents_of, fix_oct, robust_scan

PAIRS = [
    ("自由飞翔", r"test_dataset\my_singing\shige_medium.wav",
     r"test_dataset\reference\vocals\自由飞翔_凤凰传奇_vocals.wav"),
    ("最炫民族风", r"test_dataset\my_singing\zuixuan_normal.wav",
     r"test_dataset\reference\vocals\最炫民族风_凤凰传奇_vocals.wav"),
    ("平凡之路", r"test_dataset\f1_dataset_v2\A_thin\self\wav\平凡之路正常版.wav",
     r"test_dataset\reference\vocals\平凡之路_朴树_vocals.wav"),
    ("雅俗共赏", r"test_dataset\f1_dataset_v2\A_thin\self\wav\雅俗共赏-正常版.wav",
     r"test_dataset\reference\vocals\雅俗共赏_许嵩_vocals.wav"),
    ("我们的歌", r"test_dataset\f1_dataset_v2\A_thin\self\wav\我们的歌正常版.wav",
     r"test_dataset\reference\vocals\我们的歌_王力宏_vocals.wav"),
    ("刚好遇见你", r"test_dataset\f1_dataset_v2\A_thin\self\wav\因为刚好遇见你-原版.wav",
     r"test_dataset\reference\vocals\刚好遇见你_李玉刚_录音室_vocals.wav"),
]

print(f"{'歌曲':<10} {'原曲位置':>10} {'中位差':>8} {'全曲中位':>8}  判定")
for song, user, ref in PAIRS:
    try:
        q = fix_oct(cents_of(user, 130, 450))
        r = fix_oct(cents_of(ref, 130, 450))
        (cost, s), costs = robust_scan(q, r)
        valid = costs[costs < 1e8]
        base = float(np.median(valid)) if len(valid) else float("nan")
        ok = cost < 120 and cost < base * 0.4
        print(f"{song:<10} {s * 0.05:9.1f}s {cost:8.0f} {base:8.0f}  {'对上了' if ok else '不确定'}")
    except Exception as e:
        print(f"{song:<10} ERR {type(e).__name__}: {e}")
