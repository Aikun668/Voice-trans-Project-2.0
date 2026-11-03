"""对清唱版重新定位其在原唱中的位置。"""
import numpy as np

from scan2 import cents_of, fix_oct, robust_scan

q = fix_oct(cents_of(r"test_dataset\f1_dataset_v2\A_thin\self\wav\因为刚好遇见你-清唱版.wav", 130, 450))
r = fix_oct(cents_of(r"test_dataset\reference\vocals\刚好遇见你_李玉刚_录音室_vocals.wav", 130, 450))
(cost, s), costs = robust_scan(q, r)
valid = costs[costs < 1e8]
print(f"清唱版: 位置 {s * 0.05:.1f}s  中位差 {cost:.0f}  全曲中位 {np.median(valid):.0f}")
