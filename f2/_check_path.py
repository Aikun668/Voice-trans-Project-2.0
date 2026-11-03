"""验证 self-ref 对齐质量: 按帧建映射后检查平滑度与斜率。"""
import sys

import numpy as np

from self_ref_align import cents_track, fill, fix_oct, global_dtw

target, reference = sys.argv[1], sys.argv[2]
q = fill(fix_oct(cents_track(target)))
r = fill(fix_oct(cents_track(reference)))
path, cost = global_dtw(q, r)

map_i = np.full(len(q), np.nan)
for i, j in path:
    map_i[i] = j
valid = ~np.isnan(map_i)
map_i[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], map_i[valid])

print(f"目标 {len(q)} 帧 ({len(q)*0.05:.2f}s)  参考 {len(r)} 帧 ({len(r)*0.05:.2f}s)")
print(f"DTW 代价 {cost:.0f} 音分/帧  平均斜率 {np.gradient(map_i).mean():.2f} "
      f"(期望 {len(r)/len(q):.2f})")
print(f"映射范围: {map_i.min():.1f} - {map_i.max():.1f} 帧; 单调递增? {bool(np.all(np.diff(map_i) >= 0))}")
for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    i = int(frac * (len(q) - 1))
    print(f"  目标 {i*0.05:5.1f}s -> 参考 {map_i[i]*0.05:6.1f}s")
