# A3 · MFA / HubertFA 冒烟准备（Agent 已列清单，待本机跑）

目的：只复查「零时长」子集，看能否救回正时长；救不回保持不可用。不是换全家桶。

## 输入清单

见 `results/a3_smoke_manifest.json`（9 条含零时长的音轨版本，优先心墙）。

每条含：source 相对路径、window_start_s、text、zero_indices。

## 建议最小实验

1. 先跑 **心墙01 本人对照**（已知首字「我」零时长，歌词已确认）。
2. 同段再跑 MFA 或 HubertFA（有哪个用哪个；HubertFA 参考 lyric-align 歌声路径）。
3. 对比：该字是否得到正时长；邻字起点是否大跳。
4. 再抽 1 条原唱零时长（如离开我的依赖02 原唱）验证边界。

## 本机注意

- Qwen 权重在：`D:\My_Grand_Projects2026\Videos-Master-Project\Qwen3-ForcedAligner-0.6B`（本次冒烟可不用它）。
- 原始 WAV 只读；结果写到新目录 `experiments/F2_ALIGN_SMOKE_*`。
- 未装 MFA/HubertFA 前不硬跑；装环境另建，勿污染现有 .venv。

## 当前状态

清单已就绪；**尚未在本机执行对齐复查**（缺环境/权重时等你或本机有空再跑）。Jade 的听感坏点结果回来后，可缩小再跑。
