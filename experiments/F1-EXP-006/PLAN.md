# F1-EXP-006：P3A-3 固定 drive 的 Wet 诊断

日期：2026-10-31。状态：pending_jade。

## 目的

P3A-1 基本不可听；P3A-2 增强后主要发闷。本轮固定 soft tanh 的 drive=3，仅比较 wet=0.5/0.6/0.7，判断是否存在一个“可听但不闷、饱满有改善”的有效区间。若中间强度仍无重复正向改善，停止当前 tanh 实现并归档。

## 样本与复用

- A01、A05；Holdout 未处理。
- wet=0.5：复用 F1-EXP-004 的 drive=3 raw 输出。
- wet=0.7：复用 F1-EXP-005 的 drive=3 raw 输出。
- wet=0.6：本轮仅生成这一档。
- 算法不变：tanh(drive*x)/drive；4x oversampling；20 kHz 低通；时间对齐。

## 试听与评价

Original + A/B/C 盲听；三档 wet 随机隐藏。分别评价饱满度、清晰度、自然度；不能把变响/变亮/变闷当成功。

## 边界

不引入 EQ、压缩、模型或组合；不碰 Holdout；不继续扫描其他参数。