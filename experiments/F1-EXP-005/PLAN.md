# F1-EXP-005：P3A-2 增强谐波饱和

日期：2026-10-31。状态：pending_jade。

## 原因

P3A 首档（drive 1.5/3/6，wet 0.5）用户听不出 A/B/C 与原音差异。说明该档饱和过轻，或饱和对该素材的峰值作用不足。本轮只提高饱和强度，不改算法家族、不叠加 EQ/压缩。

## 方案

- 仍为 soft tanh：`tanh(drive * x) / drive`。
- 4x oversampling + 20 kHz 低通后下采样；时间对齐；Wet/Dry = 0.7。
- Light/Medium/Strong：drive = 3.0 / 6.0 / 12.0。
- 样本：A01、A05；Holdout 未处理。
- 试听版等响；A/B/C 随机隐藏强度；原始保留。

## 评价

饱满度是否真正提升？是否只是更亮/更响？是否自然、像本人？是否刺耳、沙哑或失真？