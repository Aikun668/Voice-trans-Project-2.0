# F1-EXP-005 结果（negative_audible_but_muffled）

Soft Saturation 三强度已生成；A01/A05；Holdout 未处理。技术检查见 qa/chain_checks.json。

## 盲听映射与实际听感

- A01：A=Medium（drive 6），B=Light（drive 3），C=Strong（drive 12）。
  - 用户听感：A 有一点点变化；B 中等程度偏闷；C 更闷，像“完全闷在里面”。
- A05：A=Light（drive 3），B=Strong（drive 12），C=Medium（drive 6）。
  - 用户听感：A 与原音几乎无区别；B、C 分别有不同程度闷感。

## 结论

- P3A 首档（drive 1.5/3/6，wet 0.5）听不出差异；P3A-2 提高 wet 到 0.7、drive 到 3/6/12 后，变化可听，但主要表现是“变闷”，不是饱满。
- 技术检查通过：无削波、长度不变、时间对齐 lag=0、等响；因此不是处理链故障。
- 饱和在这个实现/参数下会造成明显的峰值压缩和音色变闷；A01/A05 的感知强度顺序不完全随 drive 单调，存在样本/响度/顺序影响。
- 当前饱和方向不能证明可改善单薄感；状态改为 negative_audible_but_muffled，暂不继续加 drive/wet。
