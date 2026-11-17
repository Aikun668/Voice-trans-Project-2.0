# 文件整理索引

2026-11-05 整理，当前规则和决策以 [ROADMAP](ROADMAP.md) 为准。此表帮助找到资料。

| 文件或目录 | 用途和状态 |
|---|---|
| 根目录 README、AGENTS | 当前入口和工作规则。整理前的 README 原样保存到根目录 archive。 |
| `archive/PoC_2026-10/` 下 PoC 需求 v1.2 | 一周可行性验证的原始目标。项目已进入后续阶段。 |
| 同目录双人协作规范 v1.1 | 当时的杨负责 F1、徐负责 F2 的分工。新分工见 [讨论稿](TEAM_COLLABORATION_DRAFT.md)。 |
| 同目录 F1/F2 接口对齐契约 v1.0 | 当时约定 WAV 和模块边界，采样率等规格尚未确定。后续由双方实测商定。 |
| 同目录技术调研报告 v1.1 | 2026-10-24 的 FFmpeg 和 python-auto-tune PoC 调研，技术选型仅反映当时。 |
| `docs/ROADMAP.md` | 唯一现行路线和决策日志。 |
| `docs/开发日志_有参考修音.md` | 新阶段待办、素材初查和持续更新的工作记录。 |
| `docs/TEAM_COLLABORATION_DRAFT.md` | 新阶段任务、学习建议和交接方式，待两人讨论。 |
| `docs/archive/F1_FIXED_DSP_CLOSEOUT_2026-11-05.md` | 最近的固定算法阶段总结、反馈与复用路径。 |
| `docs/F1_FEATURE_MAP.md`、`docs/F1_FEATURE_MAP.html` | 旧功能锚定图，历史参考。 |
| `docs/F1_LISTENING_ACCEPTANCE.md` | 旧听感验收约定，历史参考。 |
| `docs/F1_REBASE_2026-10-27.md` | 2026-10-27 路线收尾，历史参考。 |
| `docs/superpowers/plans/` | 10 月各项实现计划，历史执行记录。 |
| `docs/archive/` | 原 README、旧 Agent 规则和阶段收尾。 |
| `f1_adaptive_v0/README.md` | 固定规则 V0 的运行说明，作为历史实现使用。 |
| `neuralsvb_f1/README.md` | 旧版外部 NeuralSVB 权重试验说明。 |
| `f2/PHASE1_NOTES.md`、`f2/PHASE2_NOTES.md` | F2 早期实验记录，可供理解现有代码。 |
| `experiments/` | 各轮实验目录和原始反馈。旧结果与其中的 README 按原样保留。 |
| `output/archives/` | 固定 DSP 阶段本机压缩快照和校验值。 |
| `output/playwright/` | 历史浏览器检查截图和日志。 |
| `test_dataset/` | 原始录音、参考和现有目录资料。素材盘点和训练划分待后续进行。 |

新阶段继续按主题日期新建实验目录，不需要重命名旧实验。现有 `output/` 文件保持原路径，后续实验产物放在新的实验目录中；实际模型权重和数据输出位置在训练方案中决定。
