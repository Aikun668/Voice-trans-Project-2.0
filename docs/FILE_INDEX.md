# 文件索引：按角色找，不按版本号猜

更新：2026-11-14。本文件只管导航；唯一现行路线和授权边界是 [ROADMAP](ROADMAP.md)。路径跨三个工作目录，不能把同名文件全部替换为同一根目录。

## 当前入口

| 入口 | 用途 |
|---|---|
| [ROADMAP](ROADMAP.md) | 当前目标、主次线、工作规则、用户反馈及决策 |
| [分区说明](项目分区说明_2026-11-14.md) | 几句大白话说明这次怎么整理 |
| [版本与引用核查](版本与引用核查_2026-11-14.md) | 原唱引用错误、同名副本差异和建议落实；旧扫描是当时快照 |
| [本轮整理记录](../experiments/F2_PROJECT_ORGANIZATION_2026-11-14_v001/REPORT.md) | 修改清单、文档快照、实验分类、来源登记和核验 |

## 主线与次线产物

| 角色 | 确定入口 | 状态与范围 |
|---|---|---|
| 主线首次无参考试训 | [REPORT](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/REPORT.md)；[用户原话](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/followup_2026-11-14_v001/jade_feedback.md) | 工程通过，效果失败，`jade_rejected`；当前页只有原音和模型诊断版 |
| 主线诊断权重 | [practice_training_v001/model.pt](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/practice_training_v001/model.pt) | SHA256 `d63eea5b…`；56项本人音高上下文特征的轻量模型，未用MusicBERT；不作正式默认 |
| 同轮另外两种训练 | [GRU停止记录](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/training_v001/STOPPED.json)；[直接标签模型](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/context_training_v001/report.json) | 各自的失败/弱收益证据；不因同名权重或修改时间而互换 |
| MusicBERT独立次线 | [REPORT](../experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/REPORT.md)；[预定标准](../experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/rules.md)；[严格复核](../experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/DELIVERY_CHECK.md) | 官方small权重`f5bb685a…`冻结、0微调；工程通过，效果门槛未过，`pending_jade` |
| 次线原音与处理版 | [离线试听页](../experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/delivery_v001/index.html) | 仅次线实验对照；不覆盖主线，不代表已经修准 |

两条线目前都没有通过“有用且自然”的完整验收，没有全项目可直接采用的正式默认模型。

## 共用资产与来源

| 资产 | 来源 | 注意事项 |
|---|---|---|
| PopBuTFy原始数据 | `C:/Users/杨玉璋/Downloads/PopBuTFy` | 只读；英文配对数据不等于本人中文训练数据 |
| 全量清洗和映射 | [清洗报告](../experiments/F2_POPBUTFY_CLEAN_2026-11-13_v001/REPORT.md) | pilot_v001为小批历史探索；全量结果及11-14复核接替其交付角色 |
| 主线复核与划分 | [gate.json](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/data_v001/gate.json)；[label_manifest.json](../experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/data_v001/label_manifest.json) | 粗筛5,643、复核候选4,648、最终训练/验证/测试3,805/442/359是不同阶段，不是人工逐帧真值 |
| 当前本人研究录音 | [三首重录目录](../test_dataset/my_singing/rerecord_2026-11-04_f2/) | 非严重档；已跑两线均用《心墙01-轻度跑调》完整8.612秒作本人输入 |
| 17对确认切口 | [H中confirmed_pairs.json](/C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0/experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001/confirmed_pairs.json)；[确认原话](/C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0/experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001/feedback.md) | 不用重标；确认切口、对应内容及修复意图，不等于每时刻目标音可靠 |
| 手动编辑来源 | [H中v002编辑记录](/C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0/experiments/F2_MANUAL_MARKS_REVIEW_2026-11-09_v002/) | v001已退出切口权威；v002及最终表保留 |
| 每对源文件身份 | [source_bindings.csv](../experiments/F2_PROJECT_ORGANIZATION_2026-11-14_v001/source_bindings.csv) | 17对本人/原唱路径、完整哈希、秒数、采样点及采样率；局部时间差不能外推完整范围 |
| 原始素材与环境 | 三个工作目录的test_dataset、已有环境及第三方依赖 | 原位保留，不凭同名覆盖、不卸载、不统一根路径 |

## 历史归档与仍用依赖

归档是退出当前入口、标明历史、文件保留原位。旧代码仍有调用关系，不整目录搬走。

| 历史对象 | 当前定位 |
|---|---|
| [有参考日志](开发日志_有参考修音.md)、[前三项计划](新阶段_前三项任务初步计划.md)、[首轮训练计划](F2_第一轮有参考修音_训练计划书_给第三方Agent.md) | 历史过程，不重做17对，不从首轮训练重新开始 |
| [双人协作草案](TEAM_COLLABORATION_DRAFT.md) | 人员讨论背景，本次没有替双方确定新的人事分工 |
| [H的旧交接](/C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0/docs/F2_TEAM_HANDOFF.md) | Jade的17对核准已完成；徐同学独立排查是否完成待核实 |
| H的active_pitch_model.json、prepared_v003及后续有参考序列 | 旧有参考基线，权重08e8d7e9…；不代表主线或次线默认模型，旧权重已有归档清单 |
| H的F2_MODEL_LISTEN…v002/run.py、F2_REFERENCE_SEQUENCE…/render_v002 | 仍被旧有参考流程调用，保留代码和数据 |
| I的F2_TASK3_PAIR_AUDIT…/逐字候选对照.csv | H的参考序列仍读取作辅助上下文；不替代最终17对范围 |
| F1_demo_chain…v001/v003、f2/poc_phase1/2/4b/4c等 | 保留母带、音高和拼接等函数；旧P3输出作废不等于可删所有辅助函数 |
| [固定DSP收尾](archive/F1_FIXED_DSP_CLOSEOUT_2026-11-05.md)、[历史入口索引](archive/README.md) | F1旧轮试听、NeuralSVB、WOMENDE、DEMO、旧功能图和10月计划退出当前入口，反馈保留 |
| 主线旧三版delivery/verification/artifact_hashes及检查脚本 | 撤下前证据；当前两版状态读版本核查轮，不能用旧三版检查称当前页已完整通过 |
| [早期PoC原件](../archive/README.md)、output/archives、output/playwright | 历史需求、压缩快照和浏览器证据 |

逐实验归属见[experiment_roles.csv](../experiments/F2_PROJECT_ORGANIZATION_2026-11-14_v001/experiment_roles.csv)，已知调用链见[retained_dependencies.json](../experiments/F2_PROJECT_ORGANIZATION_2026-11-14_v001/retained_dependencies.json)。字符串引用数只是线索，不是可安全搬移证明。

## 三个工作目录

| 简称 | 绝对位置 | 角色 |
|---|---|---|
| M | D:/My_Grand_Projects2026/Voice-trans-Project-2.0 | 当前主项目、唯一现行ROADMAP、两线实验及公开数据处理 |
| H | C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0 | 已确认17对、编辑来源、旧有参考模型与后续序列代码 |
| I | C:/Users/杨玉璋/.codex/worktrees/f2-initial-plan/Voice-trans-Project-2.0 | 早期任务、逐字辅助表和部分确认坐标绑定的原音频 |

两棵旧工作树不另起现行路线。新实验新建目录，不覆盖旧输出。模型和原唱身份按角色、完整哈希及确认来源确定。
