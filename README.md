# AI 歌声美化与修音

输入清唱人声，探索让歌声更有力、自然，并改善音准，同时保留本人音色、咬字和演唱表现。

当前状态（2026-11-14）：F2 无参考修音主线已完成 PopBuTFy 清洗和首次试训，Jade 判定没有有用改善。最新实验在独立 MusicBERT 次线：官方现成权重加适配已跑通，未微调，工程通过但效果门槛未过，听感待 Jade。主线尚未确定下一轮方案；目前没有已验收的正式默认修音模型。F1 固定算法调试继续暂停。

- [项目路线和工作规则](docs/ROADMAP.md)
- [这次项目怎么分](docs/项目分区说明_2026-11-14.md)
- [主线试训与失败反馈](experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/REPORT.md)
- [MusicBERT 次线最新结果](experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/REPORT.md)
- [双人分工与学习任务草案](docs/TEAM_COLLABORATION_DRAFT.md)
- [固定算法阶段收尾与试听记录](docs/archive/F1_FIXED_DSP_CLOSEOUT_2026-11-05.md)
- [项目文件说明](docs/FILE_INDEX.md)
- [历史文档归档](archive/README.md)
- [Agent 工作规则](AGENTS.md)

## 目录

| 目录 | 内容 |
|---|---|
| `docs/` | 唯一现行路线、按角色查找的索引、分区说明和历史资料 |
| `archive/` | 早期 PoC 文档原件 |
| `experiments/` | 每轮实验、试听页、脚本、数据和反馈 |
| `test_dataset/` | 原始素材和参考素材 |
| `f1_adaptive_v0/` | 固定 DSP 的历史规则基线 |
| `f2/` | F2 音高试验与记录 |
| `neuralsvb_f1/` | 旧 NeuralSVB 实验适配代码 |
| `output/` | 本机归档 ZIP 和历史浏览器验证产物 |

实验新建 `experiments/<主题>_<日期>_vNNN/`，保留代码、报告、试听材料和反馈。现有实验按“主线、次线、共用资产、历史记录”登记，保留原位置；归档不代表可以删除其中的运行依赖。两个旧工作树保留独有的确认表、素材和历史代码，现行方向统一读本主项目的 ROADMAP。详细来源与权重位置见文件索引，不按目录编号猜当前版本。

技术检查和实际听感分开记录。Agent 无法听音，由本人或试听者作听感判断。F1 不主动改音高，整体升降调由独立的 F2-K 开关控制。
