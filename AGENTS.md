# AGENTS · 入口与硬规则（2026-11-14 入口同步）

**先读 [docs/ROADMAP.md](docs/ROADMAP.md)**。它是本项目的最高书面依据：目标、已确认偏好、已证实结论、路线阶段、接手后的第一个任务、工作方式、工程坑。
优先级：Jade 当前原话 > docs/ROADMAP.md > 本文件 > 需求原件 > 其他历史文档。
旧版 AGENTS.md（"每轮只验证一项""最多一次修订"等流程规则）已备份到 docs/archive/AGENTS_before_roadmap_2026-11-03.md，**不再执行**。

## 当前状态（一句话）

2026-11-14，无参考修音主线首次试训被 Jade 判为无有用改善；独立 MusicBERT 次线已用冻结官方权重完成《心墙》实验，零微调，工程通过、效果门槛未过、听感待 Jade。最新实验在次线，主次关系不变；没有已验收的正式默认模型，不自动进入重训或微调。17 对切口已确认，不重复要求标注；源文件与秒数按哈希绑定。F1 固定 DSP 仍暂停，老师 demo 等 Jade 点名。入口和分区见 [ROADMAP §5](docs/ROADMAP.md#5-分阶段路线) 与[文件索引](docs/FILE_INDEX.md)。

## 硬规则（任何时候都适用）

1. **原始素材只读**。test_dataset/ 和旧实验输出都不覆盖、不删除、不搬移。每次实验新建 `experiments/<主题>_<日期>_vNNN/`，脚本遇到已存在的输出目录要拒绝运行。作废结果移到 `_superseded_<日期>/`。
2. **环境隔离**。不往现有 .venv-* 里装包；需要新依赖就新建环境，或放进实验目录的 vendor/。环境表见 ROADMAP §8.1。
3. **不推送 Git、不提交**，除非 Jade 明确要求。
4. **Agent 不能听音**，必须明说。不能把测量变化、测试通过说成"变好听"。工程结果记 technical_pass，听感记 pending_jade，直到 Jade 给出判定。
5. **否决项永久有效**，除非 Jade 主动改口：混响、双叠合唱、硬吸附修音（电音）、谐波激励（录音机感）。
6. **F1 不改音高**。整体升调只能通过独立的 F2-K 音区开关实现，默认关闭，并且要写明。
7. **每个输出文件做工程检查**（ROADMAP §8.2）：样本数、削波、实测 LUFS 与真峰值、音高偏差与浊音重合、源文件哈希。不为交付放宽门槛。
8. **试听负担**：一轮一首歌，最多 3–4 个处理版 + "原音·同样响"锚点，也可只交原音与一个处理版；总时长 ≤ 10 分钟。版本标名字，题目用日常话、给选项，反馈直接复制，不下载文件（格式见 ROADMAP §7.2）。
9. **Jade 原话原样存档**到该轮目录，并追加到 ROADMAP §11 反馈总表；方向性决定追加到 §12 决策日志。
10. **子 Agent 一律 Haiku + high effort**（Jade 的成本要求）。连续的听感调参留在总督手里。
11. **复用优先**：先找现成模块或项目，记录地址、版本、许可证、接口和适配。自写代码只限适配层、测量和守护逻辑。

## 常用入口

| 要做的事 | 看哪里 |
|---|---|
| 当前主次线、授权边界 | docs/ROADMAP.md §5；docs/FILE_INDEX.md |
| 主线失败试训 | experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/REPORT.md |
| MusicBERT 次线零微调实验 | experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/REPORT.md |
| 已确认17对和源文件绑定 | docs/FILE_INDEX.md 的“共用资产”；不改根路径替换来源 |
| F1 成品链（A：去空响） | experiments/F1_demo_chain_2026-11-02_v003/run.py `chain_dry` |
| 母带到 -13 LUFS | experiments/F1_demo_chain_2026-11-02_v001/run.py `master` |
| 整体升 / 降调 | f2/poc_phase1.py `shift_uniform` |
| 旧有参考修音与可复用函数 | f2/poc_phase4c.py（拖音版）/ poc_phase4b.py；不是当前默认模型 |
| 共振峰上移（S45） | experiments/F1_demo_chain_2026-11-02_v003/formant_shift.py |
| 声源 / 力度测量 | experiments/WOMENDE_FOCUS_2026-11-03_v001/measure_source.py |
| 原唱参考与对齐区间 | ROADMAP §9.2 |
