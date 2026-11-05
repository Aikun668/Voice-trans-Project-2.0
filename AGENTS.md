# AGENTS · 入口与硬规则（2026-11-03 重写）

**先读 [docs/ROADMAP.md](docs/ROADMAP.md)**。它是本项目的最高书面依据：目标、已确认偏好、已证实结论、路线阶段、接手后的第一个任务、工作方式、工程坑。
优先级：Jade 当前原话 > docs/ROADMAP.md > 本文件 > 需求原件 > 其他历史文档。
旧版 AGENTS.md（"每轮只验证一项""最多一次修订"等流程规则）已备份到 docs/archive/AGENTS_before_roadmap_2026-11-03.md，**不再执行**。

## 当前状态（一句话）

2026-11-05，P0 固定 DSP 调试暂停。新阶段先处理《心墙》《离开我的依赖》《莫问归期》三首重录的有参考局部修音，排除严重跑调和凤凰传奇旧素材；前三项分工见[初步计划](docs/新阶段_前三项任务初步计划.md)，进度见[开发日志](docs/开发日志_有参考修音.md)，路线以 [ROADMAP](docs/ROADMAP.md) 为准。老师 demo 等 Jade 点名「可以给老师」后再做。

## 硬规则（任何时候都适用）

1. **原始素材只读**。test_dataset/ 和旧实验输出都不覆盖、不删除、不搬移。每次实验新建 `experiments/<主题>_<日期>_vNNN/`，脚本遇到已存在的输出目录要拒绝运行。作废结果移到 `_superseded_<日期>/`。
2. **环境隔离**。不往现有 .venv-* 里装包；需要新依赖就新建环境，或放进实验目录的 vendor/。环境表见 ROADMAP §8.1。
3. **不推送 Git、不提交**，除非 Jade 明确要求。
4. **Agent 不能听音**，必须明说。不能把测量变化、测试通过说成"变好听"。工程结果记 technical_pass，听感记 pending_jade，直到 Jade 给出判定。
5. **否决项永久有效**，除非 Jade 主动改口：混响、双叠合唱、硬吸附修音（电音）、谐波激励（录音机感）。
6. **F1 不改音高**。整体升调只能通过独立的 F2-K 音区开关实现，默认关闭，并且要写明。
7. **每个输出文件做工程检查**（ROADMAP §8.2）：样本数、削波、实测 LUFS 与真峰值、音高偏差与浊音重合、源文件哈希。不为交付放宽门槛。
8. **试听负担**：一轮一首歌、3–4 个处理版 + "原音·同样响"锚点，总时长 ≤ 10 分钟；版本标名字；题目用日常话、给选项、可导出答案（格式见 ROADMAP §7.2）。
9. **Jade 原话原样存档**到该轮目录，并追加到 ROADMAP §11 反馈总表；方向性决定追加到 §12 决策日志。
10. **子 Agent 一律 Haiku + high effort**（Jade 的成本要求）。连续的听感调参留在总督手里。
11. **复用优先**：先找现成模块或项目，记录地址、版本、许可证、接口和适配。自写代码只限适配层、测量和守护逻辑。

## 常用入口

| 要做的事 | 看哪里 |
|---|---|
| F1 成品链（A：去空响） | experiments/F1_demo_chain_2026-11-02_v003/run.py `chain_dry` |
| 母带到 -13 LUFS | experiments/F1_demo_chain_2026-11-02_v001/run.py `master` |
| 整体升 / 降调 | f2/poc_phase1.py `shift_uniform` |
| 有参考修音 | f2/poc_phase4c.py（拖音版）/ poc_phase4b.py |
| 共振峰上移（S45） | experiments/F1_demo_chain_2026-11-02_v003/formant_shift.py |
| 声源 / 力度测量 | experiments/WOMENDE_FOCUS_2026-11-03_v001/measure_source.py |
| 本轮试听（v005，听感未答） | experiments/WOMENDE_FOCUS_2026-11-04_v005/ |
| 上一轮 presence（v004，已答，不判到顶） | experiments/WOMENDE_FOCUS_2026-11-04_v004/ |
| 原唱参考与对齐区间 | ROADMAP §9.2 |
