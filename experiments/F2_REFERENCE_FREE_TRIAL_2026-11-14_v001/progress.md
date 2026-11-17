# 本轮执行记录

授权：feedback.md。规则及计划：rules.md。不要覆盖上轮清洗或任何已有阶段目录。

- A 完成：12 个风险测试已通过（每批新增功能均先看到失败，再实现通过）。
- B 完成：独立英文 ASR 内容特征与 MFCC 时间映射交叉核对；双音高估计。5,643 对全部核对，4,648 对候选通过、995 对隔离。
- 标签门槛通过：data_v001/gate.json。后续标签过滤保留训练 3,805 对 / 220 歌组、验证 442 对 / 28 歌组、测试 359 对 / 20 歌组。合并了 21 条由长歌词集合重合识别的歌曲别名关联。不代表人工逐音真值。
- C 进行中：training_v001 为 GRU 首次试训，日志 training_run.log，前三轮验证收益弱、输出接近零。并行测试 context_training_v001（HistGradientBoostingRegressor，56 个本人音高上下文特征），日志 context_training_run.log。只用验证歌曲选模型；测试集最后评估；未让中文录音参与训练或调参。
- D 待执行：infer.py 新进程只输入主项目 test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-轻度跑调.wav，源哈希 9fb9c2d365355715cb1a425e98be8695bc67628108efeb28af8098ad5eb15faa。审计钩子阻止读取下载数据、参考人声和本轮标签文件。
- E 待执行：deliver.py 已准备，原唱从已确认对照表匹配并在预测后读取；audio_qa.py 实测工程门槛；serve.py 仅绑定 127.0.0.1；尚未启动试听服务或浏览器。

原唱对应表：C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0/experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001/confirmed_pairs.json。所选源录音对应 id 1、2；reference.path 指向 f2-initial-plan 工作树中的原唱文件，已核对与表中 e96ab53... 哈希相同。只用于交付参照，不输入模型。

决策：不为“颠覆旧范式”强行取消必要的音频生成步骤。关键改变是模型在处理时不读原唱或音高差，真正从本人输入预测；渲染复用 Praat PSOLA，并以互补权重拼接保护原音。零命令精确旁路和已知 100 音分测试通过，实际本人音频仍待工程检查及 Jade 听感。

## 后续发现与改向

- GRU 7 个 epoch 验证收益仍小、输出接近零，已停止本轮 GRU 进程，原因保存在 training_v001/STOPPED.json，原日志保留；没有把它当成音频交付模型。
- 直接真人逐帧标签的上下文树模型 context_training_v001 也接近零：验证 MAE 57.8467，零修改 57.8142，未过能力检查。已保存真实权重；infer.py 在新进程只输入《心墙》完整录音后，因修正量太小而主动报错。inference_context_v001 不作为可交付结果。
- 原因尚不能单一归结为数据或模型，但两次不同唱法逐帧差异不等于应修的跑调。采取已在 rules.md 允许的备选：专业演唱音符块上注入已知特征级偏音，保留滑音/颤音，学习已知逆变换；真人标签只以较低权重补充，真人验证与合成验证分开。
- 新入口 train_practice.py，输出 practice_training_v001，日志 practice_training_run.log，工具 session 2783。新数据模型仍无原唱输入；没有在音频上合成造偏假冒真人。代码与13项风险测试均已通过；尚未重新输入/交付。
- 推理 load_model/predict_array 现支持真实已训 HistGradientBoostingRegressor；其输入由 context_features(f0) 的56个本人特征组成。老GRU仍保留独立 schema。最终若有输出，须再运行 infer.py → deliver.py → audio_qa.py → 浏览器验证。
- 已查现成更强旋律先验路线：Microsoft MusicBERT 官方 README（2026-08-05 更新的下载入口）与 malcolmsailor/musicbert_hf（维护者提供的 HuggingFace 适配）。尚未下载、安装或声称跑通；如果当前备选仍不成立，这属于下一条有根据的技术思路，不能继续把逐帧差强行称为有效标签。

待办：检查 practice_training_v001/report.json。若产生有意义预测且工程合格，交付明确标为试验的音频与原唱、原音对照，真人收益即使不成立也必须如实报告；如果新进程仍近乎不改或工程不通过，不伪造成功音频，记录失败及下一路线。更新本文件与 rules.md、ROADMAP。当前没有浏览器或试听服务在运行。

## 本轮交付状态

- practice_training_v001 完成，选中40棵树真实权重。合成偏音部分修回，但受控正常帧误改明显；真人候选验证也变差。effectiveness_pass=false，未当成成功模型。
- infer.py 新进程重新输入整条《心墙》，仅本人源文件和权重参与推理。431帧中118帧命令≥20音分；独立重载和应用曲线重建完全相同。
- delivery_v001 交付原音、原唱和模型诊断版，均8.612秒附近、−18.5 LUFS；本人两版413,376样本不变。模型执行中位误差1.24音分、有声重合99.85%，工程通过。
- Playwright三版均启动播放（静音），切换、结尾回到改动前、反馈保留和导出都通过，13项风险测试通过。浏览器功能测试答案已清空，不是用户反馈。
- verification.json 独立核验通过。REPORT.md 与 ROADMAP 已更新；rules.md 计划步骤完成，效果失败和听感pending_jade明确保留。
- 本地服务 URL=http://127.0.0.1:62512/index.html，仅绑定127.0.0.1；server_ready.json记录服务进程52092。保留服务供用户试听。所有训练进程均结束或已停止，没有后台继续训练。
- 后续已有根据的路线：音符级可信监督＋预训练音乐上下文（MusicBERT官方small入口与HuggingFace适配已查，未安装/下载/跑通）。本轮按用户约定先交付诊断试听，等待真实反馈，不反复调本人片段制造成功。
