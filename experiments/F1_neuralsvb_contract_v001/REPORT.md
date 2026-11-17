# 第二、三步任务报告

完成范围：修正Praat音高帧的时间定位，接回原声学模型与声码器，复跑三条正常素材与SHIGE中/严重对照；生成一次后验均值推理修订并筛选。Agent不能直接听音，全部听感为pending_jade。

## 可验证的结论

- 先修时间坐标，zuixuan纯声码器音高偏差由17.23/106.76降到4.44/33.17音分，现有工程检查通过。未用修音高、平滑旋律或移调。
- 保留原官方PWG的50–11025Hz/log10特征路径：声学模型输入与输出沿用同一约定，直接接原声码器。此前范围匹配实验未改善，不能只依配置文件认定声码器必须改成0–8000Hz；实际训练特征仍有公开信息缺口。
- 共核验25条处理输出，9条通过、16条拦截。每条最多筛出声码器对照和一个完整美化候选；若官方采样已通过则优先保留它，不凭指标猜测均值版本更悦耳。
- 全部最终WAV为44100Hz/单声道/PCM16，各自样本数与原录音完全一致；完整FFmpeg解码一致；每条素材内共同响度和真峰值余量按原规则计算。
- 原素材和之前实验WAV哈希保持。诊断失败产物保留在full_normal/full_severity，delivery只放通过者及匹配响度原音。
- 中度/严重使用SHIGE，因为zuixuan没有medium；未用mild冒充medium。这些是保留原始跑调的音色处理实验，不是音准修复。

## 工程结果（项目阈值不变）

项目音高偏差门槛为中位≤10、P95≤35音分；动态跨度≤8dB、局部相对增益≥-12dB、相邻采样跳变比≤2.5。Praat原为独立信息项，边界/失败值照实列出，不隐去。不同提取器的有效帧集合有区别，不把两个汇总数的差当严格逐帧改善量。

| 素材 | 变体 | 状态 | 项目音高中位/P95 | Praat中位/P95 | 动态跨度/最小增益dB | 拦截原因 |
|---|---|---|---|---|---|---|
| 茉莉花 · 正常 | vocoder | PASS | 2.13/13.70 | 2.61/17.77 | 1.30/-1.54 | 无 |
| 茉莉花 · 正常 | a2a | PASS | 2.10/16.03 | 2.77/22.66 | 6.31/-8.99 | 无 |
| 茉莉花 · 正常 | a2p | PASS | 2.06/16.80 | 2.73/21.57 | 5.82/-6.62 | 无 |
| 茉莉花 · 正常 | a2a_mean | PASS | 2.10/15.54 | 2.74/22.50 | 6.26/-8.73 | 无 |
| 茉莉花 · 正常 | a2p_mean | PASS | 2.04/16.80 | 2.73/21.71 | 5.90/-6.52 | 无 |
| SHIGE · 正常 | vocoder | PASS | 3.21/18.61 | 4.31/22.70 | 2.82/-6.24 | 无 |
| SHIGE · 正常 | a2a | BLOCKED | 3.33/26.32 | 4.25/29.14 | 9.85/-20.98 | dynamic change / dropout in active singing |
| SHIGE · 正常 | a2p | BLOCKED | 3.34/24.98 | 4.22/29.76 | 11.31/-26.44 | dynamic change / dropout in active singing |
| SHIGE · 正常 | a2a_mean | BLOCKED | 3.30/26.33 | 4.21/29.31 | 9.77/-20.87 | dynamic change / dropout in active singing |
| SHIGE · 正常 | a2p_mean | BLOCKED | 3.33/24.67 | 4.20/30.26 | 11.30/-26.55 | dynamic change / dropout in active singing |
| zuixuan · 正常 | vocoder | PASS | 4.44/33.17 | 5.53/35.24 | 3.15/-11.46 | 无 |
| zuixuan · 正常 | a2a | BLOCKED | 4.86/66.61 | 5.55/42.11 | 12.84/-20.40 | dynamic change / dropout in active singing, pitch diagnostic changed (manual check required) |
| zuixuan · 正常 | a2p | BLOCKED | 4.77/98.81 | 5.35/42.29 | 16.24/-23.52 | dynamic change / dropout in active singing, pitch diagnostic changed (manual check required) |
| zuixuan · 正常 | a2a_mean | BLOCKED | 4.82/63.78 | 5.56/42.75 | 12.74/-21.79 | dynamic change / dropout in active singing, pitch diagnostic changed (manual check required) |
| zuixuan · 正常 | a2p_mean | BLOCKED | 4.76/98.98 | 5.32/42.02 | 16.23/-23.54 | dynamic change / dropout in active singing, pitch diagnostic changed (manual check required) |
| SHIGE · 中度跑调 | vocoder | PASS | 4.07/22.56 | 4.85/29.18 | 2.61/-6.17 | 无 |
| SHIGE · 中度跑调 | a2a | BLOCKED | 4.40/32.94 | 5.11/37.56 | 11.64/-21.22 | dynamic change / dropout in active singing |
| SHIGE · 中度跑调 | a2p | BLOCKED | 4.30/33.46 | 5.04/38.48 | 12.68/-25.11 | dynamic change / dropout in active singing |
| SHIGE · 中度跑调 | a2a_mean | BLOCKED | 4.36/32.80 | 5.08/37.79 | 11.58/-20.94 | dynamic change / dropout in active singing |
| SHIGE · 中度跑调 | a2p_mean | BLOCKED | 4.34/33.44 | 5.00/38.13 | 12.77/-25.42 | dynamic change / dropout in active singing |
| SHIGE · 严重跑调 | vocoder | PASS | 4.06/24.92 | 4.86/40.14 | 3.38/-7.66 | 无 |
| SHIGE · 严重跑调 | a2a | BLOCKED | 4.34/33.17 | 5.08/38.68 | 15.08/-48.22 | dynamic change / dropout in active singing |
| SHIGE · 严重跑调 | a2p | BLOCKED | 4.33/32.47 | 5.03/40.19 | 17.43/-50.95 | dynamic change / dropout in active singing |
| SHIGE · 严重跑调 | a2a_mean | BLOCKED | 4.25/33.45 | 5.05/39.04 | 14.91/-47.90 | dynamic change / dropout in active singing |
| SHIGE · 严重跑调 | a2p_mean | BLOCKED | 4.31/32.79 | 5.05/40.07 | 17.40/-50.84 | dynamic change / dropout in active singing |

## 自查与限制

时间适配器有三项合成信号/边界测试：保留无声间隔、随时间移动的音高落在真实帧中心、精确保留时长且拒绝补零掩盖缺失。先有失败后实现，随后增加移动音高回归用例。

后验均值修订仅把同一声学模型中的随机潜变量改为其均值，作为有限的确定性推理对照；没有训练/微调，不等同官方默认流程。所有未通过的变体仍按拦截处理，没有通过混原音、增益补洞或调宽门槛隐藏问题。

相邻采样跳变比通过不能证明听觉跳变消失；音高估计也可能受谐波/气声影响。44.1kHz交付由22.05kHz模型输出重采样，不恢复高频带宽。模型单WAV/自身F0/恒等对齐属于项目适配，官方原始输入推理仍WIP；中文内容特征泛化和说话人特征一致性仍需听感验证。

## Jade验收

打开index.html。先以固定播放音量比较原音和声码器对照，判断全程沙哑、局部跳变是否减轻；再听本组若有的完整美化候选，判断是否更自然悦耳、是否变嗲、是否仍像本人。可直接反馈更喜欢哪个，以及不满意的时间点。没有候选的组不要求勉强选择。

正常/中度/严重是不同演唱录音，不能把它们的绝对指标或主观质量差异全部归因于处理；每条只与自己原音比较。正常单素材暂通过之后再看其他录音能否复现收益。

正常SHIGE的额外局部衰减集中在约0.8–1.0s、16.8–17.0s，最低约-26.4dB；zuixuan包含约2.24s、7.8–8.2s、9.48–9.62s、19.34–19.52s等事件。这些事件多落在源F0无基频帧中心，可能涉及清音/呼吸/句子边缘，但无基频不等于没有歌词或重要声音。不能以此取消工程拦截，也不将这些时刻自动当成Jade全部主观跳变的精确边界。详表见两组dropout_localization.json。

参数、版本及中间结果见run_pipeline.py、feature_contract.py、provenance.json、各组report.json和execution.log。工作日志见TASK_LOG.md。既有production入口未改动；本轮为隔离实验，尚未把结果整合成正式产品实现。

## 最终文件与浏览器验收

2026-10-26：再次核验30条整链WAV和11条筛选交付WAV及源录音哈希；三项适配器测试通过。真实浏览器校验11条HTTP音频哈希、完整解码、44100Hz/单声道/样本数，全部静音8倍速播放至末尾，定位及切换互斥通过。详见final_artifact_verification.json与browser_verification.json；不包含主观听感结论。
