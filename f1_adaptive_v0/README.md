# F1 自适应分析与决策 V0

原音→特征分析→可解释规则→有界DSP→A/B/C等响对照。没有训练、权重、重建人声、修音高、改节奏或前端。

## 运行

项目根目录，Python3.12，需要PATH中的FFmpeg：

```powershell
python -m venv .venv-f1-adaptive
.\.venv-f1-adaptive\Scripts\python.exe -m pip install -r f1_adaptive_v0/requirements-lock.txt
.\.venv-f1-adaptive\Scripts\python.exe f1_adaptive_v0/run.py --input test_dataset/my_singing/shige_normal.wav --output f1_adaptive_v0/outputs/my-new-run
.\.venv-f1-adaptive\Scripts\python.exe -m pytest f1_adaptive_v0/tests -q
```

输出目录必须不存在。输入至少3秒、16–96kHz、单声道PCM16/24 WAV。保留输入采样率、样本数；输出PCM16；不自动混声道。这只是本地实验接口，不代表F2共同确认。

`--stage analyze`只到特征；`--stage decide`到决策；默认full。`--rules path.yaml`指定规则。`--disable compressor eq deesser`可关闭任意效果器（同时作用于固定B）。

## 文件职责

- analyzer.py：soundfile原采样率读入；librosa的RMS/STFT；20ms窗、10ms步长，center=False；活动片段、动态、频段和高频候选时段。
- decision_engine.py：纯features→decision接口，不读取文件名/素材类别；路径和哈希只由运行器记录。
- dsp_processor.py：Pedalboard Compressor/PeakFilter薄封装及硬参数校验，无合成器。
- run.py：等响导出、实际WAV读回检查、参数与依赖记录。复用根f1_poc.py工程工具，不沿用旧美化预设。
- config/rules.yaml：有版本的规则；每次另存快照。
- requirements.txt：直接依赖；requirements-lock.txt：本次Windows/Python3.12完整依赖快照。

## 自动处理范围与规则

**V0自动决策当前只有温和压缩。** 同时满足：有效片段≥2秒、活动RMS P95−P10≥18dB、响段连续≥80ms、无满幅样本、未关闭压缩。A特征v0.2改为最大短时RMS−40dB作为活动门限，且不低于−75dBFS；这是电平分组，不是人声识别。低于门槛的非零声音单列于low_level（unknown），不宣称是静音或噪声。突出响声可能抬高参考，峰值参考附近支持不足80ms时给sparse_loud_reference提示。分析专用边界补零用于纳入首尾不完整窗口，结果时间坐标重新映射到原音且不改变输出音频。详情见 `experiments/F1_module_a_lowlevel_2026-10-27_v001/低电平分析修正报告.md`。

前一轮数字静音修正及旧分析器保留在 `experiments/F1_module_a_silence_fix_2026-10-27_v001/`；本轮修改前分析器另存于lowlevel实验目录。低于活动门槛的声音不参与主动态统计，但在不确定片段中保留位置，不表示已识别或处理。接近/高于门槛的背景声仍会明显影响统计。B仍是早期规则，没有消费新增不确定提示；不能据此宣称能自动区分背景声和尾音，进入B阶段前须重核决策。旧试听文件和报告对应当时版本，不能用新代码假定能逐字节复现。

ratio随动态从1.3升到最高1.8；attack=25ms，release=180ms。threshold从RMS P90−3dB推导，另受最高4dB静态峰值减益估计约束。执行器硬限制ratio≤2、threshold∈[−60,0]、attack∈[15,60]、release∈[80,400]。静态估计不代替实际QA。

大小声可能是音乐表达；触发只表示实验建议，不能据此断言原音有缺陷。规则没有主观准确率声明。

**EQ执行器可用，但自动EQ关闭**，频谱不足以区分自然音色与缺陷，输出unknown。手工decision的PeakFilter边界：±2dB、Q 0.3–2、频率80至min(8000,0.45×采样率)Hz。

**De-esser未实现**，只记录5–8kHz突发候选，状态unknown。没有用固定削高频冒充去齿音；配置强行启用EQ/去齿音自动规则会报错。

固定B预先锁定：只有threshold=−18dBFS、ratio=1.5、attack=25ms、release=180ms压缩，用于对照固定参数，并非已通过的美化方案。A只调固定增益；C不触发时诚实旁路。

## 输出与验收

features.json、decision.json、fixed_decision.json、rules_snapshot.json、qa_report.json、LISTENING.md，以及完整对齐的：

- A original_matched.wav：等响原音。
- B fixed_baseline.wav：固定压缩。
- C adaptive.wav：自适应压缩或旁路。

共同目标最高−18LUFS，依据三版需要共同降低以留真峰值余量；只施加固定增益，没有隐藏限幅。导出后完整读回，检查规格、样本数、削波、响度、真峰值、异常静音/跳变、频带变化及粗略音高。沿用旧QA门槛、不为交付放宽。失败文件保留供诊断，状态technical_blocked且CLI非零。旧音高诊断只覆盖约60–500Hz；不足时警告，不能证明全音域保留或没有所有听觉伪影。

raw_dsp_delta_rms与raw_rms_reduction_db记录效果器实际变化，避免“显示触发但空转”。报告记录源哈希、版本、常量增益和输出哈希。无直接Agent听音能力；技术通过不代表好听，全部pending_jade。

## 本轮结果

`outputs/*-v001`为首跑诊断：峰值参照阈值太高，茉莉花空转、SHIGE变化极弱。先补真实录音失败测试，再修为RMS参照（规则v0.2），未覆盖旧输出。当前目录：

- outputs/2026-10-27-shige-v002/：优先试听。
- outputs/2026-10-27-molihua-v002/：第二素材工程检查，不算听感复验通过。

13项新测试及2项旧工程工具回归通过，两条实录三版通过现有工程QA。独立审查发现ratio=1配置边界除零，已补失败测试并修复；自适应规则要求ratio>1，旁路用enabled:false。先听SHIGE，固定播放器音量比较A/B/C：是否更舒服、轻字更清楚、表达还在；否决闷、抽吸、掉字/尾音、压平、变得不像本人。原录音此前没有确认明显局部缺陷，本轮是新路线技术实验，不改写旧反馈。

## 复用与未验证

Pedalboard0.9.26（GPL-3.0）：[项目](https://github.com/spotify/pedalboard)、[官方API](https://spotify.github.io/pedalboard/reference/pedalboard.html)。标准流程：创建效果器→Pedalboard→float32音频及原采样率。只复用Compressor/PeakFilter；规则、边界、QA为项目适配。

librosa0.11.0（ISC）：[官方RMS接口](https://librosa.org/doc/0.11.0/generated/librosa.feature.rms.html)，复用rms/stft/frame；NumPy1.26.4、SciPy1.13.1（BSD）用于统计与既有工程工具；soundfile0.13.1用于PCM读取。FFmpeg仅作计量、固定增益和导出，版本逐次记录。

尚未验证：听感收益、广泛歌手/设备适用性、自动EQ/去齿音、F2串联、硬件及实时性能。
