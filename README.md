> **2026-11-03：现行路线、目标与交接见 [docs/ROADMAP.md](docs/ROADMAP.md)；规则入口见 [AGENTS.md](AGENTS.md)。下方内容为历史记录。**

# F1 当前主线：自适应分析与决策 V0

2026-10-27 重启基线：**原音WAV → 特征分析 → 有界规则 → Pedalboard DSP → 等响A/B/C**。不训练、不找预训练权重、不重建人声。

运行与规则统一看 **[新V0说明](f1_adaptive_v0/README.md)**；旧路线状态看 **[收尾记录](docs/F1_REBASE_2026-10-27.md)**。

当前自动处理只有温和压缩；自动EQ关闭，去齿音未实现。新代码 f1_adaptive_v0/，新环境 .venv-f1-adaptive/。两条实录工程通过，听感仍待Jade。尚无稳定美化或F2组合通过结论。

```powershell
.\.venv-f1-adaptive\Scripts\python.exe f1_adaptive_v0/run.py --input test_dataset/my_singing/shige_normal.wav --output f1_adaptive_v0/outputs/new-run
.\.venv-f1-adaptive\Scripts\python.exe -m pytest f1_adaptive_v0/tests -q
```

**以下全部为重启前历史记录**，其中“当前试听”等字样保留原文，不代表现在主线。完整旧README备份：docs/archive/README_before_adaptive_v0_2026-10-27.md。

---

# 历史：F1 音色美化 PoC · Jade

当前试听：`experiments/F1_S1_density_v001/index.html`（功能锚定后的第一步：单项密度实验）。
规则见 `AGENTS.md`，功能锚定图见 `docs/F1_FEATURE_MAP.md` / `docs/F1_FEATURE_MAP.html`。
只复用 Pedalboard Distortion 验证轻谐波处理，不额外叠加 EQ、压缩器或其他美化模块。
输入电平校准、处理分支增益补偿、原音混合与响度匹配是适配层，不是自行实现新的处理内核。
主素材：本人 shige_medium.wav。原音、温和、较明显三条；听感仍为 pending_jade。
反馈好则换素材复验；副作用或无收益最多一次有依据的修订，随后保留或更换候选，详见规则文件。

```powershell
python -m venv --system-site-packages .venv-f1
./.venv-f1/Scripts/python.exe -m pip install --only-binary=:all: --no-deps -r requirements-f1.txt
./.venv-f1/Scripts/python.exe f1_density.py --out experiments/F1_S1_density_v002
./.venv-f1/Scripts/python.exe -m pytest -q
```

旧的跑调素材对照：`experiments/F1_offpitch_v001/index.html`。
使用本人《是谁在唱歌》中度/严重版，以及新换的网络 GTSinger 女中音《十年》。
只换素材，v0.5 两档参数与上轮正常素材实验完全一致；未做输入电平适配，也不修正跑调。
全部原始文件保留，网络录音另转测试副本。各条内部匹配响度，技术检查与人工听感验收分开。

```powershell
python transfer_f1.py --dataset offpitch --out experiments/F1_offpitch_v002
```

上一轮试听：`experiments/F1_transfer_v002/index.html`（正常演唱素材）。
固定 v0.5 两档，使用本人 shige_normal、zuixuan_normal，以及网络 GTSinger 女中音《告白气球》。
网络原始 48 kHz / 24-bit WAV 保留，另转为 44.1 kHz / 16-bit 测试副本。
各首内部匹配响度，原音与处理版比较；网络女声不是本人音色目标。
女声仍沿用 60–500 Hz 粗略诊断，不能由此确认完整音高范围保留。
每条提供完整原音与两档处理、同一句连续对照；此轮不按新素材重新调参。
首次迁移试跑 `F1_transfer_v001` 中的 v0.3 旧基准触发动态变化拦截，记录保留，v002 不交付旧基准。

```powershell
python transfer_f1.py --out experiments/F1_transfer_v003
```

茉莉花第五版：`experiments/F1_molihua_v005/index.html`（清晰方向：明显处理、强处理）。
v0.4 人工反馈：除 v0.3 基准略闷、很认真听有一点区别外，其余基本与原音无区别，暂未选出通过版本。
v0.5 不继续堆低频或整体压暗，改为 650 Hz 削减与 2 kHz 提升，两档压缩均减轻到 2:1。
这仍是候选，需判断改善是否更易感知、有无变尖变薄；本轮后换另外两条本人录音验证。

```powershell
python f1_poc.py test_dataset/my_singing/molihua_normal.wav --profile v5 --out experiments/F1_molihua_v006
```

第三版试听：`experiments/F1_molihua_v003/index.html`（增强档与更强档）。
Jade 对第二版强档反馈：认真听可辨认，感觉略有改善；第三版沿此方向加强。
第三版同时提供上轮强档作基准、四段同句连续对照。更强不代表已经更好。

```powershell
python f1_poc.py test_dataset/my_singing/molihua_normal.wav --profile v3 --out experiments/F1_molihua_v004
```

第二版试听：`experiments/F1_molihua_v002/index.html`（轻档与强档）。
第一版目录保留；Jade 反馈整体听不出差别，未通过听感验收。

第二版增加低频厚度、中频与高频的调整，并加强动态处理；响度继续匹配。
`04_same_phrase_ABC.wav` 将同一句原音、轻档、强档各 3 秒顺序拼接，供快速对照。
第二版复现：

```powershell
python f1_poc.py test_dataset/my_singing/molihua_normal.wav --profile v2 --out experiments/F1_molihua_v003
```

第一轮素材：`test_dataset/my_singing/molihua_normal.wav`。
路线：FFmpeg DSP（EQ + 轻压缩 + 轻去齿音），不训练模型、不移调。

## 听第一版

打开 `experiments/F1_molihua_v001/index.html`，或直接播放该目录下的 WAV：

- `01_before_matched.wav`：原音只调音量，公平对照。
- `02_F1_light.wav`：轻处理。
- `03_F1_medium.wav`：稍强处理，便于判断处理痕迹。
- `00_original.wav`：原文件完整副本，响度更大，不用于直接 A/B 比较。

所有试听文件为 44.1 kHz / 单声道 / PCM 16-bit，约 13.01 秒。
本轮采用素材原规格，尚不代表与 F2 负责人共同确认了接口规格。

## 重跑

环境：Python 3.10+，NumPy、SciPy；FFmpeg 在 PATH 上。
本机本轮使用 Python 3.12.7 / NumPy 1.26.4 / SciPy 1.13.1 / FFmpeg 9.0.1。

```powershell
python f1_poc.py test_dataset/my_singing/molihua_normal.wav --out experiments/F1_molihua_v002
python -m pytest -q
```

每次必须用新的输出目录；脚本拒绝覆盖已有实验。原素材不修改。
初版输入只接受至少 3 秒的单声道 PCM16 WAV，采样率保持不变。
主流程只依赖 NumPy、SciPy 和 FFmpeg；测试额外需要 pytest。

## 检查与验收

导出后重新读取实际 WAV，检查完整解码、规格、样本数、响度、真峰值、削波、
波形相关性、异常静音/跳变、过大频带变化和粗略音高变化。
`qa_report.json` 保存参数、哈希、版本和实际测量值；异常时脚本返回非零。
输出中途失败时可能留有不完整目录，缺少通过的报告就不能当成成功结果。

响度匹配采用逐文件测量后施加固定增益，三个版本使用共同目标，默认 -18 LUFS。
若任何版本无法保留 -3 dBTP 余量，就一起降低共同目标；没有额外限幅或动态响度处理。

技术检查通过不代表音色改善。当前工具没有直接听音能力，主观验收由 Jade 完成。
请反馈：偏好原音/轻处理/稍强处理/无明显差别，并尽量给出发闷、变薄、抽吸等问题的时间点。

## 资料与范围

滤镜能力依据 [FFmpeg 官方文档](https://www.ffmpeg.org/ffmpeg-filters.html)。
两档参数是本实验的初始预设，不是官方美化方案，也不保证对所有歌手有效。
只验证 F1 的这一段素材，尚未进行 F2 串联、其他样本泛化或人工听感验收。
