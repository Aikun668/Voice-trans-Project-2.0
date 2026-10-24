# F1 音色美化 PoC · Jade

当前试听：`experiments/F1_molihua_v003/index.html`（第三版，增强档与更强档）。
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
