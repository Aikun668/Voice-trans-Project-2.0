# F2 Phase-1 执行记录（2026-11-01）

最小闭环验证：Praat/Parselmouth Manipulation(TD-PSOLA) 读入 WAV → 音高修改 → 重合成 → 导出。
不含自动修音决策。听感验收由人完成。

## 环境（全部隔离在 .venv-f2，未动 F1）

- Python 3.12.7（.venv-f2 新建，无 system-site-packages）
- praat-parselmouth 0.4.7、numpy 2.5.4
- 安装：`python -m venv .venv-f2; .\.venv-f2\Scripts\python.exe -m pip install praat-parselmouth numpy`

## 复现命令

```powershell
# 实际执行（正式版）
.\.venv-f2\Scripts\python.exe f2\poc_phase1.py --input test_dataset/my_singing/molihua_normal.wav --output f2\outputs\phase1_molihua_v003 --cents 50
# 复现时换新目录（脚本拒绝覆盖已有目录）
```

输入：molihua_normal.wav（本人清唱，44.1kHz/单声道/PCM16，13.012s）。

## 输出（正式版 v003）

`f2/outputs/phase1_molihua_v003/`，5 个 WAV + qa_report.json，另有试听页 `listening/index.html`
（格式对齐 F1-EXP-008：登记框 + 逐版本播放器，音频副本在 listening/audio/ 下）：

| 文件 | 内容 |
|---|---|
| 00_original_untouched.wav | 原音位精确副本 |
| 01_original_matched.wav | 原音响度匹配版（对照基准） |
| 02_resynth_passthrough.wav | 分析→重合成，零修改（检验 PSOLA 链路） |
| 03_shift_plus50c.wav | 整体上移 50 音分 |
| 04_shift_minus50c.wav | 整体下移 50 音分 |

v001/v002 为调试过程记录（见"发现的问题"），按项目规则保留不删。

## 验证结果（v003，8/8 门槛通过）

- 时长：三版本与原音样本数逐点一致（0.0ms 偏差）
- F0 实测：passthrough -0.17 音分（≈无漂移）；+50.03 / -50.03 音分（目标 ±50，误差 0.03 音分）
- 频谱带能量差（vs 匹配原音）：passthrough 各带 ≤0.28dB；移调版 ≤1.12dB（高频带，谐波跨带移动所致）
- 无削波；匹配版本 RMS 一致（-13.69dBFS）；峰值余量 -3dB（统一 -2.12dB 修正，输入峰值原为 -1.2dBFS）

## 发现的问题

1. **Praat 命令名不符**：本版 Praat 无 `Get resynthesis`，正确命令为 `Get resynthesis (overlap-add)`（无参数）。
   最初报 "Command not available"，靠官方手册确认。
2. **波形相关系数不可用作 PSOLA 检查**：TD-PSOLA 重置相位，passthrough 与原音零滞后相关仅 0.03，
   属正常而非损坏。改为频谱带能量对比作为客观音色指标（v002 曾误用未匹配响度的原音做基准导致误报，
   v003 修正为响度匹配版基准）。
3. 听感层面是否有音色损伤（机器人感、粗糙、气声丢失），当前工具无法判断，待人耳验收。

## 听感反馈（molihua，用户原话）

> "2确实和原版没有什么大差别，三四的音色确实还像我自己"
> "当前这首茉莉花是唱得比较简单的，如果要验证，可能要选别的一首我自己唱的歌，比如说平凡之路"

结论：molihua 链路验证通过（②透明、③④音色保留）。按惯例换第二条本人录音复验。

## 复验（平凡之路正常版，2026-11-01）

- 输入：test_dataset/f1_dataset_v2/A_thin/self/wav/平凡之路正常版.wav（48kHz/单声道/PCM16，13.352s）
- 输出：`f2/outputs/phase1_pingfan_v001/`（参数与茉莉花完全一致：±50 音分，未调参）
- 试听页：`listening/index.html`（同 F1-EXP-008 格式）
- 工程：8/8 门槛通过；F0 实测 +49.8/-50.81 音分；②带能量差 ≤0.15dB；时长逐样本一致
- 听感反馈（用户原话）："OK, 仍旧符合" → 复验通过

**Phase-1 结论：PSOLA 音高修改链路在 molihua_normal + 平凡之路正常版双素材上通过（链路透明、移调不损音色）。**

## 下一步候选（待定，未实施）

- 平凡之路复验通过 → Phase-2：无参考保守修音决策层（pYIN F0 + 音符分段 + 有界向最近半音收缩），复用同一条重合成链
- 复验发现副作用 → 定位问题（音区起伏处的 PSOLA 伪影 / 音高跟踪误差），考虑 World vocoder 或 Praat LPC 重合成对比
