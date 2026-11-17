# F1 路线一：Smule Renaissance（重建类）验证 — 记录

日期：2026-10-27。任务来源：导师验证阶段，路线一（重建类）可行性判断。

## 许可核查（已核实）

- 代码仓库：https://github.com/smulelabs/smule-renaissance ，LICENSE 为 **MIT**（Copyright (c) 2025 smulelabs），允许本机评测使用。
- 权重：HuggingFace `smulelabs/Smule-Renaissance-Small`（模型卡标注 **License: mit**），checkpoint `smule-renaissance-small.pt`（约 42 MB），公开下载、无需登录。
- 结论：代码与权重均为 MIT，允许本项目本机评测使用。**继续执行。**
- 论文：arXiv 2510.21659（Smule Renaissance Small，10.4M 参数，歌声修复 SVR）。

## 接口与版本

- 官方用法：`python main.py {input} -o {output} -c {checkpoint}`，Python 3.10。
- 预处理（main.py 核实）：>1 声道转单声道（均值）→ 重采样到 48 kHz → 60 Hz 高通；峰值归一化处理后还原增益；输出 48 kHz WAV，时长与输入一致。
- 设备：有 CUDA 用 FP16，否则 CPU FP32。本机为 CPU（torch 2.7.1+cpu）。
- STFT：n_fft=4096，hop=2048。

## 本机环境（隔离，未动系统 Python）

- 目录：`experiments/F1_route1_smule_v001/`，venv 由 uv 创建，托管版 CPython 3.10.18。
- 依赖：torch==2.7.1、**torchaudio==2.7.1（本机修正）**、numpy==2.2.6。
  - 官方 requirements.txt 未钉 torchaudio 版本，uv 默认装 2.11.0，与 torch 2.7.1 ABI 不兼容（WinError 127），已钉回 2.7.1 匹配版本。属依赖版本适配，不改官方算法代码。
- 运行方式：`experiments/F1_route1_smule_v001/.venv/Scripts/python.exe -I`（隔离模式）。
- 权重与下载文件：`experiments/F1_route1_smule_v001/downloads/`。
- 仓库代码：`experiments/F1_route1_smule_v001/smule-renaissance/`（git clone，未改动）。

## 已验证 / 未验证

已验证：
- 许可（代码 MIT、权重 MIT、可本机评测）。
- 权重可公开下载（42 MB）。
- venv 内 torch/torchaudio 导入正常。
- 样例 sanity check 通过（2026-10-27）：官方 ExtremeDegradationBench 的 `noisy/0001_singing.wav`（14.34 s，48 kHz 单声道）→ CPU FP32 推理成功；输出 48 kHz 单声道，样本数与输入完全一致（688128），无削波（peak 0.878）。输出在 `sanity/`。
  - 模型本身行为正常，"流程错"与"模型不适配"已区分开：后续 Jade 素材若有问题，归因于模型对干净清唱的适配，不是流程。

## 本机必要适配（均不改官方算法代码）

- `run_smule.py`：`-I` 隔离模式不把脚本目录加入 sys.path，官方 `from spectral_ops import ...` 会失败；运行器显式补官方仓库路径后以 runpy 执行 main.py。
- `soundfile==0.14.0`：Windows pip 版 torchaudio 无音频解码后端，`torchaudio.load/save` 直接失败；soundfile 是 torchaudio 官方标准后端。

## 步骤 4–5：SHIGE 单条 + 批跑结果（2026-10-27）

运行目录：`run_shige_v001/`（M2 单条）、`run_batch_v001/`（M1、M3、O1–O3）。全部输出与原音同时长、单声道、原音采样率，响度 -18 LUFS 共同目标；原音哈希、完整解码、规格、样本数检查全过。素材哈希未改动。

| 素材 | 结果 | 原因（测量值） |
|---|---|---|
| M1 molihua_normal | technical_pass | 全项通过 |
| M2 shige_normal | technical_blocked | 演唱段轻声/尾音被压 13–32 dB，共 15 段（最重 16.70–17.08s，结尾整段被吃） |
| M3 zuixuan_normal | technical_blocked | 同类掉字 18 段，最深 -35 dB（23.04–23.56s） |
| O1 官方中文 gt_a 3 | technical_pass | 全项通过 |
| O2 官方中文 gt_a 9 | technical_blocked | 无掉字，但动态被重塑：活跃段增益起伏 9.6 dB（阈值 8） |
| O3 官方英文 gt_a 6 | technical_pass | 全项通过 |

定位工具：`diagnose_gain.py`。拦截归因：Smule 是"受损人声修复"模型，把干净清唱里的轻声、气声、尾音衰减当作"损伤"压制（噪声门类行为），并会重塑动态。与 NeuralSVB 的失败（重建引入沙哑/鼻音）机制不同，但同属"重建类在干净清唱上吃掉原音内容"。音高中位偏差 1.3–1.7 音分、P95 ≤18 音分，全部在阈值内；跳变、削波无异常。

通过线核算：需 8 条中 ≥6 条 Jade 判"更好/同样好听"且无否决项。当前技术过线仅 3/6，即便 G1–G2 两条全过且听感全好，上限 5/8 < 6/8，**路线一已无法达到通过线**。剩余价值：听感页确认 3 条过线素材的听感与拦截归因，写进路线结论（哪类录音会失败、为什么）。

未验证（下一步）：
- 对 Jade 干净清唱的实际输出质量（注意：官方定位是"受损人声修复"，干净清唱属分布外；NeuralSVB 教训提示重建类可能引入沙哑/鼻音/字被吃）。
- 输出采样率 48 kHz 与原音（可能 44.1 kHz）的规格对齐方式（计划：重采样回原规格）。

## 素材约定（与路线二共用，不改原始素材）

- M1–M3：`test_dataset/my_singing/` 的 molihua_normal、shige_normal、zuixuan_normal。
- O1–O3：`experiments/F1_neuralsvb_pitch_target_v001/official_all/` 的 Chinese gt_a 3.wav、9.wav，English gt_a 6.wav。
- G1–G2：GTSinger，待路线二对话选定后补跑。
