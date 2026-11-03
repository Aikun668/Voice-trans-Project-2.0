# 我们的歌 · 薄音版定点第 2 轮（2026-11-03 v002）

按 docs/ROADMAP.md §6 做。听感：**pending_jade**。工程：**technical_pass = true**（四版都通过）。Agent 不能听音，下面的数字不是「变好听」。

## 这一轮在听什么

上一轮原话在 `feedback.md`。方向是：升调本身更有力；整体再响一些；还有一点点空旷；参数可以再试高一点。补谐波不再当主杠杆。

一首歌，约 14 秒。四个版本来回听，素材合计约 56 秒。试听页 `index.html`。

**F1 不改音高。** 整体升调是单独的音区开关（F2-K），产品默认关闭。本轮为了试力度，②④ 打开 +310 音分，③ 打开 +400 音分。① 关闭。

## 四个版本

| 页面上的名字 | 文件 | 做了什么 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 只增益 + 前视真峰值限幅到约 -13 LUFS |
| ② 升一点 | `audio/k310_a.wav` | F2-K +310 音分 → 电平对齐原音 → A 链 `chain_dry` → 母带 |
| ③ 再升一点 | `audio/k400_a.wav` | F2-K +400 音分 → 同上 |
| ④ 升一点 + 更厚实 | `audio/k310_formant70_a.wav` | F2-K +310 音分 → 共振峰 +70 Hz → A 链 `chain_effort` → 母带 |

+70 Hz 取 §6 所写 +45~70 Hz 的上沿，也是 v003 `SHIFT_HZ`。S45 原实验的移位量是 +45 Hz（`verify_formant_shift.py`）。本轮没有再做一档 +45。

没做：混响、双叠、硬吸附、谐波激励、`harmonic_tilt`。

## 怎么打开试听页

在本目录用浏览器打开 `index.html`（音频用相对路径 `audio/`）。耳机，音量固定。一个播放器，点名字从当前进度接着放。底部「生成答案」导出文字。

## 怎么重跑

不要装进已有 `.venv-*`。输出目录 `audio/` 已存在时脚本拒绝覆盖。

```bash
python3 -m venv .venv-womende-v002
.venv-womende-v002/bin/pip install -r experiments/WOMENDE_FOCUS_2026-11-03_v002/requirements.txt
.venv-womende-v002/bin/python experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py
```

需要系统里的 `ffmpeg`（本轮读回用的是 ffmpeg 6.1.1 的 ebur128）。

## 复用

| 模块 | 地址 | 版本 | 许可证 | 接口 | 本轮适配 |
|---|---|---|---|---|---|
| Praat / Parselmouth | https://github.com/YannickJadoul/Parselmouth | 0.4.7 | GNU GPLv3 | `f2/poc_phase1.py` `shift_uniform(sound, cents)` | 升调后按原音 RMS 对齐再进 F1（压缩门限是绝对电平）；长度补齐到源文件 |
| Pedalboard | https://github.com/spotify/pedalboard | 0.9.26 | GPL-3.0 | v003 `chain_dry` / `chain_effort` 里的现成效果器 | 参数未改 |
| 共振峰 | `experiments/F1_demo_chain_2026-11-02_v003/formant_shift.py` | 仓库内 S45 复刻 | 本仓库 | `chain_effort` → `apply_shift(..., 70)` | 放在 F2-K 之后 |
| 母带 | `experiments/F1_demo_chain_2026-11-02_v001/run.py` | 仓库内 | 本仓库 | `master` / `ebur` / `tp_limit` | 直接调用。链内还有 Pedalboard Limiter -3 dB；最后一级是 `tp_limit` |
| NumPy / SciPy / soundfile | PyPI | 2.5.4 / 1.18.1 / 0.14.0 | BSD | 数组、包络、读写 | 新 venv，未改旧环境 |
| FFmpeg ebur128 | 系统 ffmpeg 6.1.1 | — | LGPL/GPL（视构建） | `ebur()` | 响度和真峰值以它读回为准 |

`experiments/F1_demo_chain_2026-11-02_v002/run.py` 不在这个快照里。`chain_dry` 导入的 `floor_cut`（句间底噪下压）因此没有原实现。`floor_cut_stub.py` 原样返回音频，**没有做句间降噪**。`qa_extra.py` 同样不在，齿音段指标没有另写一套来冒充原工具。

## 工程结果（technical_pass）

源文件只读。SHA-256 `6923e1e390604a804d07df4ff1e80e4c49324bf4beb86f361c3eab2ad271c4e8`（前 16 位 `6923e1e390604a80`，与 v003 记录一致）。处理前后哈希相同。48000 Hz，669696 样本，13.952 秒。四版读回后样本数、采样率一致，无 NaN，削波样本 0。

升调前 Praat（75–600 Hz，并用 800 Hz 天花板复核）：浊音中位 136.96 Hz，最高 224.23 Hz。+310 音分后最高约 268 Hz，+400 音分后最高约 283 Hz。没有帧会越过约 600 Hz 跟踪上限。

通过线：LUFS 在 -13 ± 0.4；真峰值 ≤ -1.0 dBTP；移调中位与目标差 < 5 音分；浊音帧重合 ≥ 0.95；A 链字尾抬升 ≤ +3 dB。F2-K 是整段 PSOLA，没有「修正段外逐样本一致」可查的区间。

| 版本 | LUFS | 真峰值 | 字尾抬升 | 句间底噪相对抬升 | 中位音分 | 浊音重合 | 通过 |
|---|---|---|---|---|---|---|---|
| ① 原音·同样响 | -13.1 | -1.3 dBTP | +1.43 dB | +1.40 dB | 0.00 | 1.000 | 是 |
| ② 升一点 | -13.0 | -1.3 | +2.46 dB | +0.63 dB | +310.55 | 0.997 | 是 |
| ③ 再升一点 | -13.0 | -1.3 | +2.30 dB | +0.68 dB | +400.38 | 0.997 | 是 |
| ④ 升一点 + 更厚实 | -13.1 | -1.3 | +2.27 dB | +1.55 dB | +309.93 | 0.989 | 是 |

字尾抬升是 v003 `decay_mask` 上的中位差，门槛 +3 dB 只套在走了 A 链的 ②③④。① 没有 A 链，+1.43 dB 来自母带限幅改变响段统计，不拿这条门槛判它。

句间底噪用 v001 `level_stats`：全段 P5 相对 P95，再减原音。正值表示轻的地方相对响段被抬高。这里是 +0.6 ~ +1.6 dB，不是以前快释放压缩的 +6 ~ +10 dB。因为 `floor_cut` 没做，这不是「已经降过噪」。

移调中位误差：② +0.55 音分，③ +0.38，④ −0.07；升调文件本身（进 F1 之前）② +0.39、③ +0.32。帧间并不都贴着中位：② 绝对误差的 95 分位约 29 音分，④ 约 44 音分。④ 更散，和已知情况一致——共振峰移位会干扰 Praat 的音高分位，中位仍在 5 音分以内。通过与否看中位和浊音重合，不把分位加宽写成听感。

④ 的共振峰：浊音帧 914/1313，源 F1 中位 219.5 Hz，单点最大增益 15.53 dB（模块已知会到 15–16 dB）。

响度计是写成 16-bit WAV 之后用 FFmpeg ebur128 读回的。②③④ 的成品链里还有一道约 -3 dB 的 Limiter，四版最后都有前视真峰值限幅。页面上写了。

## 听感

pending_jade。不要把上面的通过当成好不好听。
