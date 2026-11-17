# 我们的歌 · 薄音版第 4 轮（2026-11-04 v004）

v003 的听感：②③④力度分不出来；③还是挺空，和②差不多；④不嗲，问题仍是空旷。
本轮不再把 A 链当空旷的药。在 v001 候选2（升约 310 音分 + 补谐波）后面，只做 2500 Hz 的 presence 峰值。
听感：**pending_jade**。工程：**technical_pass = true**。Agent 不能听音。

没有做老师 demo。P0 没有停。

## 四个版本

| 页面上的名字 | 文件 | 处理 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 只增益 + 限幅到约 -13 LUFS |
| ② 上次有力那版 | `audio/op_replay.wav` | v001 的 +310 音分文件 → 同一补谐波 → 母带。没有 presence EQ |
| ③ 有力 + 补中高频 | `audio/presence.wav` | 与②相同 → 2500 Hz +6 dB，Q=1.4 → 母带 |
| ④ 有力 + 更强补中高 | `audio/presence_strong.wav` | 与②相同 → 2500 Hz +12 dB，Q=1.4 → 母带 |

升调是音区开关 F2-K，产品默认关。① 关，②③④ 开 +310 音分。补谐波和 presence EQ 不改音高。
没有 chain_dry、decay_guard、混响、双叠、硬吸附、谐波激励。

试听页：`index.html`。

## ③④相对②改了什么

③ 在②的声音上多了一个峰值均衡：2500 Hz、+6 dB、Q=1.4。
没有第二段削减，也没有 250 Hz 以下的滤波器。Q=1.4 的钟形自己带出过渡：
实测大约 2000 Hz +4.22 dB、2500 Hz +6.00 dB、3000 Hz +4.66 dB，
1500 Hz +1.86 dB、4000 Hz +2.04 dB。
250 Hz 以下最高 +0.033 dB，最低 +0.000 dB。

④ 频率和 Q 不变，增益改成 +12 dB。
实测大约 2000 Hz +8.23 dB、2500 Hz +12.00 dB、3000 Hz +9.08 dB，
1500 Hz +3.93 dB、4000 Hz +4.26 dB。
250 Hz 以下最高 +0.082 dB，最低 0.000 dB。

上面这些分贝是均衡器自己的曲线，发生在母带之前。
母带再把四版拉到同一响度，所以成片里各频段的绝对电平还会一起挪，不会正好等于曲线上的数字。
2–3 kHz 相对 250 Hz 以下的比例，见下面的测量。这只说明频谱变了，不说明听起来更好。

§3.1 写手机录音的 2–3 kHz 比录音室人声少约 18 dB。
+6 dB 大约是这个缺口的三分之一，+12 dB 大约是三分之二，用来听两档，不是把频谱配平到录音室。

## 和 v001 候选2 的参数对照

补谐波没有重新求解。斜率、上限、窗长都从 v001 的脚本和 `report.json` 读出来用。presence EQ 加在补谐波之后。

| 项 | v001 候选2 | 本轮 ②③④ |
|---|---|---|
| 模块 | `harmonic_tilt.py` `process` | 同一个函数 |
| 升调 | `shift_uniform` +310 音分，`work/thin_keyup.wav` | 直接读这个文件。相关系数 1.0 |
| 斜率 | 7.992 dB/倍频程 | 同一值，来自 v001 `report.json` 的 `c2` |
| 5 kHz 增益上限 | 15.0 dB | 15.0 dB |
| 谐波窗 | σ = 0.12·F0，700 Hz 起，5 kHz 封顶，10 kHz 收回 | 模块常量未改 |
| alpha ratio | -20.11 dB | 补谐波输出、加 EQ 之前 -20.11 dB，差 0.0 dB |

和 v003 的②（`op_replay.wav`）相关系数 1.000000。
母带增益如果因响度计略有差别，相关系数仍然只看形状。

## 工程结果

源 SHA-256 `6923e1e390604a804d07df4ff1e80e4c49324bf4beb86f361c3eab2ad271c4e8`，处理前后相同。48000 Hz，669696 样本。

| 版本 | LUFS | 真峰值 dBTP | 中位音分 | 浊音重合 | 相对②的 2–3 kHz 减 50–250 Hz | 通过 |
|---|---|---|---|---|---|---|
| 原音·同样响 | -13.1 | -1.3 | +0.00 | 1.000 | -8.26 dB | 是 |
| 上次有力那版 | -13.1 | -1.3 | +310.53 | 0.997 | +0.00 dB | 是 |
| 有力 + 补中高频 | -13.1 | -1.3 | +310.55 | 0.996 | +4.58 dB | 是 |
| 有力 + 更强补中高 | -13.1 | -1.2 | +310.67 | 0.995 | +8.83 dB | 是 |

presence 之前到之后，F0 中位变化：③ 0.04 音分，④ 0.09 音分。门槛是 ±5 音分。

成片相对②：③ 的 2–3 kHz +4.21 dB，50–250 Hz -0.36 dB；④ 的 2–3 kHz +7.87 dB，50–250 Hz -0.96 dB。
低段这点下降来自整段被拉回同一响度，不是 250 Hz 以下的削减。均衡器曲线在 250 Hz 以下没有下切。
安静段相对响段（v001 `level_stats` 的 floor 抬升，相对原音）：原音·同样响 +1.40 dB、上次有力那版 +2.32 dB、有力 + 补中高频 +3.98 dB、有力 + 更强补中高 +6.48 dB。④ 更高，和它的峰值更吃限幅一致。这不是 A 链的字尾指标，本轮没有用那条链。

句间降噪没做。本轮也不使用 A 链的字尾守护。

母带先调用 v001 的 `master`（前视真峰值限幅，内部天花板 -1.3 dBTP）。
④ 的峰值更高，这个内部天花板会把响度留在大约 -13.2 LUFS，进不了它自己的 ±0.15 窗口。
④ 改用同一支 `tp_limit`，内部天花板放到 -1.2 dBTP。成片的真峰值仍用 ffmpeg 读回，门槛仍是 ≤ -1 dBTP。①②③ 没有改这个天花板。

## 复用

| 部件 | 地址 | 版本 / 许可证 | 本轮用的接口 |
|---|---|---|---|
| Pedalboard PeakFilter | https://github.com/spotify/pedalboard | 0.9.26 / GPL-3.0 | `PeakFilter` + `Pedalboard` |
| Parselmouth | https://github.com/YannickJadoul/Parselmouth | 0.4.7 / GPL-3.0 | Praat `to_pitch`；升调仍走 `f2/poc_phase1.shift_uniform` |
| 补谐波 | `experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py` | 仓库内 | `process` / `alpha_ratio`，斜率锁 v001 c2 |
| 母带 | `experiments/F1_demo_chain_2026-11-02_v001/run.py` | 仓库内 | `master` / `tp_limit` / `ebur` |
| 测量辅助 | `experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py` | 仓库内 | 音高预检、`pitch_delta`、哈希、长度对齐 |

自写部分只是把这些接口串起来，加上 250 Hz 以下不得削减的曲线检查，以及 F0 中位是否被 EQ 带走的检查。

## 重跑

`audio/` 已存在时脚本拒绝覆盖。依赖见 `requirements.txt`。不要装进已有的 `.venv-*`。

```bash
python3 -m venv .venv-womende-v004
.venv-womende-v004/bin/pip install -r experiments/WOMENDE_FOCUS_2026-11-04_v004/requirements.txt
.venv-womende-v004/bin/python experiments/WOMENDE_FOCUS_2026-11-04_v004/run.py
```

## 听感

pending_jade。
