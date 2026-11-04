# 《我们的歌》P0 交接

日期：2026-11-04。读者：接手 P0 的第三方 Agent。

本文件是索引和约束清单。它不另写一条路线，不改 `docs/ROADMAP.md` 的目标，也不指定下一次算法。下一步用哪一个杠杆，由接手方根据 ROADMAP、各轮测量和下面的 Jade 原话自行诊断。

优先级（ROADMAP §0）：Jade 当前原话 > `docs/ROADMAP.md` > `AGENTS.md` > 需求原件 > 各实验目录 README / 历史文档。本文件低于 ROADMAP。

Agent 不能听音。`technical_pass` 只表示工程检查通过。听感只以 Jade 的原话为准，直到她给出下一次判定。

## 先读

1. `docs/ROADMAP.md`：§1、§2、§3、§7、§8、§11、§12。§6 是当时写给 v002 的任务书，v002 已经做完；见下文「怎么读已有章节」。
2. `AGENTS.md` 硬规则。
3. 本文件。
4. `experiments/WOMENDE_FOCUS_2026-11-04_v005/feedback.md`（第5轮原话）。
5. 第4轮原话在 `experiments/WOMENDE_FOCUS_2026-11-04_v004/feedback.md`。第3轮原话在 `experiments/WOMENDE_FOCUS_2026-11-03_v003/feedback.md`。第1轮、第2轮原话分别在 v001、v002 的 `feedback.md`。

## 素材与路径

| 项 | 路径 |
|---|---|
| 薄音版（只读） | `test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌-薄音版.wav` |
| 本人正常版（只读） | `test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌正常版.wav` |
| 原唱人声（Demucs，只作方向参考） | `test_dataset/reference/vocals/我们的歌_王力宏_vocals.wav` |

v002 / v004 的 README 记录：源 SHA-256 `6923e1e390604a804d07df4ff1e80e4c49324bf4beb86f361c3eab2ad271c4e8`，48000 Hz，669696 样本。对齐区间见 ROADMAP §9.2：薄音版 → 王力宏人声 15.5–29.1 s（调差 −350 音分）；正常版 → 107.8–121.7 s（−40 音分）。

新实验只建在本仓库 `experiments/<主题>_<YYYY-MM-DD>_vNNN/`（ROADMAP §8.3）。不要写到桌面（Desktop）或仓库外的路径。`test_dataset/` 与已有实验输出只读：不覆盖、不删除、不搬移。脚本遇到已存在的输出目录要拒绝运行。作废结果移到该实验下的 `_superseded_<日期>/`。

`experiments/WOMENDE_FOCUS_2026-11-04_v005/` 已经有试听页和 `audio/`。不要覆盖这个目录，也不要删掉其中的音频。新实验另开目录，写在仓库 `experiments/` 下，不要写到桌面。

## 硬约束

以下都是 ROADMAP / AGENTS 里已经写明的规则。本交接不新增处理偏好。

- **250 Hz 以下不做 EQ 削减。** ROADMAP §3.1：这是 Jade 的基频区；demo v002 削过之后，Praat 浊音帧重合掉到 0.895。
- **永久否决，除非 Jade 主动改口**（AGENTS.md；ROADMAP §2.1、§3.3）：混响；双叠合唱；硬吸附修音及其电音伪影；谐波激励（录音机感）。
- **F1 不改音高。** 整体升调只走独立的 F2-K 音区开关，默认关闭，页面和 README 要写明本轮开或关、开了多少音分（ROADMAP §4；AGENTS.md）。
- **按 §7 做一轮。** 先读最新原话并原样存档；先诊断再动手，试过的方向先查 §3.3；一轮一首歌，最多 3–4 个处理版加「① 原音·同样响」；工程检查（§8.2）全部通过才放进试听页；试听页格式见 §7.2；响度见 §7.3（成品轮 −13 LUFS / ≤ −1 dBTP，含锚点）；然后更新 §11 / §12。总听音时长 ≤ 10 分钟。版本标名字，不做盲听。
- **Agent 不能听音。** 测量变化、`technical_pass` 不能写成「变好听」。听感写 `pending_jade`，直到 Jade 判定。
- **环境。** 不往已有 `.venv-*` 里装包。新依赖用新环境，或放进该实验的 `vendor/`（§8.1）。
- **子 Agent。** Haiku + high effort。连续听感调参留在总督手里（§7.6）。
- **Git。** 未经要求不提交、不推送（§8.3）。
- **老师 demo。** `AGENTS.md` 当前状态写：仍等 Jade 点名「可以给老师」后再做。本交接不改这句话，也不启动 demo。

§8.2 的检查项：样本数与源一致、采样率不变、无削波、无 NaN；响度与真峰值用 FFmpeg ebur128 读回；音高中位偏差与浊音帧重合；源文件哈希在处理前后一致。不为交付放宽。

## 各轮目录里有什么

听感状态以 `feedback.md` 和 ROADMAP §11 为准。有的 README 文末仍写 `pending_jade`，那是写 README 时的句子，见下一节。

### v001 · `experiments/WOMENDE_FOCUS_2026-11-03_v001/`

测量、对齐、等响试听（README 写全部 −18.5 LUFS）。没有 `run.py`。

| 文件 | 内容 |
|---|---|
| `align_ref.py`、`align_ref.json` | 薄音版、正常版对王力宏人声的子序列 DTW |
| `measure_source.py`、`source_measures.json` | 响段浊音帧的声源指标 |
| `prep_pitch.py`、`work/thin_keyup.wav` | `shift_uniform` +310 音分 |
| `harmonic_tilt.py` | 元音谐波倾斜 |
| `pitch_qa.py`、`report.json` | 音高检查与 C1/C2 记录。`report.json` 的 `c2.slope_db_per_oct` 为 7.992 |
| `feedback.md` | 【我们的歌 定点反馈 2026-11-03】 |
| `index.html` | 试听页 |
| `audio/thin.wav` | 薄音 |
| `audio/normal.wav` | 本人正常版 |
| `audio/wang.wav` | 王力宏对应段 |
| `audio/c1_tilt.wav` | 只补谐波 |
| `audio/c2_keyup_tilt.wav` | 升调 + 补谐波（后文称 v001 候选2） |
| `audio/keyup_only.wav` | 只升调 |

### v002 · `experiments/WOMENDE_FOCUS_2026-11-03_v002/`

ROADMAP §6 的四版。成品响度。`run.py` 在 `audio/` 已存在时拒绝覆盖。`floor_cut_stub.py` 原样返回音频，句间降噪没有做。README 写：没有 `harmonic_tilt`，没有混响、双叠、硬吸附、谐波激励。工程记录为 `technical_pass`。

| 页面名字 | 文件 | README 所写处理 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 增益 + 限幅 |
| ② 升一点 | `audio/k310_a.wav` | F2-K +310 音分 → `chain_dry` → 母带 |
| ③ 再升一点 | `audio/k400_a.wav` | F2-K +400 音分 → 同上 |
| ④ 升一点 + 更厚实 | `audio/k310_formant70_a.wav` | +310 音分 → 共振峰 +70 Hz → `chain_effort` → 母带 |

`feedback.md` 是第2轮原话。`report.json`、`requirements.txt`、`index.html`、`README.md` 在同目录。

### v003 · `experiments/WOMENDE_FOCUS_2026-11-03_v003/`

README：接回 v001 候选2，再在 ③④ 叠 A 链 `chain_dry`。② 没有 A 链。工程记录为 `technical_pass`。`feedback.md` 是第3轮原话。

| 页面名字 | 文件 | README 所写处理 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 增益 + 限幅，约 −13 LUFS |
| ② 上次有力那版 | `audio/op_replay.wav` | v001 的 +310 音分文件 → 同一补谐波 → 母带 |
| ③ 上次有力那版 + 收字尾 | `audio/op_a.wav` | 与②相同的升调和补谐波 → `chain_dry` → 母带 |
| ④ 再高一点 | `audio/k400_tilt_a.wav` | +400 音分 → 同一补谐波斜率 → `chain_dry` → 母带 |

同目录还有 `run.py`、`report.json`、`requirements.txt`、`index.html`、`README.md`。

### v004 · `experiments/WOMENDE_FOCUS_2026-11-04_v004/`

README：在 v001 候选2 后面只加 2500 Hz 峰值；③ +6 dB，④ +12 dB，Q=1.4；不削 250 Hz 以下；不用 `chain_dry`。工程记录为 `technical_pass = true`。`feedback.md` 已有第4轮导出原文。该目录 `README.md` 的听感行仍写 `pending_jade`。第5轮原话不在这个目录。

| 页面名字 | 文件 | README 所写处理 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 增益 + 限幅，约 −13 LUFS |
| ② 上次有力那版 | `audio/op_replay.wav` | v001 候选2 的升调 + 补谐波 → 母带。没有 presence |
| ③ 有力 + 补中高频 | `audio/presence.wav` | 与②相同 → 2500 Hz +6 dB，Q=1.4 → 母带 |
| ④ 有力 + 更强补中高 | `audio/presence_strong.wav` | 与②相同 → 2500 Hz +12 dB，Q=1.4 → 母带 |

同目录还有 `run.py`、`report.json`、`requirements.txt`、`index.html`、`README.md`。响度、真峰值、音分、浊音重合和频段差写在该 README 的工程表里。那些数字不是听感结论。

`index.html` 的题是：

1. ③ 比 ②，空旷少了吗？（`data-q`：`③比②空旷`）
2. ④ 比 ②，空旷少了吗？（`data-q`：`④比②空旷`）
3. ③ 和 ④ 是更实了，还是仍然薄、仍然空，或者变尖了？（`data-q`：`③④更实、仍空，还是变尖`）

第 3 题的选项：更实了，也不尖；还是薄，还是空；变尖、刺耳了；③还行，④变尖了；和②差不多，听不出补了什么。

页面脚本把导出标题写成 `【我们的歌 第4轮反馈 2026-11-04】`。`feedback.md` 里的原话用的就是这个标题。不要改编号。

### v005 · `experiments/WOMENDE_FOCUS_2026-11-04_v005/`

试听页、处理脚本和四版音频都在。README 记录工程 `technical_pass = true`，听感行仍写 `pending_jade`。第5轮原话在 `feedback.md`，不在 README。没有混响、双叠、硬吸附、谐波激励；README 写没有 250 Hz 以下的削减，没有 `chain_dry`。

| 页面名字 | 文件 | README 所写处理 |
|---|---|---|
| ① 原音·同样响 | `audio/anchor.wav` | 增益 + 限幅，约 −13 LUFS |
| ② 上次有力那版 | `audio/op_replay.wav` | v001 的 +310 音分文件 → 同一补谐波 → 母带 |
| ③ 有力 + 只补唱出来的中高频 | `audio/shelf.wav` | 与②相同 → 谐波宽架 +10 dB（800→1600 Hz 升起，持平到 4000 Hz）→ 母带 |
| ④ 有力 + 字尾再压低 | `audio/tail.wav` | 与②相同 → 字尾衰减段再压 10 dB → 母带 |

同目录还有 `run.py`、`report.json`、`requirements.txt`、`index.html`、`README.md`、`feedback.md`。工程数字在 README 和 `report.json`。那些数字不是听感结论。

`index.html` 的题和导出用的 `data-q`：

1. ③ 比 ②，听得出多补的中高频吗？空旷有变化吗？（`③比②`）
2. ④ 比 ②，字尾的空旷少了吗？（`④比②空旷`）
3. ③ 或 ④ 有没有变尖、变嗲，或者像电音？（`尖嗲电音`）

页面脚本把导出标题写成 `【我们的歌 第5轮反馈 2026-11-04】`。`feedback.md` 里的原话用的就是这个标题和这三个 `data-q`。第 3 题没有选项。不要改写原话，也不要覆盖 `audio/`。

## Jade 原话（第3轮至第5轮）

第1轮、第2轮的全文在 v001、v002 的 `feedback.md`，此处不另抄。

### 第3轮

目录：`experiments/WOMENDE_FOCUS_2026-11-03_v003/feedback.md`。对应 v003 的试听页。

```
【我们的歌 第3轮反馈 2026-11-04】
哪版最有力：②③④分不出来
空旷比②少没少：不知道；③还是挺空旷，②③差不多
④会不会嗲：不会；问题还是空旷
补充：你说有利的话，二三四我还真分不出来
3比2空旷，少没少我不知道，反正3还是挺空旷的，他俩还是差不多
4不会嗲，但就是这个空旷
```

### 第4轮

目录：`experiments/WOMENDE_FOCUS_2026-11-04_v004/feedback.md`。对应 v004 的试听页。该文件在原话后面还有一段以「判定：」开头的文字，那不是导出原文。

```
【我们的歌 第4轮反馈 2026-11-04】
③比②空旷：和②差不多
④比②空旷：和②差不多
③④更实、仍空，还是变尖：和②差不多，听不出补了什么
补充：依旧空旷，依旧听不出2234有啥区别
```

### 第5轮

目录：`experiments/WOMENDE_FOCUS_2026-11-04_v005/feedback.md`。对应 v005 的试听页（`index.html` 的导出标题与 `data-q` 与下面这段一致）。

```
【我们的歌 第5轮反馈 2026-11-04】
③比②：听得出，但空旷差不多
④比②空旷：和②差不多
尖嗲电音：（未答）
补充：第三题，我没有选。第三题拿3和4的话，啥意思？4的空旷确实好像要更少一点点
但是还是比较有电，比如说在接近11.12秒的时候，这边还有一点小波动
3属于这种太空旷了,是的话，空旷稍微好了一点，但是还有一点，并且有一些地方不稳定，还带一点电
```

`feedback.md` 末尾有一行注：试听页第 3 题未作答。那一行不是 Jade 的原话。补充里的「第三题，我没有选」是原话。

## 书面记录里和原话不一致的位置

第4轮、第5轮的原话以各自 `feedback.md` 和 ROADMAP §11 为准。下面几处是页面做出时留下的句子，这次没有改那些文件：

- `AGENTS.md`「当前状态」仍写「v005 待听」。
- `experiments/WOMENDE_FOCUS_2026-11-04_v004/README.md` 与 `experiments/WOMENDE_FOCUS_2026-11-04_v005/README.md` 的听感行仍写 `pending_jade`。
- v001、v002 的 `README.md` 文末仍有 `pending_jade`。这两轮的原话在各自的 `feedback.md`。

## 怎么读已有章节

- §6 的四版表是 v002 的任务说明。v002 目录已经存在。不要把那张表当成第5轮之后的配方。
- §12 里 2026-11-04 有多行。写出 v005 目录的那一行，记的是当时已经做出的处理。第5轮原话的那一行只存原话，并写明 P0 继续由本文件交接。本文件不把其中任何一行延长成下一次算法。
- §3.1 写了空旷 / 录音机感的两条已记录成因（原因 A、原因 B），并有 v004 / v005 的测量补充。§3.3 是已证伪或已否决的方向。诊断时先查这两节。查完之后选哪一个杠杆，由接手方决定，不由本文件决定。
- §9.4 写 v005 已答。§11 / §12 的第5轮行是原话。

可复用的代码地址在 ROADMAP §9.3 和 `AGENTS.md` 的入口表。用到时按 §7.4 写下项目地址、版本、许可证、接口和适配。自写部分限于适配层、测量和守护逻辑。

## 流程清单

接手后按这个顺序。诊断没有写清之前，不开新的音频目录。

1. 读完「先读」里的文件。第3轮、第4轮、第5轮原话一字不改地使用。第5轮 `feedback.md` 末尾那一行注不是 Jade 的原话。
2. 写诊断：Jade 原话里哪一句对应哪一版（对照上面的文件表），相关数字在哪个 `report.json` 或 README 工程表。分开写工程事实和听感原话。不写「我听过」或「变好听了」。
3. 对照 §3.3 和永久否决项，标出哪些方向已经有判定。
4. 由接手方选定下一次杠杆。本文件没有预定。
5. 若要出试听：新建 `experiments/<主题>_<YYYY-MM-DD>_vNNN/`，目录必须在仓库的 `experiments/` 下，不能在桌面。不要使用 v001–v005 作为输出目录。
6. 一首歌。3–4 个处理版 + ① 原音·同样响。页面按 §7.2。成品响度按 §7.3。每个版本写明做了什么，以及 F2-K 是关还是开。
7. 250 Hz 以下无 EQ 削减。无混响、无双叠、无谐波激励、无硬吸附电音伪影。F1 不改音高。
8. §8.2 全部通过后才放上试听页。结果记 `technical_pass` 或失败原因。听感留 `pending_jade`。
9. Jade 回复后，原话写入该轮 `feedback.md`，并追加 ROADMAP §11 / §12。

## 可复制任务

```
你接手仓库 Voice-trans-Project-2.0 的 P0（《我们的歌》薄音）。先读 docs/ROADMAP.md（§1、§2、§3、§7、§8、§11、§12）、AGENTS.md、experiments/WOMENDE_P0_HANDLE.md，以及 v003、v004、v005 的 feedback.md。优先级：Jade 当前原话 > ROADMAP > AGENTS.md。AGENTS.md 仍写「v005 待听」；第5轮原话以 v005/feedback.md 和 ROADMAP §11 为准。

P0 继续。交接文件是 experiments/WOMENDE_P0_HANDLE.md。该文件不指定下一次算法。下一步杠杆由你根据 ROADMAP、各轮 report.json / README 里的测量、以及 Jade 第3轮、第4轮、第5轮原话自行诊断。诊断写清楚之后再决定要不要新开一轮。

硬约束：EQ 不削 250 Hz 以下；不要混响、双叠、谐波激励；不要硬吸附修音的电音伪影；F1 不改音高，整体升调只走默认关闭的 F2-K，并写明开关；工作方式按 ROADMAP §7；新目录只建在仓库 experiments/<主题>_<日期>_vNNN/，不要写到桌面。test_dataset/ 与已有实验输出只读，不要覆盖 v005 的 audio/。已存在的输出目录拒绝覆盖。你不能听音。technical_pass 不要写成变好听。老师 demo 等 Jade 点名「可以给老师」。未经要求不提交、不推送。
```
