# F2 重录清唱 · 2026-11-04

给 F2（徐）用的跑调档素材。**不是薄音。**

## 三档含义（务必分清）

| 文件名里的档位 | 含义 |
|---|---|
| **对照微微跑调** | 词基本对、整体接近正常，但**故意留一点点跑调**；不是干净 normal，也不是轻度 |
| **轻度跑调** | 明显有几处长音停错 |
| **严重跑调** | 更多错音或整段偏音（原「重度」并入此档） |

不要把「对照微微跑调」当成完美零跑调对照。

## 文件清单（中文歌名）

| 歌 | take | 对照微微跑调 | 轻度跑调 | 严重跑调 |
|---|---|---|---|---|
| 心墙 | 01 | 心墙01-对照微微跑调 | 心墙01-轻度跑调 | 心墙01-严重跑调 |
| 离开我的依赖 | 01 | 离开我的依赖01-对照微微跑调 | 离开我的依赖01-轻度跑调 | 离开我的依赖01-严重跑调 |
| 离开我的依赖 | 02 | 离开我的依赖02-对照微微跑调 | 离开我的依赖02-轻度跑调 | 离开我的依赖02-严重跑调 |
| 莫问归期 | 01 | 莫问归期01-对照微微跑调 | 莫问归期01-轻度跑调 | 莫问归期01-严重跑调 |
| 莫问归期 | 02 | 莫问归期02-对照微微跑调 | 莫问归期02-轻度跑调 | 莫问归期02-严重跑调 |

- wav/：48 kHz mono PCM16（工程用）
- raw/：手机原 m4a
- 旧英文文件名（xinqiang / likaiwodeyilai / mowenguiqi + micro_off/mild/severe）已全部改为上表中文名

## 目录结构

- wav/ — 清唱工程 wav
- raw/ — 清唱原 m4a
- ref_vocals/ — 三首原唱 demucs 人声（与 test_dataset/reference/vocals/ 相同，方便和 wav/ 对照）

## 原唱参考（已补齐）

| 歌 | 歌手 | 原唱 | demucs 人声 | F2 镜像 |
|---|---|---|---|---|
| 心墙 | 郭静 | test_dataset/reference/心墙_郭静.wav | test_dataset/reference/vocals/心墙_郭静_vocals.wav | ref_vocals/心墙_郭静_vocals.wav |
| 离开我的依赖 | 王艳薇 | test_dataset/reference/离开我的依赖_王艳薇.wav | test_dataset/reference/vocals/离开我的依赖_王艳薇_vocals.wav | ref_vocals/离开我的依赖_王艳薇_vocals.wav |
| 莫问归期 | 蒋雪儿 | test_dataset/reference/莫问归期_蒋雪儿.wav | test_dataset/reference/vocals/莫问归期_蒋雪儿_vocals.wav | ref_vocals/莫问归期_蒋雪儿_vocals.wav |

- 源：YouTube 官方 MV（心墙 VhCRokYVF1I，离开我的依赖 p13tH7Etlzc，莫问归期 XaTFvLSP1oI）；源 m4a 与 wav 同放在 test_dataset/reference/。
- 去伴奏：demucs htdemucs --two-stems=vocals。
- 心墙 MV 片头约 4.4s 静音，对齐时注意。