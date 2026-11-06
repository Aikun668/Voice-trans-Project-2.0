# F2 Task2 最小努力包 v001

## 目的
Jade 标记「希望修 / 故意这样唱 / 拿不准 / 不需要修」片段清单。不依赖徐同学音高。

## 目录
- `index.html` — Jade 唯一入口（相对路径播 `../../test_dataset/.../wav/*.wav`）
- `JADE_ONLY.md` — 三步操作
- `data/takes.json` — 10 条轻度/对照 takes + 任务一字时标（若有）
- `results/fragments_template.csv` — 空模板列
- `results/pitch_suggestions.json` — 音高候选（本轮 skipped）

## 默认优先
1. 心墙01-轻度跑调.wav（★）
2. 心墙01-对照微微跑调.wav
3. 离开 / 莫问 轻度+对照（钩子，后扩）

未纳入：严重跑调档（本阶段不先啃）。

## Agent 注意
- 不要宣称音高真理；Agent 听不见。
- 收答案后写入 `results/jade_answers_*.txt/json` 与 `results/fragments.csv`。
- 未要求勿 commit/push。

## 音高候选
base Anaconda python 无 librosa/parselmouth/crepe；未扫描其他 env 以免拖慢。UI 无候选仍可用。
