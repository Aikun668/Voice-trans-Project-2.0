# F2 最小操作包（2026-11-05）

Agent 已完成：五组 Qwen 统一打标、坏点清单、歌词候选预填、你的一页操作页、徐同学交接、A3 冒烟输入清单。
你只需要打开 `index.html` 做两步，生成答案发回来。

## 你现在做什么（约 10–15 分钟）

1. 双击打开本目录 `index.html`（用 Chrome / Edge）。
2. **第一步**：确认另外四组歌词（心墙已确认可跳过）。每组点「听本人对照」，对 →「歌词正确」；有改口 → 改字 →「我改过」。莫问归期 01 再选原唱 A/B。
3. **第二步**：只听 17 处坏点（零时长/漂移），每处约 1.4 秒，点一个选项。
4. 点「生成答案并复制」，粘贴发回给我。

不要整表听、不要改源音频、不要等 MFA。

## Agent 已交付文件

| 文件 | 用途 |
|---|---|
| `index.html` | 你的最小操作页 |
| `results/labeled_status.csv` | 349 行全量打标（OK候选/零时长/漂移可疑） |
| `results/bad_spots.csv` | 17 处坏点 |
| `results/proposed_lyrics.json` | 五组候选歌词与路径 |
| `results/reliable_pairs_template.csv` | 心墙非坏点模板（抽听前不可训练） |
| `results/a3_smoke_manifest.json` | MFA/HubertFA 冒烟输入 |
| `results/summary.json` | 统计 |
| `XU_HANDOFF.md` | 给徐同学，音高并行 |
| `A3_SMOKE_READY.md` | 复查准备说明 |
| `JADE_ONLY.md` | 你的超短清单（可只看这个） |

## 统计

- 总字时标：349
- 零时长：12
- 漂移可疑（扩窗 Δstart≥0.35s）：5
- OK候选：332（≠可训练）

## 分工不变

- 你：听感/歌词确认/坏点选项
- 徐：音高候选（见 XU_HANDOFF.md），不堵对齐
- Agent：打标、汇总、后续可靠表；不能听音
