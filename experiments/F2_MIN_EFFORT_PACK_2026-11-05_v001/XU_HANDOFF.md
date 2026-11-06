# 徐同学 · B1 音高并行（一页）

日期：2026-11-05（答案入库后更新）。字对齐与音高分开；**B1 不等 A3**。

## 可并行用的字对齐子集

- `experiments/F2_MIN_EFFORT_PACK_2026-11-05_v001/results/reliable_pairs.csv`
- **332 行**可靠字（OK 候选 + 歌词已确认 + 未踢坏点）
- 可与音高工作对照着用；**不是**整表 Qwen 金标

### 训练歌词（实唱为准）

- 离开我的依赖01：`…仿佛你在身边…`（官网「身旁」不用）
- 莫问归期01：`…刺无枉寻也如此…`（官网候选「不枉寻」不用）

### 联合评估时请排除

- bad **#5**（离开01 原唱「佛」，wrong_time）
- bad **#6**（离开01 原唱「答」，drop / 超出干声）

## 你现在做

1. 先任选一组本人 WAV（建议《心墙》01 对照微微跑调）。
2. 画音高曲线，标可能唱偏的短时间段（带前后文）。
3. 每个候选：`song, version, start_s, end_s, why_suspect, confidence, notes`
4. 交 Jade 标：希望修 / 故意 / 拿不准 / 不修。

## 路径

- worktree：`C:\Users\杨玉璋\.codex\worktrees\f2-initial-plan\Voice-trans-Project-2.0\`
- 本人 WAV：`test_dataset\my_singing\rerecord_2026-11-04_f2\wav\`
- 原唱人声（可选对照）：`test_dataset\my_singing\rerecord_2026-11-04_f2\ref_vocals\`

范围：三首重录的对照/轻度档；排除严重跑调与凤凰传奇旧素材。