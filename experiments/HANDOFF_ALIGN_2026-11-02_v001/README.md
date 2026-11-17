# 接手前需求对齐页（2026-11-02）

目的：新 Agent 接手前确认 Jade 对 F1 / F2 的真实目标。页里的音频是"方向探针"，故意夸张，**不是候选方案，也不计入 F1/F2 实验**。

- 试听页：`index.html`（同组内可按当前位置无缝切换；底部"生成答案"导出文字）
- 生成：`scripts/f2_hard_snap.py`（.venv-f2，复用 f2/poc_phase1.py 的 Praat 参数），`scripts/make_probes.py`（.venv-f1-adaptive，Pedalboard 0.9.26 现成效果器 + FFmpeg ebur128 计量）
- 响度：除两条"更响"探针外，统一 -18 LUFS，只施加固定增益。`f1_orig_loud` 受峰值上限 -1 dBFS 约束，实际 -14.2 LUFS；`f1_chain_loud` 经 Limiter，-13 LUFS。明细见 `levels.json`（含源文件哈希）。
- 工程检查：样本数与源一致（F1 探针 630784），无削波、无 NaN。混响与双叠会带来时间拖尾/副本，属于有意为之。硬吸附平均改动约 25 音分，最大 50 音分。
- 原始素材、旧实验未改动；没有推送 Git。
- 听感：pending_jade。Agent 没有听音能力。

探针含义见 index.html 各按钮说明；混响超出 PoC v1.2 范围，只用于问路。
