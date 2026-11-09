# F2 按确认切口重训 v001

2026-11-09。实验权重已训练并完成数值核验，听感为 `pending_jade`。新版训练入口在 `f2/train_confirmed.py`，本工作区的默认实验权重由 `f2/active_pitch_model.json` 指向。

## 核验与结果

- `confirmed_pairs.json` 冻结 Jade 确认的 17 对切口，含重复段选择、34 个源区间和 14 个 WAV 哈希。原话在 `feedback.md`。
- 千问只在确认范围附近复查字位。实际音高样本始终从确认区间内读取，调差用区间外的稳定同字测量。最终样本在 `prepared_v003/`，`audit.json` 逐对记录证据和排除原因。
- 训练用《心墙》和《离开我的依赖》的 4 对片段中 100 帧，含「学」手工目标的 16 帧。其余对保留核验记录。全程按歌曲分组，《莫问归期》留出。
- 梯度提升模型拟合移调幅度，方向和近准音是否保持由参考测量决定。留出歌曲有两对的 55 帧近准音，均输出零改动。
- 重新加载 `results/model.pkl` 后，55 帧留出预测逐条一致。既有《莫问归期》升半音 WAV 实测差约 100.42 音分，模型给约 +77.08、−77.08 和 0。该测试验方向、幅度和零改动，真实跑调修复和电音由 Jade 后续试听。
- `archive/` 收录旧 v001、v002、v003 权重、样本表、训练报告和脚本。`archive/inventory.json` 记源路径及哈希。

## 本机复跑

在本工作区根目录运行。使用现有独立环境的 NumPy、scikit-learn 和 Praat 库。

```powershell
$py = 'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/.venv-f1-adaptive/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:F2_PRAAT_SITE = 'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/.venv-f2/Lib/site-packages'
$exp = 'experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001'
$context = 'C:/Users/杨玉璋/.codex/worktrees/f2-initial-plan/Voice-trans-Project-2.0/experiments/F2_TASK3_PAIR_AUDIT_2026-11-05_v001/逐字候选对照.csv'
$manual = 'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/experiments/F2_XINQIANG_XUE_CONTOUR_2026-11-08_v001/report.json'
$control = 'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/experiments/F2_ONE_NOTE_ROUND_TRIP_2026-11-07_v002/report.json'
& $py -B f2/train_confirmed.py verify --prepared "$exp/prepared_v003" --results "$exp/results" --control-report $control
& $py -B f2/train_confirmed.py predict --id 3
```

需要复跑训练时，用新的输出目录。预测输出是供后续合成的逐帧移调建议。

```powershell
& $py -B f2/train_confirmed.py prepare --pairs "$exp/confirmed_pairs.json" --probes "$exp/word_probes.json" --context $context --manual $manual --out "$exp/prepared_recheck"
& $py -B f2/train_confirmed.py train --prepared "$exp/prepared_recheck" --out "$exp/results_recheck"
```
