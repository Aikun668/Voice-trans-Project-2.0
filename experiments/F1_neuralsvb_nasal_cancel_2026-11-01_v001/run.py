"""A01 nasal reverse-calibration test: a2p vs colour_match 100% / 50% (single sample).

Read-only on the original experiment. Reuses:
- experiments/F1_neuralsvb_pitch_target_v001/colour_match.py (zero-phase output EQ, already used before)
- f1_poc.py + run_poc_batch.check_output (project thresholds unchanged)
No model re-run, no weight changes, no DSP other than the output calibration.
Listening status: pending_jade.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neuralsvb_f1"))
sys.path.insert(0, str(ROOT / "experiments" / "F1_neuralsvb_pitch_target_v001"))

if not shutil.which("ffmpeg"):
    links = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
if not shutil.which("ffmpeg"):
    raise SystemExit("ffmpeg not found")

import f1_poc as qa  # noqa: E402
import neuralsvb_infer as nsvb  # noqa: E402
import colour_match as cm  # noqa: E402
from run_poc_batch import check_output, common_target, sha256  # noqa: E402

SRC = ROOT / "experiments" / "F1_neuralsvb_8songs_2026-11-01_v001"
DATA = ROOT / "test_dataset" / "f1_dataset_v2"
OUT = ROOT / "experiments" / "F1_neuralsvb_nasal_cancel_2026-11-01_v001" / "run-001"
if OUT.exists():
    raise SystemExit(f"{OUT} already exists")
OUT.mkdir(parents=True)
AUDIO = OUT / "listening" / "audio"
AUDIO.mkdir(parents=True)

SID, NID = "A01", "A02"
SR = nsvb.SAMPLE_RATE


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    return qa.read_pcm16(path)


def resample_to_22(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    import math
    from scipy import signal
    g = math.gcd(sr, SR)
    return signal.resample_poly(x, SR // g, sr // g)


orig48, sr48 = load_wav(DATA / "A_thin" / "self" / "wav" / "平凡之路薄音版.wav")
normal48, _ = load_wav(DATA / "A_thin" / "self" / "wav" / "平凡之路正常版.wav")
a2p, sr_a = load_wav(SRC / "listening" / "audio" / f"{SID}_a2p.wav")
voc, sr_v = load_wav(SRC / "listening" / "audio" / f"{SID}_vocoder_only.wav")
if sr_a != SR or sr_v != SR:
    raise ValueError((sr_a, sr_v))

# colour_match: reference = the input's own long-term colour; processed = a2p.
orig22 = resample_to_22(orig48, sr48)
n = min(len(orig22), len(a2p))
ref = orig22[:n]
proc = a2p[:n]
eq100, rep100 = cm.colour_match(ref, proc, SR, cm.ColourConfig(strength=1.0))
eq50, rep50 = cm.colour_match(ref, proc, SR, cm.ColourConfig(strength=0.5))
eq100 = np.asarray(eq100, dtype=np.float64)
eq50 = np.asarray(eq50, dtype=np.float64)

# One common loudness target for every file in this small A/B.
raw = {"original": (orig48, sr48), "a2p": (a2p, SR), "eq100": (eq100, SR),
       "eq50": (eq50, SR), "vocoder_only": (voc, SR), "normal": (normal48, sr48)}
measured = {k: qa.measure_loudness(x, sr) for k, (x, sr) in raw.items()}
target = common_target(list(measured.values()))
print(f"common_target_lufs={target:.3f}")

paths = {}
for key, (x, sr) in raw.items():
    dest = AUDIO / f"{SID}_{key}.wav"
    qa.export_wav(x, sr, target - measured[key]["integrated_lufs"], dest)
    paths[key] = dest

orig_m, rate_o = qa.read_pcm16(paths["original"])
loud_o = qa.measure_loudness(orig_m, rate_o)
hp = nsvb._bootstrap(None)
engine = SimpleNamespace(hp=hp)
checks = {}
for key in ("a2p", "eq100", "eq50", "vocoder_only"):
    x, rate = qa.read_pcm16(paths[key])
    checks[key] = check_output(engine, orig_m, rate_o, loud_o, x, rate)
    print(key, checks[key]["technical_pass"], checks[key]["errors"])

report = {
    "experiment": "F1_neuralsvb_nasal_cancel_2026-11-01_v001",
    "sample": SID, "song": "平凡之路", "variant": "薄音版",
    "hypothesis": "a2p 的鼻音/闷音主要来自模型输出相对原音的长期频谱倾斜；用既有 colour_match 做反向 EQ，"
                  "100% 与 50% 两个强度，检查鼻音是否减轻、是否伤到饱满度。",
    "not_doing": "没有重跑模型、没有改权重、没有处理 A03 的词粘连（那是时间包络/辅音层），也没有处理其他 7 条。",
    "target_lufs": target,
    "gain_reports": {"eq100": rep100, "eq50": rep50},
    "files": {k: {"file": str(p.relative_to(OUT)).replace("\\", "/"), "sha256": sha256(p),
                  "lufs": qa.measure_loudness(*qa.read_pcm16(p))["integrated_lufs"]}
              for k, p in paths.items()},
    "checks": checks,
    "listening_status": "pending_jade",
}
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def row(label, key, note=""):
    extra = f'<br><span style="font-size:12px">{note}</span>' if note else ""
    return (f'<div class="row"><label>{label}{extra}</label>'
            f'<audio controls preload="metadata" src="audio/{SID}_{key}.wav"></audio></div>')

def band(name):
    vals = [report["gain_reports"][name]["band_gain_db"][k] for k in ("0-500", "500-1500", "1500-4000", "4000-8000")]
    return f"{vals[0]:+.1f} / {vals[1]:+.1f} / {vals[2]:+.1f} / {vals[3]:+.1f} dB"

checks_rows = []
for key, v in checks.items():
    pc = v["pitch_vs_original"]
    pitch = (f"{pc['median_abs_cents']:.1f} / {pc['p95_abs_cents']:.1f}"
             if "median_abs_cents" in pc else "不可用")
    checks_rows.append(
        f"<tr><td>{key}</td><td>{v['loudness']['integrated_lufs']:.2f}</td>"
        f"<td>{v['loudness']['true_peak_dbtp']:.2f}</td><td>{v['samples']['clipped_samples']}</td>"
        f"<td>{v['duration_gap_s']:.4f}</td><td>{pitch}</td>"
        f"<td>{'通过' if v['technical_pass'] else '未通过'}</td></tr>")
checks_table = "".join(checks_rows)

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A01 鼻音反向校准测试</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0;font-size:14px}}.row{{display:grid;grid-template-columns:200px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #dce0e6;padding:5px 7px;text-align:right}}th{{background:#eef1f5;text-align:center}}td:first-child{{text-align:left}}</style>
<h1>A01 鼻音反向校准测试 · 100% / 50%</h1>
<div class="note"><b>这是单条小测试，不是 8 条批量。</b>原音、a2p、仅声码器沿用上一轮；新增两个版本 = 在 a2p 上做反向长期频谱校准。<br>
<b>假设：</b>a2p 相对原音，低频 <0.5k 约 −3.4 dB、4–8k 约 −5.0 dB、0.5–1.5k 约 +1.1 dB；用既有 colour_match 把这层倾斜抵消。<br>
<b>不处理：</b>A03 的“词粘连”。已测到它是模型的时间包络/辅音层（2–8Hz 调制 0.30→0.17，噪声型辅音帧 253→127），EQ 不会修这一层。<br>
<b>预期：</b>100% 最能减鼻音/闷音，但可能把模型的中频抬升一起削掉，声音变薄或变亮；50% 是折中。<br>
<b>请听：</b>先听 8.7–10.7s（A01 倾斜最强的段），再听整条。重点判断：鼻音有没有减轻、有没有变薄/变亮/换人感。如果两个都听不出，说明 A01 的鼻音不是这条倾斜单独造成，下一步换 A06 或查其他层。<br>
<b>工程：</b>全部统一到 {target:.2f} LUFS；EQ 版本沿用原阈值检查（见下表）。Agent 无听音能力，技术通过不等于变好听，听感 pending_jade。</div>
<div class="card"><h3>A01 平凡之路 · 薄音版（13.14s）</h3>
<p class="meta">重点听 8.7–10.7s（a2p 中频强调 +7.8 dB）；候选 2 7.7–9.7s（+7.5）；候选 3 2.5–4.5s（+6.3）；弱对照 6.6–8.6s（−0.2）。</p>
{row("原音（48kHz）", "original")}
{row("NeuralSVB a2p（22.05kHz）", "a2p")}
{row("a2p + 反向校准 100%", "eq100", "低频/0.5–1.5k/1.5–4k/4–8k 增益: " + band("eq100"))}
{row("a2p + 反向校准 50%", "eq50", "低频/0.5–1.5k/1.5–4k/4–8k 增益: " + band("eq50"))}
{row("仅声码器对照（22.05kHz）", "vocoder_only", "不是候选；只说明重建本身的影响")}
{row("同曲正常版参考（A02 原音）", "normal", "不同一次演唱，仅作音色参考")}
</div>
<div class="card"><h3>工程检查（原阈值，未放宽）</h3>
<table><tr><th>版本</th><th>响度 LUFS</th><th>真峰值 dBTP</th><th>削波</th><th>时长差 s</th><th>音高中位/P95 音分</th><th>结果</th></tr>
{checks_table}</table>
<p class="meta">音高诊断未通过是上一轮已记录的声码器重建抖动，不是本轮 EQ 新增；EQ 版本没有削波、时长不变。技术未通过不等于不能试听，但也不能算通过。</p></div>
<p class="meta">原始素材未改动；详细记录见 report.json。</p></html>'''
(OUT / "listening" / "index.html").write_text(page, encoding="utf-8")
print(OUT / "listening" / "index.html")
