"""S1: reuse Pedalboard Distortion with measured level adaptation, not a new DSP kernel."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pedalboard
from pedalboard import Distortion
from scipy.signal import resample_poly

from f1_poc import read_pcm16, sample_stats, measure_loudness, export_wav, compare_audio

SETTINGS = {"02_density_mild.wav": {"drive_db": 12, "mix": .35},
            "03_density_strong.wav": {"drive_db": 18, "mix": .65}}


def density(audio, rate, drive_db, mix):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite non-empty mono samples")
    if not np.isfinite(mix) or not 0 <= mix <= 1 or not np.isfinite(drive_db) or not 0 <= drive_db <= 24:
        raise ValueError("Invalid mix/drive")
    if rate <= 0:
        raise ValueError("Invalid sample rate")
    if mix == 0 or np.max(np.abs(x)) < 1e-9:
        return x.copy(), {"bypass": True}
    size = max(1, round(rate * .02))
    blocks = x[:len(x)//size*size].reshape(-1, size)
    if not len(blocks):
        raise ValueError("At least one 20 ms block required")
    levels = np.sqrt(np.mean(blocks**2, axis=1))
    mask = np.repeat(levels > np.max(levels)*.1, size)
    mask = np.pad(mask, (0, len(x)-len(mask)), constant_values=False)
    active_rms = float(np.sqrt(np.mean(x[mask]**2)))
    target_rms = 10**(-16/20)
    input_gain = target_rms/active_rms
    working = x * input_gain
    # Existing SciPy resampling reduces aliasing; Pedalboard supplies the sound processor.
    up = resample_poly(working, 4, 1).astype(np.float32)
    wet = Distortion(drive_db=drive_db)(up, rate*4)
    wet = resample_poly(np.asarray(wet, dtype=np.float64), 1, 4)
    if len(wet) != len(x) or not np.all(np.isfinite(wet)):
        raise ValueError("Plugin/resampler changed length or produced invalid samples")
    wet_rms = float(np.sqrt(np.mean(wet[mask]**2)))
    if wet_rms < 1e-10:
        raise ValueError("Unexpected silent processing branch")
    wet_compensation = target_rms/wet_rms
    result = ((1-mix)*working + mix*wet*wet_compensation)/input_gain
    return result, {"bypass": False, "active_rms_dbfs": float(20*np.log10(active_rms)),
                    "working_rms_dbfs": -16, "input_gain_db": float(20*np.log10(input_gain)),
                    "wet_compensation_db": float(20*np.log10(wet_compensation)),
                    "drive_db": drive_db, "mix": mix, "oversampling": 4,
                    "active_block_rule": "20ms RMS > maximum block RMS * 0.1"}


def listening_page(report):
    cards = []
    for label, item in zip(["01 原音（仅匹配音量）", "02 温和密度处理", "03 较明显密度处理"], report["outputs"]):
        name = html.escape(item["file"], quote=True)
        cards.append(f'<section><h2>{label}</h2><audio controls preload="metadata" src="{name}"></audio>'
                     f'<p><a href="{name}" download>保存 WAV</a> · {item["loudness"]["integrated_lufs"]:.2f} LUFS</p></section>')
    start = report["phrase_seconds"][0]
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>Jade · F1 S1 单项密度实验</title>
<style>body{font:16px/1.7 system-ui;background:#f4f5f7;color:#202530;max-width:780px;margin:32px auto;padding:0 20px}section{background:white;border:1px solid #dce0e6;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:26px}h2{font-size:20px}audio{width:100%}a{color:#2953a6}.note{color:#596579}li{margin:6px 0}</style>
<h1>第一步：只验证饱满度／声音密度</h1>
<p>素材：本人《是谁在唱歌》中度跑调版。复用 Spotify Pedalboard 的谐波处理模块；不叠加 EQ、压缩、去齿音、混响或修音高。</p>
<p>只判断声音是否更充实、是否仍自然像本人。较明显档也可能更毛、更硬；更强不代表更好。工程检查通过，听感等待 Jade。</p>
<p><a href="feature-map.html">查看功能锚定图与后续顺序</a> · <a href="qa_report.json">检查数据</a></p>
<section><h2>先听同一句：原音 → 温和 → 较明显</h2>
''' + f'<p>取同一段 {start:.2f}–{start+3:.2f} 秒，每段 3 秒，中间停顿 0.35 秒。</p>' + '''
<audio controls preload="metadata" data-montage="true" src="04_phrase_compare.wav"></audio></section>
''' + ''.join(cards) + '''
<section><h2>你反馈后怎么走</h2><ul>
<li>更充实、自然：保留本档，再换另一条本人录音复验；此时才考虑下一功能。</li>
<li>有改善，但毛／硬／闷／压住：保留基准，只改同一模块一个控制量，最多修订一次。</li>
<li>无区别或没有更好：先核验播放与实际文件；最多一次有依据的修订，再无收益就暂停此候选。</li>
<li>明显更差或不像本人：不进入组合，定位原因；不能低成本修复就换候选。</li>
</ul><p>回复：更喜欢 01／02／03／无区别；更充实还是更毛、更硬；问题时间点（有则提供）。</p></section>
<script>let active=null;document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{if(active&&active!==p){const t=active.ended||active.dataset.montage||p.dataset.montage?0:active.currentTime;active.pause();p.currentTime=Math.min(t,p.duration||t);}active=p;}));</script></html>'''


def build(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Use a NEW output directory")
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    x, rate = read_pcm16(source)
    if len(x)/rate < 3 or np.sqrt(np.mean(x*x)) < 1e-5 or np.any(np.abs(x) >= 32767/32768):
        raise ValueError("Input is too short, silent or already at full scale")
    started = time.perf_counter()
    processed = [(x, {"bypass": True})] + [density(x, rate, **setting) for setting in SETTINGS.values()]
    meters = [measure_loudness(data, rate) for data, _ in processed]
    target = min(-18., *[m["integrated_lufs"] - 3.1 - m["true_peak_dbtp"] for m in meters])
    out.mkdir(parents=True)
    (out/"00_original.wav").write_bytes(source_bytes)
    report = {"feature": "S1-density", "source": str(source), "source_sha256": source_hash,
              "project": "https://github.com/spotify/pedalboard", "module": "pedalboard.Distortion",
              "pedalboard_version": pedalboard.__version__, "license": "GPL-3.0",
              "settings": SETTINGS, "target_lufs": target, "technical_pass": False,
              "listening_acceptance": "pending_jade", "outputs": [], "errors": []}
    names = ["01_before_matched.wav", *SETTINGS]
    for name, (data, adaptation), meter in zip(names, processed, meters):
        export_wav(data, rate, target-meter["integrated_lufs"], out/name)
        exported, output_rate = read_pcm16(out/name)
        stats = sample_stats(exported, output_rate)
        loud = measure_loudness(exported, output_rate)
        checks = compare_audio(x, exported, rate)
        if output_rate != rate:
            checks["errors"].append("Sample rate changed")
        if loud["true_peak_dbtp"] > -3 or abs(loud["integrated_lufs"]-target) > .2:
            checks["errors"].append("Headroom/loudness mismatch")
        checks["passed"] = not checks["errors"]
        report["outputs"].append({"file": name, "samples": stats, "loudness": loud,
                                  "adaptation": adaptation, "comparison": checks,
                                  "sha256": hashlib.sha256((out/name).read_bytes()).hexdigest()})
        report["errors"].extend(f"{name}: {error}" for error in checks["errors"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        report["errors"].append("Original source changed")
    report["technical_pass"] = not report["errors"]
    if report["technical_pass"]:
        starts = range(0, max(1, len(x)-3*rate+1), rate//4)
        start = max(starts, key=lambda i: np.mean(x[i:i+3*rate]**2))
        report["phrase_seconds"] = [start/rate, start/rate+3]
        parts = [read_pcm16(out/name)[0][start:start+3*rate] for name in names]
        gap = np.zeros(round(rate*.35))
        export_wav(np.concatenate([parts[0],gap,parts[1],gap,parts[2]]), rate, 0, out/"04_phrase_compare.wav")
        (out/"index.html").write_text(listening_page(report), encoding="utf-8")
        shutil.copyfile(Path(__file__).parent/"docs/F1_FEATURE_MAP.html", out/"feature-map.html")
    report["processing_seconds"] = time.perf_counter()-started
    (out/"qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out/"feedback.json").write_text(json.dumps({"feature":"S1-density", "status":"pending_jade", "preferred_file":None,
        "raw_feedback":None, "side_effects":[], "revision_count":0, "next_action":"await_jade"},indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test_dataset/my_singing/shige_medium.wav")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build(args.input, args.out)
    print(json.dumps({"technical_pass":result["technical_pass"],"errors":result["errors"],"out":args.out}))
    raise SystemExit(0 if result["technical_pass"] else 1)
