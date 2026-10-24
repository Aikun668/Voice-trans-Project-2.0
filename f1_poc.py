"""F1 v0.1: reproducible offline DSP experiment, not an aesthetic scoring model.

Requires Python 3.10+, numpy, scipy and FFmpeg on PATH. No model downloads.
Input contract for this first experiment: mono, PCM16 WAV, at least 3 seconds.
The sample rate is preserved. Every run requires a NEW output directory.
"""
import argparse
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np
from scipy import signal


PRESETS = {
    "light": (
        "highpass=f=55:p=2,"
        "equalizer=f=250:t=q:w=0.8:g=-1,"
        "equalizer=f=3200:t=q:w=1:g=-0.8,"
        "acompressor=threshold=0.251189:ratio=1.35:attack=25:release=180:"
        "knee=2:detection=rms:makeup=1,"
        "deesser=i=0.08:m=0.15:f=0.5:s=o"
    ),
    "medium": (
        "highpass=f=65:p=2,"
        "equalizer=f=250:t=q:w=0.8:g=-1.8,"
        "equalizer=f=3200:t=q:w=1:g=-1.5,"
        "acompressor=threshold=0.177828:ratio=1.7:attack=25:release=200:"
        "knee=2:detection=rms:makeup=1,"
        "deesser=i=0.12:m=0.22:f=0.5:s=o"
    ),
}
OUTPUT_NAMES = ["01_before_matched.wav", "02_F1_light.wav", "03_F1_medium.wav"]
PRESETS_V2 = {
    "light": "highpass=f=55,bass=g=1.5:f=300,equalizer=f=900:t=q:w=0.7:g=-2,"
             "treble=g=-1.5:f=4000,acompressor=threshold=0.158489:ratio=1.8:attack=15:release=160:makeup=1",
    "strong": "highpass=f=55,bass=g=3:f=350,equalizer=f=900:t=q:w=0.7:g=-4,"
              "treble=g=-3:f=3200,acompressor=threshold=0.1:ratio=3:attack=8:release=180:makeup=1",
}
PRESETS_V3 = {
    "light": "highpass=f=55,bass=g=4:f=350,equalizer=f=900:t=q:w=0.7:g=-5,"
             "treble=g=-3.5:f=3200,acompressor=threshold=0.089125:ratio=3.5:attack=12:release=180:makeup=1",
    "strong": "highpass=f=55,bass=g=5:f=350,equalizer=f=900:t=q:w=0.7:g=-6,"
              "treble=g=-4:f=3200,acompressor=threshold=0.079433:ratio=4:attack=15:release=200:makeup=1",
}


def run_ffmpeg(arguments, input_bytes=None):
    command = ["ffmpeg", "-hide_banner", "-nostdin", *arguments]
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result


def read_pcm16(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("First experiment accepts mono PCM16 WAV only; convert a COPY explicitly.")
        rate = stream.getframerate()
        count = stream.getnframes()
        data = np.frombuffer(stream.readframes(count), dtype="<i2").astype(np.float64) / 32768
    if len(data) != count:
        raise ValueError("Truncated WAV payload")
    return data, rate


def db(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def sample_stats(data, rate):
    return {
        "sample_rate": rate, "channels": 1, "bits_per_sample": 16,
        "frames": len(data), "duration_seconds": len(data) / rate,
        "sample_peak_dbfs": db(np.max(np.abs(data))),
        "rms_dbfs": db(np.sqrt(np.mean(data ** 2))),
        "clipped_samples": int(np.count_nonzero(np.abs(data) >= 32767 / 32768)),
        "dc_offset": float(np.mean(data)),
    }


def float_input(rate):
    return ["-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0"]


def measure_loudness(data, rate):
    # loudnorm is used for its INPUT meter only. Its processed audio is discarded.
    result = run_ffmpeg([
        *float_input(rate), "-af", "loudnorm=I=-18:TP=-3:LRA=11:print_format=json",
        "-f", "null", "-",
    ], data.astype("<f4").tobytes())
    log = result.stderr.decode("utf-8", errors="replace")
    raw = json.loads(log[log.rfind("{"):log.rfind("}") + 1])
    measured = {
        "integrated_lufs": float(raw["input_i"]),
        "true_peak_dbtp": float(raw["input_tp"]),
        "loudness_range_lu": float(raw["input_lra"]),
    }
    if not all(math.isfinite(value) for value in measured.values()):
        raise ValueError("Loudness is not measurable; input may be silent or too short.")
    return measured


def process_audio(source, chain):
    result = run_ffmpeg([
        "-v", "error", "-f", "wav", "-i", str(source), "-map", "0:a:0",
        "-af", chain, "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
    ])
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)


def export_wav(data, rate, gain_db, destination):
    # A single constant gain matches loudness without a second compressor/limiter.
    run_ffmpeg([
        "-v", "error", *float_input(rate), "-af", f"volume={gain_db:.8f}dB",
        "-ar", str(rate), "-ac", "1", "-map_metadata", "-1",
        "-c:a", "pcm_s16le", "-n", str(destination),
    ], data.astype("<f4").tobytes())


def frame_rms(data, rate, seconds=0.02):
    size = max(1, round(rate * seconds))
    frames = data[:len(data) // size * size].reshape(-1, size)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def band_levels(data, rate):
    frequencies, density = signal.welch(data, rate, nperseg=min(8192, len(data)))
    total = np.sum(density[(frequencies >= 80) & (frequencies <= 8000)])
    bands = {}
    for name, lo, hi in [("body_80_500", 80, 500), ("mid_500_3000", 500, 3000),
                         ("presence_3000_8000", 3000, 8000)]:
        energy = np.sum(density[(frequencies >= lo) & (frequencies < hi)])
        bands[name] = float(10 * np.log10(max(energy, 1e-20) / max(total, 1e-20)))
    return bands


def estimate_pitch(data, rate):
    # Diagnostic autocorrelation tracker. This is NOT melody recognition or F2.
    divisor = math.gcd(rate, 8000)
    low = signal.resample_poly(data, 8000 // divisor, rate // divisor)
    size, hop = 640, 160  # 80 ms / 20 ms at 8 kHz
    values = []
    for start in range(0, len(low) - size + 1, hop):
        frame = low[start:start + size]
        if np.sqrt(np.mean(frame ** 2)) < 0.008:
            values.append(np.nan)
            continue
        frame = (frame - np.mean(frame)) * np.hanning(size)
        corr = signal.correlate(frame, frame, method="fft")[size - 1:]
        corr /= max(corr[0], 1e-12)
        peaks, _ = signal.find_peaks(corr[16:134])  # roughly 60..500 Hz, this male sample
        peaks = peaks + 16
        if not len(peaks) or np.max(corr[peaks]) < 0.7:
            values.append(np.nan)
            continue
        # Prefer earliest strong peak, avoiding integer multiples of a period.
        lag = int(peaks[corr[peaks] >= 0.92 * np.max(corr[peaks])][0])
        y0, y1, y2 = corr[lag - 1:lag + 2]
        denominator = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-12 else 0
        values.append(8000 / (lag + delta))
    return np.array(values)


def compare_audio(before, after, rate):
    errors, warnings = [], []
    details = {"passed": False, "errors": errors, "warnings": warnings}
    if len(before) != len(after):
        errors.append("Sample count changed (trim, pad or time stretch).")
        return details
    if not np.all(np.isfinite(after)) or np.max(np.abs(after)) < 1e-5:
        errors.append("Non-finite or silent output.")
        return details
    if np.any(np.abs(after) >= 32767 / 32768):
        errors.append("Output contains full-scale/clipped samples.")
    if abs(np.mean(after)) > 0.01:
        errors.append("Excessive DC offset.")
    # Correlation is gain independent; it catches large shifts and corrupted content.
    correlation = float(np.corrcoef(before, after)[0, 1])
    details["waveform_correlation"] = correlation
    if not math.isfinite(correlation) or correlation < 0.85:
        errors.append("Large waveform/content change or time shift.")
    b_rms, a_rms = frame_rms(before, rate), frame_rms(after, rate)
    active = b_rms > max(np.max(b_rms) * 0.08, 1e-4)
    gain = 20 * np.log10(np.maximum(a_rms[active], 1e-12) / b_rms[active])
    if len(gain):
        gain -= np.median(gain)
        spread = float(np.percentile(gain, 95) - np.percentile(gain, 5))
        details["active_gain_spread_db"] = spread
        if spread > 8:
            errors.append("Excessive dynamic change in active singing.")
        if np.min(gain) < -12:
            errors.append("Unexpected dropout in active singing.")
    # Catch newly introduced transients relative to the signal at matched RMS.
    scale = np.sqrt(np.mean(after ** 2) / max(np.mean(before ** 2), 1e-20))
    jump_ratio = float(np.max(np.abs(np.diff(after))) /
                       max(np.max(np.abs(np.diff(before))) * scale, 1e-9))
    details["max_sample_jump_ratio"] = jump_ratio
    if jump_ratio > 2.5:
        errors.append("Large new sample discontinuity (possible click).")
    b_bands, a_bands = band_levels(before, rate), band_levels(after, rate)
    changes = {key: a_bands[key] - b_bands[key] for key in b_bands}
    details["relative_band_change_db"] = changes
    # Compare meaningful bands only. Below -50 dB relative energy, a large dB
    # ratio may just be PCM16 rounding of an almost empty band (e.g. a sine test).
    if any(abs(value) > 6 and max(b_bands[key], a_bands[key]) > -50
           for key, value in changes.items()):
        errors.append("More than 6 dB relative band change; review excessive EQ/de-essing.")
    b_pitch, a_pitch = estimate_pitch(before, rate), estimate_pitch(after, rate)
    common = np.isfinite(b_pitch) & np.isfinite(a_pitch)
    details["pitch_comparable_frames"] = int(np.sum(common))
    if np.sum(common) >= 10:
        cents = np.abs(1200 * np.log2(a_pitch[common] / b_pitch[common]))
        details["pitch_median_abs_cents"] = float(np.median(cents))
        details["pitch_p95_abs_cents"] = float(np.percentile(cents, 95))
        if np.median(cents) > 10 or np.percentile(cents, 95) > 35:
            errors.append("Pitch diagnostic changed unexpectedly (manual check required).")
    else:
        warnings.append("Too few confident pitch frames; pitch preservation not verified.")
    details["passed"] = not errors
    return details


def make_report_markdown(report):
    lines = [f"# {report['version']}：茉莉花试听实验", "",
             "技术检查：" + ("通过" if report["technical_pass"] else "未通过，请勿直接验收"),
             "", "**听感验收：等待 Jade。技术通过不等于音色美化成功。**", "",
             "先听 01（原音仅调音量），再听 02（轻处理）、03（稍强处理）。",
             "00 是逐字节复制的原始录音，响度较大，不用于直接比较美化效果。", "",
             f"三个试听版本共同目标：{report['target_lufs']:.2f} LUFS；全程只用固定增益匹配响度。", "",
             "| 文件 | 秒数 | LUFS | 真峰值 dBTP | 削波样本 | 音高变化中位数/95分位（音分） |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in report["outputs"]:
        stats, loud, check = item["samples"], item["loudness"], item["comparison"]
        pitch = f"{check.get('pitch_median_abs_cents', float('nan')):.2f} / {check.get('pitch_p95_abs_cents', float('nan')):.2f}"
        lines.append(f"| {item['file']} | {stats['duration_seconds']:.6f} | {loud['integrated_lufs']:.2f} | {loud['true_peak_dbtp']:.2f} | {stats['clipped_samples']} | {pitch} |")
    lines += ["", "## 处理与边界", "",
              "轻处理：55 Hz 高通、250 Hz -1 dB、3.2 kHz -0.8 dB、1.35:1 轻压缩、轻去齿音。",
              "稍强处理：65 Hz 高通、250 Hz -1.8 dB、3.2 kHz -1.5 dB、1.7:1 压缩、稍强去齿音。",
              "参数为初始实验预设，未经过本人听感优化；两组都不移调、不加混响、不降噪、不裁剪。",
              "这些参数并不代表素材确实有齿音或低频问题；若无改善，下轮应旁路相应处理。", "",
              "## 已检查", "",
              "实际读取并重新解码导出 WAV；校验原始文件哈希、规格、样本数、响度、真峰值、削波、直流偏置、",
              "波形相关性、活跃演唱段异常衰减、新增跳变、频带变化和粗略音高变化。",
              "阈值是工程拦截线，并非公认的美化评分标准。音高检测仅作男声音区（60–500 Hz）诊断。", "",
              "## 待人工试听", "",
              "当前工具不支持直接听音，未宣称已完成主观试听。以下问题仍须人工判断：", "",
              "- 声音是否更舒服、更饱满，还是变闷、变薄？",
              "- 大小声是否更均匀，情绪起伏是否还在？",
              "- 咬字、气声、颤音是否自然，有无抽吸、齿音损伤或机械感？",
              "- 处理后是否仍像本人？更喜欢 01、02 还是 03，或几乎听不出区别？", "",
              "建议重点比较约 0.9–6.5 秒、7.1–12.1 秒的演唱；首尾及中间停顿均保留。", "",
              "本轮未测试其他歌手/其他样本，未验证 F2 串联，也未代表双方确认接口规格。"]
    if report["errors"]:
        lines += ["", "## 拦截问题", "", *[f"- {error}" for error in report["errors"]]]
    if report.get("profile") in ("v2", "v3"):
        lines = [line.replace("03（稍强处理）", "03（强处理）") for line in lines
                 if not line.startswith(("轻处理：", "稍强处理：", "这些参数并不"))]
        lines += ["", "## 第二版参数与依据", "",
                  "第一版人工反馈：原音、轻档、稍强档整体听不出区别；第一版未通过听感验收。",
                  "轻档：300 Hz 低架 +1.5 dB、900 Hz -2 dB、4 kHz 高架 -1.5 dB；压缩 1.8:1。",
                  "强档：350 Hz 低架 +3 dB、900 Hz -4 dB、3.2 kHz 高架 -3 dB；压缩 3:1。",
                  "两档均保留 55 Hz 高通；本轮旁路去齿音，避免无依据叠加处理。",
                  "方向是更厚、更暗、更均匀；强档可能发闷或损失动态，是试听候选，并非已证明的改善。",
                  "04 是 1–4 秒同一句的拼接对照（原音→轻档→强档，间隔 0.35 秒），不作为模块输出。"]
    if report.get("profile") == "v3":
        lines = [line for line in lines if not line.startswith(("轻档：", "强档：", "04 是"))]
        lines += ["", "## 第三版（继续加强）", "",
                  "v0.2 强档人工反馈：认真听可辨认，略有改善；本轮以此方向继续增强。",
                  "增强档：350 Hz 低架 +4 dB、900 Hz -5 dB、3.2 kHz 高架 -3.5 dB；压缩 3.5:1。",
                  "更强档：350 Hz 低架 +5 dB、900 Hz -6 dB、3.2 kHz 高架 -4 dB；压缩 4:1。",
                  "05 为上一轮强档的未改动副本；04 同句顺序是原音→上一轮强档→本轮增强→本轮更强。",
                  "本轮更强不代表更好，需重点听有无发闷、咬字损失、抽吸或呼吸声过突出。"]
    return "\n".join(lines) + "\n"


def make_listening_page(report):
    descriptions = ["01 · 原音（只匹配音量）", "02 · F1 轻处理", "03 · F1 稍强处理"]
    cards = []
    for title, item in zip(descriptions, report["outputs"]):
        name = html.escape(item["file"], quote=True)
        cards.append(f'<section><h2>{title}</h2><audio controls preload="metadata" src="{name}"></audio>'
                     f'<p><a href="{name}" download>保存 WAV</a> · {item["loudness"]["integrated_lufs"]:.2f} LUFS</p></section>')
    return '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Jade · F1 茉莉花试听 v0.1</title>
<style>
body{font:16px/1.7 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:740px;margin:32px auto;padding:0 20px}
h1{font-size:26px}h2{font-size:19px;margin:0 0 12px}section{background:white;border:1px solid #dce0e6;border-radius:12px;padding:20px;margin:16px 0}
audio{width:100%}a{color:#2953a6}.note{color:#525d70}p{margin:8px 0}
</style>
<h1>Jade · F1 茉莉花试听 v0.1</h1>
<p>同一段 13.01 秒录音，三个版本已匹配响度。先听原音，再比较两档处理。</p>
<p class="note">技术检查通过 ≠ 更好听。请用同一副耳机、固定系统音量试听。开始播放另一版本时，会暂停上一版本，接着同一时间点播放。</p>
''' + "\n".join(cards) + '''
<section><h2>这一轮请判断</h2>
<p>你更喜欢哪版？是否更舒服、更饱满，还是变闷、变薄？咬字和情绪起伏是否还自然？</p>
<p>约 0.9–6.5 秒、7.1–12.1 秒是主要演唱段，首尾及停顿都保留了。</p>
<p class="note">可回复：偏好 01/02/03/无明显区别；问题时间点；具体听感。</p>
<p><a href="00_original.wav" download>保存未调音量的原始副本（较响）</a> · <a href="qa_report.json">技术检查数据</a></p>
</section>
<script>
let active = null;
document.querySelectorAll('audio').forEach(player => {
  player.addEventListener('play', () => {
    if (active && active !== player) {
      const position = active.ended ? 0 : active.currentTime;
      active.pause();
      player.currentTime = Math.min(position, player.duration || position);
    }
    active = player;
  });
});
</script></html>
'''


def experiment(source, out, profile="v1"):
    presets = {"v1": PRESETS, "v2": PRESETS_V2, "v3": PRESETS_V3}[profile]
    names = ["01_before_matched.wav", "02_F1_light.wav", "03_F1_strong.wav"] if profile != "v1" else OUTPUT_NAMES
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Output directory already exists; use a new version directory.")
    original_bytes = source.read_bytes()
    source_hash = hashlib.sha256(original_bytes).hexdigest()
    original, rate = read_pcm16(source)
    if len(original) / rate < 3 or np.sqrt(np.mean(original ** 2)) < 1e-5:
        raise ValueError("Input is too short or silent for this experiment.")
    if np.any(np.abs(original) >= 32767 / 32768):
        raise ValueError("Input already reaches full scale; inspect clipping before beautification.")
    # Validate/process in memory before creating a delivery directory.
    versions = [original, *[process_audio(source, chain) for chain in presets.values()]]
    if any(len(data) != len(original) or not np.all(np.isfinite(data)) for data in versions):
        raise ValueError("DSP changed sample count or produced non-finite audio.")
    measured = [measure_loudness(data, rate) for data in versions]
    target = min(-18.0, *[m["integrated_lufs"] - 3.1 - m["true_peak_dbtp"] for m in measured])
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "version": {"v1": "F1-DSP-v0.1", "v2": "F1-DSP-v0.2", "v3": "F1-DSP-v0.3"}[profile], "profile": profile,
        "created_at": datetime.now().astimezone().isoformat(),
        "input": str(source), "source_sha256": source_hash,
        "input_samples": sample_stats(original, rate), "input_loudness": measured[0],
        "target_lufs": target, "presets": presets,
        "ffmpeg_version": run_ffmpeg(["-version"]).stdout.decode("utf-8").splitlines()[0],
        "technical_pass": False, "listening_acceptance": "pending_jade",
        "outputs": [], "errors": [],
    }
    (out / "00_original.wav").write_bytes(original_bytes)
    for name, data, meter in zip(names, versions, measured):
        gain = target - meter["integrated_lufs"]
        export_wav(data, rate, gain, out / name)
        # Always inspect the EXPORTED PCM16 file, not just intermediate floats.
        exported, output_rate = read_pcm16(out / name)
        stats, loudness = sample_stats(exported, output_rate), measure_loudness(exported, output_rate)
        checks = compare_audio(original, exported, rate)
        if output_rate != rate:
            checks["errors"].append("Sample rate changed.")
        if loudness["true_peak_dbtp"] > -3:
            checks["errors"].append("True peak exceeds -3 dBTP headroom.")
        if abs(loudness["integrated_lufs"] - target) > 0.2:
            checks["errors"].append("Loudness differs from comparison target by >0.2 LU.")
        checks["passed"] = not checks["errors"]
        report["outputs"].append({
            "file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
            "gain_db": gain, "samples": stats, "loudness": loudness, "comparison": checks,
        })
        report["errors"].extend(f"{name}: {message}" for message in checks["errors"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        report["errors"].append("Source file changed during processing.")
    report["technical_pass"] = not report["errors"]
    (out / "qa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "试听说明与检查结果.md").write_text(make_report_markdown(report), encoding="utf-8")
    if report["technical_pass"]:
        page = make_listening_page(report)
        if profile in ("v2", "v3"):
            # Comparison montage only. The three module WAVs remain untrimmed.
            tracks = [read_pcm16(out / name)[0][rate:4 * rate] for name in names]
            gap = np.zeros(round(rate * 0.35))
            if profile == "v3":
                previous = process_audio(source, PRESETS_V2["strong"])
                meter = measure_loudness(previous, rate)
                export_wav(previous, rate, target - meter["integrated_lufs"], out / "05_previous_v2_strong.wav")
                tracks.insert(1, read_pcm16(out / "05_previous_v2_strong.wav")[0][rate:4 * rate])
            montage = np.concatenate([part for i, track in enumerate(tracks) for part in ((gap, track) if i else (track,))])
            export_wav(montage, rate, 0, out / "04_same_phrase_ABC.wav")
            page = page.replace("v0.1", "v0.2").replace("稍强处理", "强处理")
            page = page.replace("<section><h2>这一轮请判断</h2>",
                '<section><h2>同一句连续对照：原音 → 轻档 → 强档</h2>'
                '<p>每段 3 秒，间隔 0.35 秒；建议先听这条，再听上面的完整版。</p>'
                '<audio controls preload="metadata" src="04_same_phrase_ABC.wav" data-montage="true"></audio></section>'
                '<section><h2>这一轮请判断</h2>')
            # The montage has a different timeline and must start independently.
            page = page.replace("const position = active.ended ? 0 : active.currentTime;",
                "const position = active.ended || active.dataset.montage || player.dataset.montage ? 0 : active.currentTime;")
            if profile == "v3":
                page = page.replace("v0.2", "v0.3").replace("02 · F1 轻处理", "02 · 本轮增强档").replace("03 · F1 强处理", "03 · 本轮更强档")
                page = page.replace("原音 → 轻档 → 强档", "原音 → 上轮强档 → 本轮增强 → 本轮更强")
                page = page.replace('<section><h2>同一句连续对照',
                    '<section><h2>上一轮强档（基准）</h2><audio controls preload="metadata" src="05_previous_v2_strong.wav"></audio></section><section><h2>同一句连续对照')
        (out / "index.html").write_text(page, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="NEW versioned directory")
    parser.add_argument("--profile", choices=["v1", "v2", "v3"], default="v1")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.exit(1, "FFmpeg is not on PATH.\n")
    try:
        report = experiment(args.input, args.out, args.profile)
    except (ValueError, OSError, wave.Error, subprocess.SubprocessError, RuntimeError) as error:
        parser.exit(1, f"F1 failed: {error}\n")
    print(json.dumps({"technical_pass": report["technical_pass"], "out": str(args.out),
                      "errors": report["errors"], "listening_acceptance": "pending_jade"}, ensure_ascii=False))
    return 0 if report["technical_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
