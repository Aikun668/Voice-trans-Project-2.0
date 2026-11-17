"""路线二（原波形类）素材客观测量：为问题登记提供证据，不做任何处理。

只测量，不修改音频。复用项目 f1_poc.py 的响度/频带/分帧函数；
对非 PCM16 输入（如 24-bit/stereo）先经 ffmpeg 解码为单声道 f32。
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import f1_poc  # noqa: E402


def ensure_ffmpeg_on_path():
    """当前会话 PATH 可能没有 ffmpeg（WinGet 安装）；找到后注入 PATH。"""
    if shutil.which("ffmpeg"):
        return
    links = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
        return
    raise SystemExit("ffmpeg.exe not found; add its directory to PATH.")


ensure_ffmpeg_on_path()

FILES = {
    "M1_molihua_normal": ROOT / "test_dataset/my_singing/molihua_normal.wav",
    "M2_shige_normal": ROOT / "test_dataset/my_singing/shige_normal.wav",
    "M3_zuixuan_normal": ROOT / "test_dataset/my_singing/zuixuan_normal.wav",
    "O1_chinese_gt_a_3": ROOT / "experiments/F1_neuralsvb_pitch_target_v001/official_all/Chinese/gt_a/3.wav",
    "O2_chinese_gt_a_9": ROOT / "experiments/F1_neuralsvb_pitch_target_v001/official_all/Chinese/gt_a/9.wav",
    "O3_english_gt_a_6": ROOT / "experiments/F1_neuralsvb_pitch_target_v001/official_all/English/gt_a/6.wav",
    "G1_qinghuaci": ROOT / "test_dataset/web_singing/gtsinger_alto_control/青花瓷_normal.wav",
    "G2_beijiaer": ROOT / "test_dataset/web_singing/gtsinger_alto_control/贝加尔湖畔_normal.wav",
}

OUT_DIR = Path(__file__).resolve().parent


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,bits_per_sample,duration",
         "-of", "json", str(path)],
        capture_output=True, timeout=60)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bits_per_sample": int(stream["bits_per_sample"] or 0),
        "container_duration_s": float(stream["duration"] or 0),
    }


def decode_mono_f32(path):
    """解码为单声道 f32，保持原采样率；仅用于测量。"""
    result = f1_poc.run_ffmpeg([
        "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-ac", "1", "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
    ])
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)


def dynamics(data, rate, frame_s=0.05):
    """活跃段的帧 RMS 分布，用于评估大小声/动态。"""
    rms = f1_poc.frame_rms(data, rate, frame_s)
    rms_db = np.array([f1_poc.db(v) for v in rms])
    active = rms_db[rms_db > np.max(rms_db) - 35]  # 排除接近静音的帧
    return {
        "frame_rms_p95_dbfs": float(np.percentile(active, 95)),
        "frame_rms_p50_dbfs": float(np.percentile(active, 50)),
        "frame_rms_p10_dbfs": float(np.percentile(active, 10)),
        "active_spread_p95_p10_db": float(np.percentile(active, 95) - np.percentile(active, 10)),
        "sample_peak_dbfs": f1_poc.db(np.max(np.abs(data))),
        "crest_factor_db": f1_poc.db(np.max(np.abs(data))) - f1_poc.db(np.sqrt(np.mean(data ** 2))),
    }


def spectral(data, rate):
    """频带占比 + 谱倾斜（亮度/单薄/刺耳的粗略证据）。"""
    bands = f1_poc.band_levels(data, rate)
    freqs, density = signal.welch(data, rate, nperseg=min(8192, len(data)))
    mask = (freqs >= 300) & (freqs <= 6000)
    slope, _ = np.polyfit(np.log2(freqs[mask]), 10 * np.log10(np.maximum(density[mask], 1e-20)), 1)
    nyq = rate / 2
    hi_top = min(10000, nyq - 100)
    total = np.sum(density[(freqs >= 80) & (freqs <= 8000)])
    sib_ratio = float(10 * np.log10(max(np.sum(density[(freqs >= 5000) & (freqs < hi_top)]), 1e-20) / max(total, 1e-20)))
    return {
        "band_levels_db": bands,
        "sibilance_band_5k_plus_ratio_db": sib_ratio,
        "tilt_db_per_octave_300_6k": float(slope),
    }


def sibilance_frames(data, rate, frame_s=0.032, hop_s=0.010):
    """定位齿音帧：5 kHz 以上能量相对 0.3–4 kHz 明显偏高的帧。"""
    size = round(rate * frame_s)
    hop = round(rate * hop_s)
    hi_band, mid_band, times = [], [], []
    window = np.hanning(size)
    for start in range(0, len(data) - size + 1, hop):
        frame = data[start:start + size] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        freqs = np.fft.rfftfreq(size, 1 / rate)
        hi_top = min(10000, rate / 2 - 100)
        hi = np.sum(spectrum[(freqs >= 5000) & (freqs < hi_top)])
        mid = np.sum(spectrum[(freqs >= 300) & (freqs < 4000)])
        hi_band.append(10 * np.log10(max(hi, 1e-20) / max(mid, 1e-20)))
        mid_band.append(mid)
        times.append(start / rate)
    ratio = np.array(hi_band)
    # 阈值：高频只比中频低 6 dB 以内视为齿音帧（元音通常低 15 dB 以上）
    flags = ratio > -6
    segments, start_idx = [], None
    for i, flag in enumerate(flags):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            if (i - start_idx) * hop_s >= 0.04:  # 至少 40 ms 才算
                seg_ratio = ratio[start_idx:i]
                segments.append({
                    "start_s": round(times[start_idx], 2),
                    "end_s": round(times[i], 2),
                    "max_hi_mid_ratio_db": round(float(np.max(seg_ratio)), 1),
                })
            start_idx = None
    if start_idx is not None and (len(flags) - start_idx) * hop_s >= 0.04:
        segments.append({
            "start_s": round(times[start_idx], 2),
            "end_s": round(times[-1], 2),
            "max_hi_mid_ratio_db": round(float(np.max(ratio[start_idx:])), 1),
        })
    segments.sort(key=lambda s: -s["max_hi_mid_ratio_db"])
    return {
        "sibilance_frame_ratio": round(float(np.mean(flags)), 4),
        "candidate_segments_top5": segments[:5],
        "candidate_segment_count": len(segments),
    }


def main():
    report = {}
    for key, path in FILES.items():
        if not path.exists():
            report[key] = {"error": f"missing file: {path}"}
            continue
        info = probe(path)
        data = decode_mono_f32(path)
        if len(data) < info["sample_rate"] * 2:
            report[key] = {"error": "decoded audio shorter than 2 s"}
            continue
        loud = f1_poc.measure_loudness(data, info["sample_rate"])
        entry = {
            "file": str(path),
            "format": info,
            "decoded_seconds": round(len(data) / info["sample_rate"], 3),
            "loudness": loud,
            "dynamics": dynamics(data, info["sample_rate"]),
            "spectral": spectral(data, info["sample_rate"]),
            "sibilance": sibilance_frames(data, info["sample_rate"]),
        }
        report[key] = entry
        print(f"[ok] {key}: {entry['decoded_seconds']}s, {loud['integrated_lufs']:.1f} LUFS, "
              f"spread {entry['dynamics']['active_spread_p95_p10_db']:.1f} dB, "
              f"sib_ratio {entry['spectral']['sibilance_band_5k_plus_ratio_db']:.1f} dB")
    (OUT_DIR / "measurements.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("saved ->", OUT_DIR / "measurements.json")


if __name__ == "__main__":
    main()
