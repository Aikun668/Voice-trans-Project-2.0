"""F2 Phase-1 PoC: 验证 Parselmouth/Praat 音高修改最小闭环。

输入: 单声道 PCM16 WAV 清唱
输出(新目录, 拒绝覆盖):
  00_original_untouched.wav  原音位精确副本(未调电平)
  01_original_matched.wav    原音, 响度匹配版本(公平对照基准)
  02_resynth_passthrough.wav Manipulation 分析->重合成, 不做任何修改(检验PSOLA链路本身)
  03_shift_plus{N}c.wav      整体上移 N 音分 (PitchTier 均匀乘法)
  04_shift_minus{N}c.wav    整体下移 N 音分
  qa_report.json             工程检查报告

技术路线: Praat Manipulation(TD-PSOLA) -> Multiply frequencies -> Replace pitch tier
         -> Get resynthesis (overlap-add)。时长与采样率保持不变。
本阶段不含任何自动修音决策。听感验收由人完成, 技术通过仅记 technical_pass。

已知特性: TD-PSOLA 重合成会重置相位, 与原音的零滞后波形相关系数接近 0 属正常,
音色保留由谱包络体现, 故以频谱带能量差异作为客观音色检查, 波形相关仅作记录。
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import parselmouth

PITCH_FLOOR = 75.0
PITCH_CEIL = 600.0
TIME_STEP = 0.01
PEAK_HEADROOM_DB = -3.0
BANDS_HZ = [(0, 300), (300, 1000), (1000, 3500), (3500, 8000), (8000, 22050)]
BAND_NAMES = ["0-300", "300-1k", "1k-3.5k", "3.5k-8k", "8k-22k"]


def read_wav_pcm16(path: Path):
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if nch != 1:
        raise SystemExit(f"仅支持单声道输入, 实际 {nch} 声道: {path}")
    if sw != 2:
        raise SystemExit(f"仅支持 16-bit PCM 输入, 实际 {sw * 8} bit: {path}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return x, sr


def write_wav_pcm16(path: Path, x: np.ndarray, sr: int):
    y = np.clip(x, -1.0, 1.0)
    yi = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(yi.tobytes())


def rms_db(x):
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-12))


def peak_db(x):
    return 20.0 * math.log10(max(float(np.max(np.abs(x))), 1e-12))


def clip_count(x):
    return int(np.sum(np.abs(x) >= 0.999))


def sha256_of(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def f0_stats(x: np.ndarray, sr: int):
    """Praat 标准自相关 F0 分析, 返回有声帧均值/中位数。"""
    s = parselmouth.Sound(x, sr)
    p = parselmouth.praat.call(s, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEIL)
    mean_f0 = parselmouth.praat.call(p, "Get mean", 0.0, 0.0, "Hertz")
    median_f0 = parselmouth.praat.call(p, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    return {"mean_f0_hz": mean_f0, "median_f0_hz": median_f0}


def corr(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float((a @ b) / d) if d > 0 else 0.0


def band_energies_db(x: np.ndarray, sr: int):
    """numpy Welch 平均功率谱 -> 频带能量(dB), 客观音色指标。"""
    n_fft = 4096
    hop = n_fft // 2
    if len(x) < n_fft:
        n_fft = max(1024, 1 << int(math.log2(len(x))))
        hop = n_fft // 2
    win = np.hanning(n_fft)
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    acc = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
    acc /= n_frames
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    out = []
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        e = float(np.sum(acc[m])) if m.any() else 1e-24
        out.append(10.0 * math.log10(max(e, 1e-24)))
    return out


def to_array(parselmouth_sound):
    return np.asarray(parselmouth_sound.values[0], dtype=np.float64)


def resynth_passthrough(sound):
    manip = parselmouth.praat.call(sound, "To Manipulation", TIME_STEP, PITCH_FLOOR, PITCH_CEIL)
    return parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")


def shift_uniform(sound, cents):
    """对整段做均匀移调(音分), 保持时长。每档位用全新 Manipulation, 避免档位间串扰。"""
    manip = parselmouth.praat.call(sound, "To Manipulation", TIME_STEP, PITCH_FLOOR, PITCH_CEIL)
    tier = parselmouth.praat.call(manip, "Extract pitch tier")
    factor = 2.0 ** (cents / 1200.0)
    dur = sound.get_total_duration()
    parselmouth.praat.call(tier, "Multiply frequencies", 0.0, dur, factor)
    parselmouth.praat.call([manip, tier], "Replace pitch tier")
    return parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="输入 WAV (单声道 PCM16)")
    ap.add_argument("--output", required=True, help="输出目录 (必须不存在)")
    ap.add_argument("--cents", type=float, default=50.0, help="移调幅度(音分), 默认 50")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if not in_path.is_file():
        raise SystemExit(f"输入不存在: {in_path}")
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out_dir}")

    x_in, sr = read_wav_pcm16(in_path)
    n_in = len(x_in)
    print(f"输入: {in_path}  sr={sr}  样本={n_in}  时长={n_in / sr:.3f}s")

    sound = parselmouth.Sound(x_in, sr)
    shifted = {
        "passthrough": to_array(resynth_passthrough(sound)),
        "plus": to_array(shift_uniform(sound, args.cents)),
        "minus": to_array(shift_uniform(sound, -args.cents)),
    }

    # 共同响度目标 = 原音 RMS; 所有匹配版本共用, 若峰值超 -3dB 余量则整体再统一降
    target_rms = rms_db(x_in)
    matched = {"01_original_matched.wav": x_in.copy(),
               "02_resynth_passthrough.wav": shifted["passthrough"],
               f"03_shift_plus{args.cents:g}c.wav": shifted["plus"],
               f"04_shift_minus{args.cents:g}c.wav": shifted["minus"]}
    gains_db = {f: target_rms - rms_db(x) for f, x in matched.items()}
    peak_after = max(peak_db(x * 10 ** (gains_db[f] / 20.0)) for f, x in matched.items())
    common_trim_db = 0.0
    if peak_after > PEAK_HEADROOM_DB:
        common_trim_db = PEAK_HEADROOM_DB - peak_after
    gains_db = {f: g + common_trim_db for f, g in gains_db.items()}
    print(f"响度策略: 共同目标RMS={target_rms:.2f}dBFS, 统一峰值余量修正={common_trim_db:.2f}dB")

    out_dir.mkdir(parents=True)
    shutil.copyfile(in_path, out_dir / "00_original_untouched.wav")
    for fname, x in matched.items():
        write_wav_pcm16(out_dir / fname, x * 10 ** (gains_db[fname] / 20.0), sr)

    # ---- 工程检查: 全部基于导出后重新读取的实际文件 ----
    files = ["00_original_untouched.wav"] + list(matched.keys())
    f0_ref = f0_stats(x_in, sr)
    # 带能量对比基准用响度匹配版(01), 排除统一电平衰减, 只看谱形差异
    bands_ref = band_energies_db(read_wav_pcm16(out_dir / "01_original_matched.wav")[0], sr)
    report = {
        "phase": "F2-Phase1 minimal pitch-shift loop",
        "tool": "praat-parselmouth",
        "parselmouth_version": parselmouth.__version__,
        "input": {
            "path": str(in_path),
            "sha256": sha256_of(in_path),
            "sample_rate": sr,
            "samples": n_in,
            "duration_s": round(n_in / sr, 4),
            "rms_dbfs": round(target_rms, 2),
            "peak_dbfs": round(peak_db(x_in), 2),
            "f0_mean_hz": round(f0_ref["mean_f0_hz"], 2),
            "f0_median_hz": round(f0_ref["median_f0_hz"], 2),
        },
        "params": {"cents": args.cents, "pitch_floor_hz": PITCH_FLOOR, "pitch_ceiling_hz": PITCH_CEIL,
                   "manipulation_time_step_s": TIME_STEP, "common_trim_db": round(common_trim_db, 2)},
        "bands_hz": BAND_NAMES,
        "versions": {},
        "gates": {},
    }

    for fname in files:
        x_chk, sr_chk = read_wav_pcm16(out_dir / fname)
        f0 = f0_stats(x_chk, sr_chk)
        bands = band_energies_db(x_chk, sr_chk)
        info = {
            "samples": len(x_chk),
            "sample_rate": sr_chk,
            "duration_dev_ms": round((len(x_chk) - n_in) / sr * 1000.0, 2),
            "rms_dbfs": round(rms_db(x_chk), 2),
            "peak_dbfs": round(peak_db(x_chk), 2),
            "clipped_samples": clip_count(x_chk),
            "f0_mean_hz": round(f0["mean_f0_hz"], 2),
            "f0_median_hz": round(f0["median_f0_hz"], 2),
            "f0_shift_cents_vs_original": round(1200.0 * math.log2(f0["median_f0_hz"] / f0_ref["median_f0_hz"]), 2),
            "band_energies_db": [round(b, 2) for b in bands],
            "band_diff_vs_matched_original_db": [round(b - r, 2) for b, r in zip(bands, bands_ref)],
            "corr_with_original_note": "TD-PSOLA重置相位, 波形相关接近0属正常, 不作门槛",
            "corr_with_original": round(corr(x_in, x_chk), 4),
        }
        report["versions"][fname] = info
        print(f"{fname}: dur_dev={info['duration_dev_ms']}ms rms={info['rms_dbfs']}dB peak={info['peak_dbfs']}dB "
              f"clip={info['clipped_samples']} f0_shift={info['f0_shift_cents_vs_original']}c "
              f"band_diff={info['band_diff_vs_matched_original_db']}")

    # ---- 工程门槛 ----
    v = report["versions"]
    plus_f = f"03_shift_plus{args.cents:g}c.wav"
    minus_f = f"04_shift_minus{args.cents:g}c.wav"
    gates = {
        "all_same_sample_rate": all(i["sample_rate"] == sr for i in v.values()),
        "duration_dev_within_10ms": all(i["duration_dev_ms"] <= 10.0 for i in v.values()),
        "no_clipping": all(i["clipped_samples"] == 0 for i in v.values()),
        "matched_versions_equal_rms_within_0.5dB": max(i["rms_dbfs"] for f, i in v.items() if f != "00_original_untouched.wav")
        - min(i["rms_dbfs"] for f, i in v.items() if f != "00_original_untouched.wav") <= 0.5,
        "passthrough_f0_within_5c": abs(v["02_resynth_passthrough.wav"]["f0_shift_cents_vs_original"]) <= 5.0,
        "passthrough_band_diff_within_2dB": max(abs(d) for d in v["02_resynth_passthrough.wav"]["band_diff_vs_matched_original_db"]) <= 2.0,
        "shift_plus_within_10c": abs(v[plus_f]["f0_shift_cents_vs_original"] - args.cents) <= 10.0,
        "shift_minus_within_10c": abs(v[minus_f]["f0_shift_cents_vs_original"] + args.cents) <= 10.0,
    }
    report["gates"] = gates
    report["technical_pass"] = all(gates.values())
    report["listening"] = "pending_human (听感验收由用户完成, 重点听02/03/04与01的音色差异)"

    with open(out_dir / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n工程门槛:")
    for k, val in gates.items():
        print(f"  {'PASS' if val else 'FAIL'}  {k}")
    verdict = "technical_pass" if report["technical_pass"] else "TECHNICAL_FAIL"
    print(f"\n结论: {verdict} (听感 pending_human)")
    print(f"输出目录: {out_dir}")
    sys.exit(0 if report["technical_pass"] else 1)


if __name__ == "__main__":
    main()
