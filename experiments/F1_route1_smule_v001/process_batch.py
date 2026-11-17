# 路线一 Smule 批处理与工程检查。
# 流程：原音(R kHz) -> Smule(48 kHz 原生输出) -> 重采样回 R kHz 并对齐样本数
#       -> 固定增益匹配共同响度(-18 LUFS 规则) -> 项目工程检查 -> report.json
# 响度/导出/统计/音高估计沿用项目 f1_poc.py；cents_stats 等检查逻辑与阈值
# 取自 neuralsvb_f1/run_poc_batch.py（阈值未改动）。Praat 第二意见依赖 NeuralSVB
# 运行环境，本实验隔离环境内不引入，记为 omitted。
# 运行（系统 Python，需 ffmpeg 在 PATH）：
#   python process_batch.py --run-name run_shige_v001 --ids M2
import argparse
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
sys.path.insert(0, str(ROOT))
import f1_poc as qa  # noqa: E402  项目自带测量函数，未改阈值

VENV_PY = EXP / ".venv" / "Scripts" / "python.exe"
RUNNER = EXP / "run_smule.py"
CKPT = EXP / "downloads" / "smule-renaissance-small.pt"
SMULE_RATE = 48000

# 与 neuralsvb_f1/run_poc_batch.py 相同的阈值，未放宽
MAX_DURATION_GAP_S = 0.05
TRUE_PEAK_LIMIT_DBTP = -3.0
MAX_PITCH_MEDIAN_CENTS, MAX_PITCH_P95_CENTS = 10, 35
MAX_GAIN_SPREAD_DB, MIN_GAIN_DB, MAX_JUMP_RATIO = 8, -12, 2.5

# 素材清单（G1-G2 待路线二选定后加入）
MANIFEST = {
    "M1": "test_dataset/my_singing/molihua_normal.wav",
    "M2": "test_dataset/my_singing/shige_normal.wav",
    "M3": "test_dataset/my_singing/zuixuan_normal.wav",
    "O1": "experiments/F1_neuralsvb_pitch_target_v001/official_all/Chinese/gt_a/3.wav",
    "O2": "experiments/F1_neuralsvb_pitch_target_v001/official_all/Chinese/gt_a/9.wav",
    "O3": "experiments/F1_neuralsvb_pitch_target_v001/official_all/English/gt_a/6.wav",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resample(x, rate_from, rate_to):
    g = np.gcd(rate_from, rate_to)
    return signal.resample_poly(x, rate_to // g, rate_from // g)


def cents_stats(ref, test):
    m = min(len(ref), len(test))
    ref, test = ref[:m], test[:m]
    ok = np.isfinite(ref) & np.isfinite(test) & (ref > 0) & (test > 0)
    if ok.sum() < 10:
        return {"comparable_frames": int(ok.sum()), "note": "too few confident frames"}
    cents = np.abs(1200 * np.log2(test[ok] / ref[ok]))
    return {"comparable_frames": int(ok.sum()), "median_abs_cents": float(np.median(cents)),
            "p95_abs_cents": float(np.percentile(cents, 95))}


def active_gain_stats(before, rate_b, after, rate_a):
    b, a = qa.frame_rms(before, rate_b), qa.frame_rms(after, rate_a)
    m = min(len(a), len(b))
    b, a = b[:m], a[:m]
    active = b > max(np.max(b) * 0.08, 1e-4)
    gain = 20 * np.log10(np.maximum(a[active], 1e-12) / b[active])
    gain -= np.median(gain)
    return {"spread_db": float(np.percentile(gain, 95) - np.percentile(gain, 5)),
            "min_db": float(np.min(gain))}


def jump_ratio(before, after):
    scale = np.sqrt(np.mean(after ** 2) / max(np.mean(before ** 2), 1e-20))
    return float(np.max(np.abs(np.diff(after))) / max(np.max(np.abs(np.diff(before))) * scale, 1e-9))


def common_target(measured):
    # f1_poc 规则：-18 LUFS；若有文件到 -18 时真峰值会超 -3.1 dBTP，则整体下调
    return min(-18.0, *[m["integrated_lufs"] - 3.1 - m["true_peak_dbtp"] for m in measured])


def smule_infer(source, raw_out):
    cmd = [str(VENV_PY), "-I", str(RUNNER), str(source), "-o", str(raw_out), "-c", str(CKPT)]
    result = subprocess.run(cmd, capture_output=True, cwd=str(EXP))
    if result.returncode:
        raise RuntimeError("Smule 推理失败:\n" + result.stderr.decode("utf-8", "replace"))
    if not raw_out.exists():
        raise RuntimeError("Smule 未产出文件: " + str(raw_out))


def process_clip(clip_id, source, run_dir):
    clip_dir = run_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    raw48 = clip_dir / (clip_id + "_smule_native48k.wav")

    orig, rate = qa.read_pcm16(source)  # 断言单声道 PCM16，否则报错先查原因
    smule_infer(source, raw48)

    raw_rate, raw = wavfile.read(raw48)
    assert raw_rate == SMULE_RATE, f"原生输出采样率异常: {raw_rate}"
    raw = raw.astype(np.float64)
    if raw.ndim > 1:
        raise ValueError("Smule 输出不是单声道，先查原因")

    # 重采样回原音采样率，并对齐到与原音完全相同的样本数
    back = resample(raw, SMULE_RATE, rate)
    pad_needed = len(orig) - len(back)
    truncated = 0
    if pad_needed > 0:
        back = np.pad(back, (0, pad_needed))
    elif pad_needed < 0:
        truncated = -pad_needed
        back = back[:len(orig)]
    if pad_needed > rate * MAX_DURATION_GAP_S:
        raise RuntimeError(f"输出比原音短 {pad_needed / rate:.3f}s，超过 {MAX_DURATION_GAP_S}s，先查原因")
    np.save(clip_dir / (clip_id + "_resampled.npy"), back)

    # 共同响度目标（原音与处理版同一规则，各自一次性固定增益）
    loud_orig = qa.measure_loudness(orig, rate)
    loud_back = qa.measure_loudness(back, rate)
    target = common_target([loud_orig, loud_back])
    paths = {}
    for name, data, meas in (("original_matched", orig, loud_orig), ("smule_matched", back, loud_back)):
        p = clip_dir / f"{clip_id}__{name}.wav"
        qa.export_wav(data, rate, target - meas["integrated_lufs"], p)
        paths[name] = p

    # 工程检查（读取实际导出文件，即 Jade 将听到的内容）
    report = {"clip": clip_id, "source": str(source), "source_sha256": sha(source),
              "sample_rate": rate, "target_lufs": target,
              "native_smule_sha256": sha(raw48),
              "resampled_pad_samples": int(max(pad_needed, 0)),
              "resampled_truncated_samples": int(truncated),
              "listening_status": "pending_jade",
              "thresholds": {"pitch_median_cents": MAX_PITCH_MEDIAN_CENTS,
                             "pitch_p95_cents": MAX_PITCH_P95_CENTS,
                             "true_peak_dbtp": TRUE_PEAK_LIMIT_DBTP,
                             "loudness_diff_lu": 0.2,
                             "gain_spread_db": MAX_GAIN_SPREAD_DB,
                             "min_gain_db": MIN_GAIN_DB,
                             "jump_ratio": MAX_JUMP_RATIO},
              "praat_second_opinion": "omitted: NeuralSVB 专属依赖，不在本隔离环境引入；音高用项目估计器单源检查",
              "outputs": {}}

    actuals = {k: qa.read_pcm16(v) for k, v in paths.items()}
    ref, _ = actuals["original_matched"]
    loud_ref = qa.measure_loudness(ref, rate)
    for name, path in paths.items():
        value, out_rate = actuals[name]
        with wave.open(str(path), "rb") as stream:
            spec = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                    stream.getnframes(), stream.getcomptype())
            pcm = stream.readframes(stream.getnframes())
        item = {"file": path.name, "sha256": sha(path), "samples": qa.sample_stats(value, out_rate)}
        errors = []
        if spec != (rate, 1, 2, len(orig), "NONE"):
            errors.append(f"WAV 规格不符: {spec}")
        decoded = qa.run_ffmpeg(["-v", "error", "-nostdin", "-xerror", "-i", str(path),
                                 "-f", "s16le", "pipe:1"]).stdout
        item["full_decode_pass"] = decoded == pcm
        if not item["full_decode_pass"]:
            errors.append("完整解码失败")
        loud = qa.measure_loudness(value, out_rate)
        item["loudness"] = loud
        if not np.all(np.isfinite(value)) or np.max(np.abs(value)) < 1e-5:
            errors.append("non-finite or silent output")
        if item["samples"]["clipped_samples"]:
            errors.append("clipped samples")
        if loud["true_peak_dbtp"] > TRUE_PEAK_LIMIT_DBTP:
            errors.append("true peak above -3 dBTP")
        if abs(loud["integrated_lufs"] - loud_ref["integrated_lufs"]) > 0.2:
            errors.append("loudness differs by >0.2 LU")
        if name != "original_matched":
            gain = active_gain_stats(ref, rate, value, out_rate)
            item["active_gain"] = gain
            if gain["spread_db"] > MAX_GAIN_SPREAD_DB or gain["min_db"] < MIN_GAIN_DB:
                errors.append("dynamic change / dropout in active singing")
            jr = jump_ratio(ref, value)
            item["max_sample_jump_ratio"] = jr
            if jr > MAX_JUMP_RATIO:
                errors.append("large new sample discontinuity (possible click)")
            pc = cents_stats(qa.estimate_pitch(ref, rate), qa.estimate_pitch(value, out_rate))
            item["pitch_vs_original"] = pc
            if "median_abs_cents" not in pc:
                errors.append("pitch preservation not verified")
            elif (pc["median_abs_cents"] > MAX_PITCH_MEDIAN_CENTS
                  or pc["p95_abs_cents"] > MAX_PITCH_P95_CENTS):
                errors.append("pitch diagnostic changed (manual check required)")
        item["technical_pass"] = not errors
        item["errors"] = errors
        report["outputs"][name] = item

    (clip_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ids", nargs="+", required=True, choices=sorted(MANIFEST))
    args = ap.parse_args()
    run_dir = EXP / args.run_name
    if run_dir.exists():
        raise SystemExit("运行目录已存在，禁止覆盖；请用新 run-name。")
    run_dir.mkdir(parents=True)
    summary = []
    for clip_id in args.ids:
        source = ROOT / MANIFEST[clip_id]
        if not source.exists():
            raise SystemExit(f"素材不存在: {source}")
        print(f"[{clip_id}] 处理 {source}", flush=True)
        report = process_clip(clip_id, source, run_dir)
        out = report["outputs"]["smule_matched"]
        print(f"[{clip_id}] technical_pass={out['technical_pass']} errors={out['errors']}", flush=True)
        summary.append(report)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    n_pass = sum(r["outputs"]["smule_matched"]["technical_pass"] for r in summary)
    print(f"完成 {len(summary)} 条，technical_pass {n_pass} 条；听感状态一律 pending_jade")


if __name__ == "__main__":
    main()
