"""Verify formant_shift.py reproduces F1-EXP-020 S45.wav from its EXP-016 M2 base.

Loads the base and the reference with f1_poc.read_pcm16 (the same loader run.py
uses), runs the extracted shift at +45 Hz, and compares against S45.wav.
The reference was written with export_wav (float32 -> PCM16 via ffmpeg), so the
primary comparison quantizes the float output the same way before comparing.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from f1_poc import read_pcm16  # noqa: E402
from formant_shift import apply_shift, shift_f1  # noqa: E402

BASE = ROOT / "experiments/F1-EXP-016/outputs/processed_raw/M2.wav"
BASE_QA = ROOT / "experiments/F1-EXP-016/qa/qa_report.json"
REF = ROOT / "experiments/F1-EXP-020/outputs/processed_raw/S45.wav"
REF_QA = ROOT / "experiments/F1-EXP-020/qa/qa_report.json"
SHIFT_HZ = 45.0
THRESH_DB = -60.0


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def metrics(y, ref):
    n = min(len(y), len(ref))
    diff = y[:n] - ref[:n]
    rms_ref = float(np.sqrt(np.mean(ref[:n] ** 2)))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    return {
        "len_out": int(len(y)),
        "len_ref": int(len(ref)),
        "len_match": len(y) == len(ref),
        "max_abs_diff_lsb": float(np.max(np.abs(diff)) * 32768.0),
        "correlation": float(np.corrcoef(y[:n], ref[:n])[0, 1]),
        "diff_db": float(20.0 * np.log10(max(rms_diff, 1e-30) / rms_ref)),
    }


def quantize_pcm16(y):
    """Mimic export_wav: float32 cast, then round-to-nearest PCM16 with clipping."""
    q = np.round(y.astype(np.float32).astype(np.float64) * 32768.0)
    return np.clip(q, -32768, 32767) / 32768.0


def main():
    base_sha = next(v for v in json.loads(BASE_QA.read_text(encoding="utf-8"))["versions"]
                    if v["id"] == "M2")["raw_sha256"]
    if sha(BASE) != base_sha:
        raise SystemExit("baseline M2.wav hash mismatch")

    x, sr = read_pcm16(BASE)
    ref, sr_ref = read_pcm16(REF)
    if sr != sr_ref:
        raise SystemExit(f"sample rate mismatch {sr} vs {sr_ref}")

    y, stats = apply_shift(x, sr, SHIFT_HZ)
    y_wrap = shift_f1(x, sr, SHIFT_HZ)
    if not np.array_equal(y, y_wrap):
        raise SystemExit("shift_f1 differs from apply_shift")

    ref_stats = next(v for v in json.loads(REF_QA.read_text(encoding="utf-8"))["versions"]
                     if v["id"] == "S45")["proc"]
    stats_match = stats == ref_stats

    float_m = metrics(y, ref)
    quant_m = metrics(quantize_pcm16(y), ref)
    print(json.dumps({"sr": sr, "input_samples": int(len(x)), "stats_match_qa": stats_match,
                      "stats": stats, "float_vs_pcm16_ref": float_m,
                      "quantized_vs_pcm16_ref": quant_m}, ensure_ascii=False, indent=1))

    ok = (stats_match and quant_m["len_match"] and float_m["len_match"]
          and quant_m["diff_db"] < THRESH_DB and float_m["diff_db"] < THRESH_DB)
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
