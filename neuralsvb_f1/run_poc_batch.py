"""F1 NeuralSVB PoC batch: original WAV -> {vocoder_only, a2a, a2p_nomap, a2p} + engineering checks.

Variants separate "model effect" from "vocoder re-synthesis effect":
  vocoder_only : original mel -> HiFi-GAN-Singing (no acoustic model, no latent mapping)
  a2a          : amateur-latent VAE reconstruction (acoustic model, NO mapping)
  a2p_nomap    : a2p graph with the Latent Mapping bypassed (disable_map=True)
  a2p          : full NeuralSVB beautification (candidate)
Loudness: original and all variants get ONE constant gain each to a common target (same rule as
f1_poc.experiment), so A/B is loudness matched and hot inputs keep -3 dBTP headroom.
Every run needs a NEW output directory. Listening status is always pending_jade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import f1_poc as qa  # noqa: E402  (project's own measurement helpers; no thresholds changed)
import neuralsvb_infer as nsvb  # noqa: E402

VARIANTS = ("vocoder_only", "a2a", "a2p_nomap", "a2p")
MODE_OF = {"vocoder_only": "vocoder", "a2a": "a2a", "a2p_nomap": "a2p_nomap", "a2p": "a2p"}
CANDIDATES = ("a2p", "vocoder_only")  # go to candidate/; the rest are diagnostics/
MAX_DURATION_GAP_S = 0.05  # model works on 4-frame groups (4*128/22050 = 23 ms); tail may be cut
TRUE_PEAK_LIMIT_DBTP = -3.0  # same headroom rule as f1_poc.experiment
MAX_PITCH_MEDIAN_CENTS, MAX_PITCH_P95_CENTS = 10, 35  # same as f1_poc.compare_audio
MAX_GAIN_SPREAD_DB, MIN_GAIN_DB, MAX_JUMP_RATIO = 8, -12, 2.5  # same as f1_poc.compare_audio


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample(x: np.ndarray, rate_from: int, rate_to: int) -> np.ndarray:
    from scipy import signal

    g = math.gcd(rate_from, rate_to)
    return signal.resample_poly(x, rate_to // g, rate_from // g)


def cents_stats(ref: np.ndarray, test: np.ndarray) -> dict:
    m = min(len(ref), len(test))
    ref, test = ref[:m], test[:m]
    ok = np.isfinite(ref) & np.isfinite(test) & (ref > 0) & (test > 0)
    if ok.sum() < 10:
        return {"comparable_frames": int(ok.sum()), "note": "too few confident frames"}
    cents = np.abs(1200 * np.log2(test[ok] / ref[ok]))
    return {"comparable_frames": int(ok.sum()), "median_abs_cents": float(np.median(cents)),
            "p95_abs_cents": float(np.percentile(cents, 95))}


def active_gain_stats(before, rate_b, after, rate_a) -> dict:
    """Level change per 20 ms RMS frame inside active singing, median removed."""
    b, a = qa.frame_rms(before, rate_b), qa.frame_rms(after, rate_a)
    m = min(len(a), len(b))
    b, a = b[:m], a[:m]
    active = b > max(np.max(b) * 0.08, 1e-4)
    gain = 20 * np.log10(np.maximum(a[active], 1e-12) / b[active])
    gain -= np.median(gain)
    return {"spread_db": float(np.percentile(gain, 95) - np.percentile(gain, 5)), "min_db": float(np.min(gain))}


def jump_ratio(before: np.ndarray, after: np.ndarray) -> float:
    scale = np.sqrt(np.mean(after ** 2) / max(np.mean(before ** 2), 1e-20))
    return float(np.max(np.abs(np.diff(after))) / max(np.max(np.abs(np.diff(before))) * scale, 1e-9))


def check_output(engine, orig, rate_o, loud_o, out, rate_out) -> dict:
    """All inputs are the loudness-matched PCM16 files as exported (what Jade would hear)."""
    errors = []
    loud = qa.measure_loudness(out, rate_out)
    stats = qa.sample_stats(out, rate_out)
    if not np.all(np.isfinite(out)) or np.max(np.abs(out)) < 1e-5:
        errors.append("non-finite or silent output")
    if stats["clipped_samples"]:
        errors.append("clipped samples")
    gap = abs(len(orig) / rate_o - len(out) / rate_out)
    if gap > MAX_DURATION_GAP_S:
        errors.append(f"duration gap {gap:.3f}s")
    if loud["true_peak_dbtp"] > TRUE_PEAK_LIMIT_DBTP:
        errors.append("true peak above -3 dBTP")
    if abs(loud["integrated_lufs"] - loud_o["integrated_lufs"]) > 0.2:
        errors.append("loudness differs by >0.2 LU")
    gain = active_gain_stats(orig, rate_o, out, rate_out)
    if gain["spread_db"] > MAX_GAIN_SPREAD_DB or gain["min_db"] < MIN_GAIN_DB:
        errors.append("dynamic change / dropout in active singing")
    jr = jump_ratio(resample(orig, rate_o, rate_out), out)
    if jr > MAX_JUMP_RATIO:
        errors.append("large new sample discontinuity (possible click)")
    pc = cents_stats(qa.estimate_pitch(orig, rate_o), qa.estimate_pitch(out, rate_out))
    if "median_abs_cents" not in pc:
        errors.append("pitch preservation not verified")
    elif pc["median_abs_cents"] > MAX_PITCH_MEDIAN_CENTS or pc["p95_abs_cents"] > MAX_PITCH_P95_CENTS:
        errors.append("pitch diagnostic changed (manual check required)")
    # Second opinion (informational): the model's own Praat extractor on both signals.
    o22 = resample(orig, rate_o, nsvb.SAMPLE_RATE)
    praat = cents_stats(nsvb.praat_f0(o22, engine.hp), nsvb.praat_f0(out, engine.hp))
    return {
        "technical_pass": not errors, "errors": errors, "duration_gap_s": gap,
        "samples": stats, "loudness": loud, "active_gain": gain, "max_sample_jump_ratio": jr,
        "pitch_vs_original": pc, "pitch_vs_original_praat_info_only": praat,
        "not_applicable": {
            "waveform_correlation": "vocoder re-synthesis does not preserve phase/waveform",
            "band_change_limit": "re-synthesis legitimately reshapes spectrum; not gated (see report)",
            "sample_count_equal": "output is 22.05 kHz (official model rate); duration gap checked instead",
        },
    }


def common_target(measured: list[dict]) -> float:
    """f1_poc.experiment rule: -18 LUFS, or lower if any file would exceed -3.1 dBTP at that level."""
    return min(-18.0, *[m["integrated_lufs"] - 3.1 - m["true_peak_dbtp"] for m in measured])


def run_clip(engine, clip: Path, out: Path, seed: int) -> dict:
    orig, rate_o = qa.read_pcm16(clip)
    raws = {}
    for name in VARIANTS:
        wav, f0_in, f0_voc = engine.run(str(clip), mode=MODE_OF[name], seed=seed)
        raws[name] = (wav.astype(np.float64), f0_in, f0_voc)
    measured = {"original": qa.measure_loudness(orig, rate_o)}
    measured.update({k: qa.measure_loudness(v[0], nsvb.SAMPLE_RATE) for k, v in raws.items()})
    target = common_target(list(measured.values()))
    (out / "candidate").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    orig_path = out / "candidate" / f"{clip.stem}__original.wav"
    qa.export_wav(orig, rate_o, target - measured["original"]["integrated_lufs"], orig_path)
    orig_m, _ = qa.read_pcm16(orig_path)
    loud_o = qa.measure_loudness(orig_m, rate_o)
    record = {"input": str(clip), "input_sha256": sha256(clip), "input_samples": qa.sample_stats(orig, rate_o),
              "input_loudness": measured["original"], "common_target_lufs": target,
              "original_matched_copy": {"file": str(orig_path.relative_to(out)), "sha256": sha256(orig_path)},
              "variants": {}}
    for name, (raw, f0_in, f0_voc) in raws.items():
        gain_db = target - measured[name]["integrated_lufs"]
        folder = "candidate" if name in CANDIDATES else "diagnostics"
        path = out / folder / f"{clip.stem}__{name}.wav"
        qa.export_wav(raw, nsvb.SAMPLE_RATE, gain_db, path)
        out_m, rate_out = qa.read_pcm16(path)
        voiced_in, voiced_voc = f0_in[f0_in > 0], f0_voc[f0_voc > 0]
        checks = check_output(engine, orig_m, rate_o, loud_o, out_m, rate_out)
        record["variants"][name] = {
            "file": str(path.relative_to(out)), "sha256": sha256(path), "gain_db": gain_db,
            "raw_peak_abs": float(np.max(np.abs(raw))), "raw_samples_over_fullscale": int(np.sum(np.abs(raw) > 1)),
            "f0_modified": bool(np.max(np.abs(f0_in - f0_voc)) > 1e-3),
            "f0_median_hz": [float(np.median(voiced_in)), float(np.median(voiced_voc))],
            "checks": checks,
        }
        print(f"{clip.stem} {name}: technical_pass={checks['technical_pass']} {checks['errors']}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="NEW experiment directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("clips", nargs="+")
    args = ap.parse_args()
    out = Path(args.out).resolve()  # engine chdirs into the official repo -> use absolute paths
    clips = [Path(c).resolve() for c in args.clips]
    if out.exists():
        sys.exit(f"{out} already exists; use a new directory (never overwrite experiments)")
    out.mkdir(parents=True)
    engine = nsvb.NeuralSVB(f0_norm=None)
    report = {
        "model": "NeuralSVB MleSVBVAE (1030_vae_mle) + HiFi-GAN singing NSF (1012) + official ASR/PPG encoder (1009)",
        "ckpt_step": engine.global_step, "seed": args.seed, "listening_status": "pending_jade",
        "f0_policy": "identity: model pitch input and vocoder F0 are the input's own contour",
        "clips": [run_clip(engine, c, out, args.seed) for c in clips],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
