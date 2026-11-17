"""Run the one-note round-trip check from the repo's .venv-f1-adaptive."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(DEST.parent / "F2_R1_TRAIN_2026-11-07_v003"))
import r1_pipeline as f2
from poc_phase1 import read_wav_pcm16, write_wav_pcm16


SEGMENT = "莫问归期02-对照微微跑调"
START, END = 4.272, 5.422
CORE_START, CORE_END = 4.420, 5.280
SHIFT_CENTS = 100
NAMES = ("original.wav", "raised.wav", "restored.wav")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inject_pitch(path, source, sr):
    ratio = 2 ** (SHIFT_CENTS / 1200)
    filters = (
        f"apad=pad_dur=0.25,rubberband=tempo=1:pitch={ratio:.12f}:"
        f"formant=preserved:pitchq=quality,atrim=end_sample={len(source)}"
    )
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-af", filters, "-f", "f32le", "pipe:1"],
        check=True, capture_output=True,
    )
    pitched = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if pitched.shape != source.shape or not np.isfinite(pitched).all():
        raise RuntimeError("注入引擎返回了意外的长度或非有限值")
    return f2.splice(source, pitched, sr, [(START, END, SHIFT_CENTS)])


def voiced_overlap(a, b):
    left, right = np.isfinite(a), np.isfinite(b)
    return np.count_nonzero(left & right) / np.count_nonzero(left | right)


def measure(original, raised, restored, sr):
    pitch = f2.parselmouth.Sound(original, sr).to_pitch(
        time_step=f2.HOP, pitch_floor=75, pitch_ceiling=600
    )
    time = pitch.xs()
    first, second, third = [f2.cents_of(y, sr) for y in (original, raised, restored)]
    core = (time >= CORE_START) & (time <= CORE_END)
    whole = (time >= START) & (time <= END)
    shared = np.isfinite(first) & np.isfinite(second) & np.isfinite(third)
    if np.count_nonzero(core & shared) != np.count_nonzero(core):
        raise RuntimeError("目标长音中段有未能测量音高的帧")

    def median(delta, mask):
        return round(float(np.median(delta[mask & shared])), 3)

    return {
        "core_seconds": [CORE_START, CORE_END],
        "core_voiced_frames": int(np.count_nonzero(core)),
        "core_f0_hz": [
            round(float(440 * 2 ** (np.median(track[core]) / 1200)), 2)
            for track in (first, second, third)
        ],
        "core_raise_vs_original_cents": median(second - first, core),
        "core_restore_vs_raised_cents": median(third - second, core),
        "core_restore_vs_original_cents": median(third - first, core),
        "full_span_restore_vs_original_cents": median(third - first, whole),
        "full_span_p90_absolute_residual_cents": round(
            float(np.percentile(np.abs((third - first)[whole & shared]), 90)), 3
        ),
        "voiced_overlap_vs_original": [voiced_overlap(first, second), voiced_overlap(first, third)],
    }


def main():
    audio = DEST / "audio"
    if audio.exists() or (DEST / "report.json").exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")

    source_path = f2.seg_path(SEGMENT)
    source_hash = digest(source_path)
    source, sr = read_wav_pcm16(source_path)
    raised, injection_mask = inject_pitch(source_path, source, sr)
    restored, correction_mask = f2.corrected_audio(
        raised, sr, [(START, END, -SHIFT_CENTS)]
    )
    if any(y.shape != source.shape for y in (raised, restored)):
        raise RuntimeError("音频长度发生变化")

    source_lufs, _ = f2.ebur128(source_path)
    gain_db = -18.5 - source_lufs
    gain = 10 ** (gain_db / 20)
    variants = [y * gain for y in (source, raised, restored)]
    if any(not np.isfinite(y).all() or np.max(np.abs(y)) >= 1 for y in variants):
        raise RuntimeError("输出包含非有限值或削波")

    audio.mkdir()
    for name, signal in zip(NAMES, variants):
        write_wav_pcm16(audio / name, signal, sr)

    loaded = [read_wav_pcm16(audio / name) for name in NAMES]
    original_wav, raised_wav, restored_wav = [signal for signal, _ in loaded]
    outside_injection = injection_mask == 0
    outside_correction = correction_mask == 0
    sample_matches = {
        "raised_vs_anchor_outside_injection": bool(
            np.array_equal(raised_wav[outside_injection], original_wav[outside_injection])
        ),
        "restored_vs_raised_outside_correction": bool(
            np.array_equal(restored_wav[outside_correction], raised_wav[outside_correction])
        ),
        "restored_vs_anchor_outside_both": bool(
            np.array_equal(
                restored_wav[outside_injection & outside_correction],
                original_wav[outside_injection & outside_correction],
            )
        ),
    }
    f0 = measure(original_wav, raised_wav, restored_wav, sr)
    levels = {}
    for name, (signal, rate) in zip(NAMES, loaded):
        lufs, dbtp = f2.ebur128(audio / name)
        levels[name] = {
            "samples": len(signal),
            "sample_rate": rate,
            "lufs": lufs,
            "dbtp": dbtp,
            "peak": round(float(np.max(np.abs(signal))), 6),
            "sha256": digest(audio / name),
        }
    same_source = source_hash == digest(source_path)
    checks = {
        "source_unchanged": same_source,
        "length_and_rate": all(n["samples"] == len(source) and n["sample_rate"] == sr for n in levels.values()),
        "finite_and_no_clipping": all(
            np.isfinite(signal).all() and np.max(np.abs(signal)) < 1 for signal, _ in loaded
        ),
        "loudness_and_true_peak": all(
            abs(n["lufs"] + 18.5) <= 0.3 and n["dbtp"] <= -1 for n in levels.values()
        ),
        "unchanged_outside_splices": all(sample_matches.values()),
        "voiced_overlap_at_least_0_95": min(f0["voiced_overlap_vs_original"]) >= 0.95,
        "median_pitch_errors_under_5_cents": all(
            abs(error) < 5 for error in (
                f0["core_raise_vs_original_cents"] - SHIFT_CENTS,
                f0["core_restore_vs_raised_cents"] + SHIFT_CENTS,
                f0["core_restore_vs_original_cents"],
                f0["full_span_restore_vs_original_cents"],
            )
        ),
    }
    report = {
        "source": str(source_path),
        "source_sha256": source_hash,
        "span_seconds": [START, END],
        "injection": {
            "engine": "FFmpeg rubberband (tempo=1, formant=preserved, pitchq=quality)",
            "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "shift_cents": SHIFT_CENTS,
        },
        "restore": {
            "engine": "F2_R1_TRAIN_2026-11-07_v003/r1_pipeline.py:corrected_audio",
            "parselmouth_version": f2.parselmouth.__version__,
            "shift_cents": -SHIFT_CENTS,
            "model_used": False,
        },
        "common_gain_db": round(gain_db, 3),
        "file_levels": levels,
        "pitch": f0,
        "outside_splice_sample_equality": sample_matches,
        "checks": checks,
        "engineering": "technical_pass" if all(checks.values()) else "technical_fail",
        "listening": "pending_jade",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    (DEST / "report.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not all(checks.values()):
        raise SystemExit("工程检查未通过")


if __name__ == "__main__":
    main()
