"""Independent synthesis adapters and objective probes for the same “学” curve."""

import importlib.util
import json
from math import gcd
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

DEST = Path(__file__).resolve().parent
V002 = DEST.parent / "F2_XINQIANG_XUE_CONTOUR_2026-11-08_v002"
spec = importlib.util.spec_from_file_location("approved_splice", V002 / "run.py")
approved_splice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approved_splice)
old = approved_splice.previous
sys.path.insert(0, str(DEST / "vendor" / "python"))
import pyworld

LEFT_FADE = approved_splice.LEFT
RIGHT_FADE = approved_splice.RIGHT
REGION = (0.550, 1.420)
RUBBERBAND = DEST / "vendor/rubberband/rubberband-4.0.0-gpl-executable-windows/rubberband-r3.exe"


def approved_curve(source, sr, report):
    times, cents = old.pitch(source, sr)
    body = (times >= old.SOURCE_BODY[0]) & (times < old.SOURCE_BODY[1]) & np.isfinite(cents)
    grid = (np.arange(int(np.ceil(len(source) / sr / old.HOP))) + 0.5) * old.HOP
    shape = np.interp(grid, times[body], cents[body])
    enter = np.clip((grid - 0.81) / (0.879 - 0.81), 0, 1)
    leave = np.clip((1.13 - grid) / (1.13 - 1.09), 0, 1)
    envelope = np.sin(0.5 * np.pi * np.minimum(enter, leave)) ** 2
    target = report["target_cents"] + report["retained_fraction_of_glide"] * (shape - report["source_median_cents"])
    return grid, (target - shape) * envelope, (times[body], cents[body])


def synthesize_world(source, sr, guide, *, passthrough=False, calibration_cents=0.0):
    grid, correction, (pitch_time, pitch_cents) = guide
    region = old.span_slice(REGION, sr)
    cut = np.ascontiguousarray(source[region])
    frame_ms = 2.5
    f0, time = pyworld.dio(cut, sr, f0_floor=75, f0_ceil=600, frame_period=frame_ms)
    f0 = pyworld.stonemask(cut, f0, time, sr)
    absolute = time + region.start / sr
    body = (absolute >= 0.883) & (absolute < 1.087)
    # Praat tracks this steep glide more continuously than the default WORLD detector.
    f0[body] = 440 * 2 ** (np.interp(absolute[body], pitch_time, pitch_cents) / 1200)
    envelope = pyworld.cheaptrick(cut, f0, time, sr)
    noise = pyworld.d4c(cut, f0, time, sr)
    requested = np.interp(absolute, grid, correction)
    requested[body] += calibration_cents
    target = f0 if passthrough else f0 * 2 ** (requested / 1200)
    reconstructed = pyworld.synthesize(target, envelope, noise, sr, frame_period=frame_ms)
    return insert_region(source, reconstructed, region)


def insert_region(source, rendered, region):
    count = region.stop - region.start
    if len(rendered) < count - 1 or not np.isfinite(rendered).all():
        raise RuntimeError("重合成音频的长度或样本异常")
    output = source.copy()
    output[region] = rendered[:count] if len(rendered) >= count else np.pad(rendered, (0, 1))
    return output


def synthesize_rubberband(source, sr, guide, *, short_window=False, command_advance_ms=0):
    grid, correction, _ = guide
    region = old.span_slice((0.35, 1.75), sr)
    clip = source[region]
    with TemporaryDirectory(prefix="rubberband_probe_", dir=DEST) as directory:
        folder = Path(directory)
        input_path, output_path, map_path = (folder / name for name in ("in.wav", "out.wav", "pitch.txt"))
        sf.write(input_path, clip, sr, subtype="FLOAT")
        samples = np.arange(0, len(clip), round(sr * 0.005))
        command_times = samples / sr + region.start / sr + command_advance_ms / 1000
        mapped = np.interp(command_times, grid, correction)
        map_path.write_text("".join(f"{i} {c / 100:.7f}\n" for i, c in zip(samples, mapped, strict=True)),
                            encoding="ascii")
        args = [str(RUBBERBAND), "-3", "-F", "-t", "1", "--pitchmap", str(map_path)]
        if short_window:
            args.append("--window-short")
        subprocess.run([*args, str(input_path), str(output_path)], capture_output=True, check=True)
        y, rate = sf.read(output_path, dtype="float64")
    if rate != sr or y.ndim != 1 or len(y) < round(0.85 * sr):
        raise RuntimeError("Rubber Band 输出缺少目标区间")
    # The CLI's pitch-map mode can end early. Only use the measured interior.
    extent = slice(region.start, region.start + min(len(y), len(clip)))
    return insert_region(source, y, extent)


def synthesize_lpc(source, sr, guide, *, calibration_cents=0.0):
    grid, correction, _ = guide
    parselmouth = old.f2.parselmouth
    call = parselmouth.praat.call
    sound = parselmouth.Sound(source, sr)
    manip = call(sound, "To Manipulation", 0.01, 75.0, 600.0)
    tier = call(manip, "Extract pitch tier")
    new = call("Create PitchTier", "target", 0.0, len(source) / sr)
    for index in range(1, call(tier, "Get number of points") + 1):
        time = call(tier, "Get time from index", index)
        hz = call(tier, "Get value at index", index)
        requested = float(np.interp(time, grid, correction))
        if 0.883 <= time < 1.087:
            requested += calibration_cents
        call(new, "Add point", time, hz * 2 ** (requested / 1200))
    call([manip, new], "Replace pitch tier")
    rendered = call(manip, "Get resynthesis (LPC)")
    rate = round(rendered.sampling_frequency)
    y = np.asarray(rendered.values[0], dtype=float)
    if rate != sr:
        divisor = gcd(sr, rate)
        y = resample_poly(y, sr // divisor, rate // divisor)
    return insert_region(source, y, slice(0, len(source)))


def stitch(source, synthetic, sr):
    left, right = (old.span_slice(span, sr) for span in (LEFT_FADE, RIGHT_FADE))
    center = old.span_slice(old.SOURCE_CORE, sr)
    gain = np.sqrt(np.sum(source[center] ** 2) / np.sum(synthetic[center] ** 2))
    synthetic = synthetic * gain
    result = source.copy()
    result[left.stop:right.start] = synthetic[left.stop:right.start]
    for span, step in ((left, 1), (right, -1)):
        weight = np.sin(np.linspace(0, np.pi / 2, span.stop - span.start)) ** 2
        result[span] += weight[::step] * (synthetic[span] - source[span])
    return result, float(20 * np.log10(gain))


def score(original, candidate, sr, guide, target):
    times, q = old.pitch(original, sr)
    _, p = old.pitch(candidate, sr)
    core = (times >= old.SOURCE_CORE[0]) & (times < old.SOURCE_CORE[1])
    shared = np.isfinite(q) & np.isfinite(p)
    points = core & shared
    voiced_ratio = np.count_nonzero(points) / np.count_nonzero(core & np.isfinite(q))
    if np.count_nonzero(points) < 5:
        return {"core_voiced_overlap": voiced_ratio, "median_command_error_cents": None}
    grid, correction, _ = guide
    error = (p - q)[points] - np.interp(times[points], grid, correction)
    return {
        "core_voiced_overlap": voiced_ratio,
        "median_command_error_cents": float(np.median(error)),
        "p90_absolute_command_error_cents": float(np.percentile(np.abs(error), 90)),
        "target_center_error_cents": float(np.median(p[points]) - target),
        "candidate_core_hz_p10_p90": (440 * 2 ** (np.percentile(p[points], [10, 90]) / 1200)).tolist(),
    }


def main():
    report = json.loads((approved_splice.PREVIOUS / "report.json").read_text(encoding="utf-8"))
    x, sr = old.read_wav_pcm16(old.f2.seg_path(old.SEGMENT))
    guide = approved_curve(x, sr, report)
    choices = {
        "world": lambda: synthesize_world(x, sr, guide),
        "rubberband_r3": lambda: synthesize_rubberband(x, sr, guide),
        "rubberband_r3_short": lambda: synthesize_rubberband(x, sr, guide, short_window=True),
        "lpc": lambda: synthesize_lpc(x, sr, guide),
    }
    for name, render in choices.items():
        try:
            y, gain = stitch(x, render(), sr)
            metrics = score(x, y, sr, guide, report["target_cents"])
            print(name, json.dumps(metrics | {"core_match_gain_db": gain}, ensure_ascii=False), flush=True)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(name, type(exc).__name__, str(exc),
                  getattr(exc, "stderr", b"").decode(errors="replace"), flush=True)


if __name__ == "__main__":
    main()
