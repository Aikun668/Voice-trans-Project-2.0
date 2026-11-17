"""对齐探针：生成"方向夸张"的对照音频 + 统一响度，供 Jade 指认想要的方向。

这些不是候选方案，是"问路用的路标"：每条故意夸张到一听就能分辨方向。
复用 Pedalboard 现成效果器；响度用 FFmpeg ebur128 计量后只施加固定增益。
用 .venv-f1-adaptive 运行（需 soundfile/pedalboard，PATH 中有 ffmpeg）。
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (
    Chorus,
    Compressor,
    Delay,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    Pedalboard,
    PeakFilter,
    PitchShift,
    Reverb,
)

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RAW = HERE / "raw"
OUT = HERE / "audio"
TARGET_LUFS = -18.0
LOUD_LUFS = -13.0
PEAK_CEIL_DBFS = -1.0

SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
A01 = SELF / "平凡之路薄音版.wav"
A02 = SELF / "平凡之路正常版.wav"
S45 = ROOT / "experiments/F1-EXP-020/outputs/processed_raw/S45.wav"
GYJYN = SELF / "因为刚好遇见你-清唱版.wav"
P3E = ROOT / "f2/outputs/phase3e_gyjyn_v002/02_corrected.wav"
REF_GYJYN = ROOT / "f2/outputs/phase3e_gyjyn_v002/03_reference_aligned.wav"
SHIGE_N = ROOT / "test_dataset/my_singing/shige_normal.wav"
SHIGE_S = ROOT / "test_dataset/my_singing/shige_severe.wav"


def lufs_of(path: Path) -> float:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    return float(re.findall(r"I:[ ]+(-?[0-9.]+) LUFS", r.stderr)[-1])


def true_peak_of(path: Path) -> float:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    return float(re.findall(r"Peak:[ ]+(-?[0-9.]+) dBFS", r.stderr)[-1])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def board_vocal_chain() -> Pedalboard:
    """常见流行人声处理链（夸张档，不含混响）：切低频浑浊→补身体→重压缩→补存在感/空气感→限幅。"""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90),
        PeakFilter(cutoff_frequency_hz=180, gain_db=3.0, q=0.9),
        PeakFilter(cutoff_frequency_hz=450, gain_db=-2.0, q=1.2),
        Compressor(threshold_db=-28, ratio=4.0, attack_ms=5, release_ms=80),
        Gain(gain_db=8.0),
        PeakFilter(cutoff_frequency_hz=3200, gain_db=4.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=3.0, q=0.7),
        Limiter(threshold_db=-3.0, release_ms=60),
    ])


def board_reverb() -> Pedalboard:
    return Pedalboard([Reverb(room_size=0.45, damping=0.5, wet_level=0.18, dry_level=0.85, width=0.0)])


def doubled(x: np.ndarray, sr: int) -> np.ndarray:
    """加厚/双叠：原声 + 两路微移音高、微延迟的副本（像同一个人唱了好几遍叠在一起）。"""
    a = Pedalboard([PitchShift(semitones=0.09), Delay(delay_seconds=0.021, feedback=0.0, mix=1.0)])(x, sr)
    b = Pedalboard([PitchShift(semitones=-0.09), Delay(delay_seconds=0.033, feedback=0.0, mix=1.0)])(x, sr)
    y = x + 0.55 * a[: len(x)] + 0.55 * b[: len(x)]
    return Pedalboard([Chorus(rate_hz=0.6, depth=0.12, centre_delay_ms=8, feedback=0.0, mix=0.25)])(y, sr)


def write_raw(name: str, y: np.ndarray, sr: int) -> Path:
    p = RAW / f"{name}.wav"
    sf.write(str(p), np.clip(y, -1, 1), sr, subtype="PCM_16")
    return p


def match(src: Path, name: str, target: float) -> dict:
    """固定增益到目标响度；超峰值上限时整体降增益而不是限幅，并如实记录。"""
    x, sr = mono(src)
    gain_db = target - lufs_of(src)
    peak = float(np.max(np.abs(x))) * 10 ** (gain_db / 20)
    note = ""
    if 20 * np.log10(max(peak, 1e-9)) > PEAK_CEIL_DBFS:
        gain_db -= 20 * np.log10(peak) - PEAK_CEIL_DBFS
        note = "peak-limited: 响度低于目标"
    dst = OUT / f"{name}.wav"
    sf.write(str(dst), x * 10 ** (gain_db / 20), sr, subtype="PCM_16")
    return {
        "file": dst.name, "source": str(src.relative_to(ROOT)), "source_sha16": sha(src),
        "sr": sr, "seconds": round(len(x) / sr, 2), "gain_db": round(gain_db, 2),
        "lufs": lufs_of(dst), "true_peak_dbfs": true_peak_of(dst), "note": note,
    }


def main() -> None:
    RAW.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    a01, sr = mono(A01)
    chain = board_vocal_chain()(a01, sr)
    probes_raw = {
        "f1_chain": write_raw("f1_chain", chain, sr),
        "f1_chain_reverb": write_raw("f1_chain_reverb", board_reverb()(chain, sr), sr),
        "f1_double": write_raw("f1_double", doubled(a01, sr), sr),
        # 唱片级响度：链后再推增益进限幅器，模拟成品歌曲的"大声"，而非等响比较
        "f1_chain_loud": write_raw(
            "f1_chain_loud", Pedalboard([Gain(gain_db=6.0), Limiter(threshold_db=-1.5, release_ms=60)])(chain, sr), sr
        ),
    }
    plan = [
        ("f1_orig", A01, TARGET_LUFS),
        ("f1_self_normal", A02, TARGET_LUFS),
        ("f1_s45", S45, TARGET_LUFS),
        ("f1_orig_loud", A01, LOUD_LUFS),
        ("f1_chain", probes_raw["f1_chain"], TARGET_LUFS),
        ("f1_chain_loud", probes_raw["f1_chain_loud"], LOUD_LUFS),
        ("f1_chain_reverb", probes_raw["f1_chain_reverb"], TARGET_LUFS),
        ("f1_double", probes_raw["f1_double"], TARGET_LUFS),
        ("f2_gyjyn_orig", GYJYN, TARGET_LUFS),
        ("f2_gyjyn_p3e", P3E, TARGET_LUFS),
        ("f2_gyjyn_hardsnap", RAW / "f2_gyjyn_hardsnap.wav", TARGET_LUFS),
        ("f2_gyjyn_reference", REF_GYJYN, TARGET_LUFS),
        ("f2_shige_normal", SHIGE_N, TARGET_LUFS),
        ("f2_shige_severe", SHIGE_S, TARGET_LUFS),
        ("f2_shige_severe_hardsnap", RAW / "f2_shige_severe_hardsnap.wav", TARGET_LUFS),
    ]
    report = [match(src, name, tgt) | {"target_lufs": tgt} for name, src, tgt in plan]
    (HERE / "levels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:32s} {r['lufs']:6.1f} LUFS  TP {r['true_peak_dbfs']:5.1f}  {r['seconds']:5.1f}s {r['note']}")


if __name__ == "__main__":
    main()
