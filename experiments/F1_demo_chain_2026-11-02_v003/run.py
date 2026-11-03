"""F1 成品链 v003（2026-11-02）：A 干声链（去"字尾空响"）、B = A + 第一共振峰上移（发声力度）。

依据 demo v002 反馈：平凡之路/我们的歌仍选 V1、仍报"空旷/回音/录音机感"；雅俗共赏选 V3；
目标是"声带能量感"，现状"只是响度增大 + 空响回音副作用"。
诊断（/tmp 测量，记录于 README）：V1/V3 让原音字尾衰减段（相对响段 −6~−25 dB、下降 >60 dB/s）
相对演唱主体抬高 +7~+10 dB，响段只 +1 dB —— 压缩器 5 ms 起音 / 80 ms 释放在每个字结束后迅速回弹，
把房间尾音放大约 3 倍。这是在句子内部发生的，v002 的句间底噪处理管不到。
A：V1 的 EQ（Jade 两首选它）不变；压缩改为 25 ms 起音 / 350 ms 释放（字尾期间增益不回弹，起音冲击保留）；
   字尾守护：衰减段相对抬升超过 DECAY_ALLOW_DB 的部分压回；再做 v001 安静段复原与 v002 句间底噪下压。
B：A 之前先做 F1 上移 SHIFT_HZ（EXP-020 LPC 极点旋转，formant_shift.py 原样复制，已逐样本复现 S45）。
   S45（相对原音 +102.5 Hz）是唯一获"饱满度 OK、清晰度 OK、像本人"的版本；此处不依赖配对正常版，单独上移。
用 .venv-f1-adaptive 运行。
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Gain, HighpassFilter, HighShelfFilter, Limiter, Pedalboard, PeakFilter
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v1 = _load("f1_v001_run", ROOT / "experiments/F1_demo_chain_2026-11-02_v001/run.py")
_v2 = _load("f1_v002_run", ROOT / "experiments/F1_demo_chain_2026-11-02_v002/run.py")
_fs = _load("formant_shift", HERE / "formant_shift.py")
ebur, master, restore_quiet, sha16 = _v1.ebur, _v1.master, _v1.restore_quiet, _v1.sha16
floor_cut, chain_v3, target_spectrum = _v2.floor_cut, _v2.chain_v3, _v2.target_spectrum

SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
SONGS = {
    "pingfan": SELF / "平凡之路薄音版.wav",
    "womende": SELF / "我们的歌-薄音版.wav",
    "yasu": SELF / "雅俗共赏-清唱版.wav",
    "gyjyn": SELF / "因为刚好遇见你-清唱版.wav",
}
SHIFT_HZ = 70.0
DECAY_ALLOW_DB = 3.0
HOP_S = 0.005


def env_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * HOP_S)
    return 10 * np.log10(uniform_filter1d(x.astype(np.float64) ** 2, size=int(0.02 * sr))[::hop] + 1e-12)


def decay_mask(orig: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """原音字尾衰减段：相对响段 −4 dB 以下、平滑后下降 >40 dB/s。返回 (mask, 原音相对电平)。"""
    e = env_db(orig, sr)
    rel = e - np.percentile(e, 95)
    slope = np.gradient(uniform_filter1d(e, 6)) / HOP_S
    return (slope < -40.0) & (rel < -4.0), rel


def decay_guard(orig: np.ndarray, proc: np.ndarray, sr: int) -> np.ndarray:
    """衰减段内，处理后相对电平比原音多出 DECAY_ALLOW_DB 以上的部分压回（20 ms 平滑）。"""
    m, rel_o = decay_mask(orig, sr)
    e = env_db(proc, sr)
    n = min(len(e), len(rel_o))
    lift = (e - np.percentile(e, 95))[:n] - rel_o[:n]
    g_db = np.where(m[:n], -np.maximum(lift - DECAY_ALLOW_DB, 0.0), 0.0)
    g_db = uniform_filter1d(g_db, size=4)                                    # 20 ms 平滑，仍为非正
    t = np.arange(len(proc)) / sr
    g = 10 ** (np.interp(t, np.arange(n) * HOP_S, g_db) / 20)
    return (proc * g).astype(np.float32)


def chain_dry(x: np.ndarray, sr: int) -> np.ndarray:
    """V1 的 EQ，压缩改慢释放；再字尾守护 + 安静段复原 + 句间底噪下压。"""
    y = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90),
        PeakFilter(cutoff_frequency_hz=180, gain_db=3.0, q=0.9),
        PeakFilter(cutoff_frequency_hz=450, gain_db=-2.0, q=1.2),
        Compressor(threshold_db=-28, ratio=3.0, attack_ms=25, release_ms=350),
        Gain(gain_db=7.0),
        PeakFilter(cutoff_frequency_hz=3200, gain_db=4.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=3.0, q=0.7),
        Limiter(threshold_db=-3.0, release_ms=60),
    ])(x, sr)
    return floor_cut(x, restore_quiet(x, decay_guard(x, y, sr), sr), sr)


def chain_effort(x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    y, stats = _fs.apply_shift(x.astype(np.float64), sr, SHIFT_HZ)
    y = np.asarray(y, dtype=np.float32)[: len(x)]
    # 守护与复原以"原音"的时间结构为键（共振峰移位不改时序）
    out = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90),
        PeakFilter(cutoff_frequency_hz=180, gain_db=3.0, q=0.9),
        PeakFilter(cutoff_frequency_hz=450, gain_db=-2.0, q=1.2),
        Compressor(threshold_db=-28, ratio=3.0, attack_ms=25, release_ms=350),
        Gain(gain_db=7.0),
        PeakFilter(cutoff_frequency_hz=3200, gain_db=4.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=3.0, q=0.7),
        Limiter(threshold_db=-3.0, release_ms=60),
    ])(y, sr)
    return floor_cut(x, restore_quiet(x, decay_guard(x, out, sr), sr), sr), stats


def lift_stats(orig: np.ndarray, proc: np.ndarray, sr: int) -> dict:
    m, rel_o = decay_mask(orig, sr)
    e = env_db(proc, sr)
    n = min(len(e), len(rel_o))
    d = (e - np.percentile(e, 95))[:n] - rel_o[:n]
    loud = rel_o[:n] > -6
    return {"decay_lift_db": round(float(np.median(d[m[:n]])), 1), "loud_lift_db": round(float(np.median(d[loud])), 1)}


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"输出目录已存在, 拒绝覆盖: {out}")
    out.mkdir()
    tgt = target_spectrum()
    report = []
    for key, src in SONGS.items():
        h0 = sha16(src)
        x, sr = sf.read(str(src), dtype="float32")
        b, fstats = chain_effort(x, sr)
        versions = {
            "orig": x,
            "v1": restore_quiet(x, _v1.chain_v1()(x, sr), sr),
            "v3": chain_v3(x, sr, tgt)[0],
            "a": chain_dry(x, sr),
            "b": b,
        }
        for name, y in versions.items():
            m, _, _, _ = master(y, sr)
            p = out / f"{key}_{name}.wav"
            sf.write(str(p), m, sr, subtype="PCM_16")
            back, _ = sf.read(str(p), dtype="float32")
            lufs, tp = ebur(p)
            row = {"song": key, "version": name, "file": p.name, "source": str(src.relative_to(ROOT)),
                   "source_sha16": h0, "samples_match": len(back) == len(x),
                   "clipped_samples": int(np.sum(np.abs(back) >= 0.9999)), "lufs": lufs, "true_peak_dbtp": tp}
            row |= lift_stats(x, back, sr)
            if name == "b":
                row["formant"] = {k: v for k, v in fstats.items() if isinstance(v, (int, float, str))}
            report.append(row)
        assert sha16(src) == h0
    (HERE / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:18s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} clip {r['clipped_samples']} "
              f"len_ok {r['samples_match']} 字尾抬升 {r['decay_lift_db']:+5.1f} dB 响段 {r['loud_lift_db']:+4.1f} "
              f"{r.get('formant', '')}")


if __name__ == "__main__":
    main()
