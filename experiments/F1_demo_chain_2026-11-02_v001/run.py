"""F1 成品链 demo（2026-11-02 v001）。

依据 Jade 对齐页反馈：最接近目标的是"唱片人声链·唱片级响度"，但"声带能量还没拉起来"；
不要混响、不要双叠；变响算效果的一部分。
本轮两条候选，都在 -13 LUFS 成品响度下比较：
- V1 唱片链：对齐页 Q5 那条链（Jade 说"接近了"）+ 安静段复原（restore_quiet）。
  对齐页 Q4 反馈"更空旷、有回音感"：实测该链把原音安静段/底噪相对演唱段抬高 9–13 dB，
  即压缩器抬起了房间声。复原只作用于原音中低于响段 20 dB 以下的部分。
- V2 能量链：在 V1 基础上只朝"更实、不嗲不蔫"推：去掉空气感高架（气声感），
  改为 1.2–6 kHz 并联谐波激励 + 2.8 kHz 存在感 + 更密的两级压缩；同样做安静段复原。
  V1→V2 只差"能量"这一组改动。
音高粗检另由 pitch_check.py（.venv-f2，Praat）完成——librosa.pyin 在本环境 numba 段错误。
另附"原音·同响度"参考（只做增益+限幅到 -13 LUFS），用来区分"链的作用"和"只是更响"。

复用 Pedalboard 0.9.26 现成效果器；FFmpeg ebur128 计量。用 .venv-f1-adaptive 运行。
"""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (
    Compressor,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    PeakFilter,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
SONGS = {
    "pingfan": SELF / "平凡之路薄音版.wav",
    "womende": SELF / "我们的歌-薄音版.wav",
    "yasu": SELF / "雅俗共赏-清唱版.wav",
    "gyjyn": SELF / "因为刚好遇见你-清唱版.wav",
}
TARGET_LUFS = -13.0
TP_CEIL = -1.0
FRAME_S = 0.02


def ebur(path: Path) -> tuple[float, float]:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    lufs = float(re.findall(r"I:[ ]+(-?[0-9.]+) LUFS", r.stderr)[-1])
    tp = float(re.findall(r"Peak:[ ]+(-?[0-9.]+) dBFS", r.stderr)[-1])
    return lufs, tp


def ebur_array(x: np.ndarray, sr: int) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.wav"
        sf.write(str(p), x, sr, subtype="FLOAT")
        return ebur(p)


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * FRAME_S)
    k = len(x) // n
    f = x[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt(np.mean(f**2, axis=1)) + 1e-9)


def level_stats(x: np.ndarray, sr: int) -> dict:
    """活动段 P95 与 P10：P10 相对 P95 的抬升≈尾音/房间声被抬起多少。"""
    db = frame_db(x, sr)
    active = db[db > db.max() - 40]
    return {"p95": float(np.percentile(active, 95)), "p10": float(np.percentile(active, 10)),
            "floor_p05_all": float(np.percentile(db, 5))}


def chain_v1() -> Pedalboard:
    """对齐页 f1_chain 原样（见 HANDOFF_ALIGN_2026-11-02_v001/scripts/make_probes.py）。"""
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


def chain_v2(x: np.ndarray, sr: int) -> np.ndarray:
    st = level_stats(x, sr)
    pre = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=200, gain_db=2.5, q=0.9),
        PeakFilter(cutoff_frequency_hz=450, gain_db=-2.5, q=1.2),
        Compressor(threshold_db=st["p95"] - 12.0, ratio=4.0, attack_ms=3.0, release_ms=60.0),
        Gain(gain_db=8.0),
    ])(x, sr)
    # 并联谐波激励：只让 1.2–6 kHz 带通部分失真，再按 -12 dB 混回，补上高次谐波密度
    exc = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=1200),
        LowpassFilter(cutoff_frequency_hz=6000),
        Distortion(drive_db=18.0),
        LowpassFilter(cutoff_frequency_hz=7000),
        Gain(gain_db=-12.0),
    ])(pre, sr)
    y = pre + exc
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=2800, gain_db=3.5, q=0.9),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.0, q=0.7),
        Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=10.0, release_ms=100.0),
        Limiter(threshold_db=-3.0, release_ms=60),
    ])(y, sr)


def restore_quiet(orig: np.ndarray, proc: np.ndarray, sr: int, full_db: float = -30.0, none_db: float = -20.0,
                  hop_s: float = 0.005, win_s: float = 0.02, smooth_s: float = 0.03) -> np.ndarray:
    """以原音电平为键的向下扩展：原音里低于响段 30 dB 的部分（房间声、底噪、尾音残响）
    被压缩抬起多少就压回多少；-30→-20 dB 之间渐变；-20 dB 以上的演唱段不动。"""
    from scipy.ndimage import uniform_filter1d

    hop, win = int(sr * hop_s), int(sr * win_s)

    def env_db(x: np.ndarray) -> np.ndarray:
        p = uniform_filter1d(x.astype(np.float64) ** 2, size=win)[::hop]
        return 10 * np.log10(p + 1e-12)

    o, p = env_db(orig), env_db(proc)
    o_rel = o - np.percentile(o, 95)
    lift = (p - np.percentile(p, 95)) - o_rel
    w = np.clip((none_db - o_rel) / (none_db - full_db), 0.0, 1.0)
    g_db = -np.maximum(lift, 0.0) * w
    g_db = uniform_filter1d(g_db, size=max(1, int(smooth_s / hop_s)))
    t = np.arange(len(proc)) / sr
    g = 10 ** (np.interp(t, np.arange(len(g_db)) * hop_s + win_s / 2, g_db) / 20)
    return (proc * g).astype(np.float32)


def tp_limit(x: np.ndarray, sr: int, ceil_db: float, look_ms: float = 5.0, release_ms: float = 60.0) -> np.ndarray:
    """前视真峰值限幅：4 倍过采样求峰值包络，增益提前 look_ms 落到位，指数释放。
    Pedalboard Limiter 在本素材的瞬态上真峰值越界（实测 +0.5 dBTP），故最后一级用它。"""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import resample_poly

    env = np.abs(resample_poly(x, 4, 1)).reshape(-1, 4).max(axis=1)[: len(x)]
    env = np.pad(env, (0, len(x) - len(env)))
    ceil = 10 ** (ceil_db / 20)
    need = np.minimum(1.0, ceil / np.maximum(env, 1e-9))
    w = max(1, int(sr * look_ms / 1000))
    g = -maximum_filter1d(-need, size=2 * w + 1)          # 前后各 w 个样本内的最小增益
    g = uniform_filter1d(g, size=w)                       # 平滑，避免增益阶跃
    g = np.minimum(g, -maximum_filter1d(-need, size=2 * w + 1))
    a = np.exp(-1.0 / (sr * release_ms / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):                             # 释放：增益只能缓慢回升
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    return (x * out).astype(np.float32)


def master(x: np.ndarray, sr: int, target: float = TARGET_LUFS) -> tuple[np.ndarray, float, float, float]:
    """增益 + 前视真峰值限幅到目标响度。返回 (y, gain_db, lufs, tp)。"""
    gain = target - ebur_array(x, sr)[0]
    ceil = TP_CEIL - 0.3
    for _ in range(8):
        y = tp_limit(x * 10 ** (gain / 20), sr, ceil)
        lufs, tp = ebur_array(y, sr)
        if tp > TP_CEIL:
            ceil -= tp - TP_CEIL + 0.05
            continue
        if abs(lufs - target) <= 0.15:
            return y, gain, lufs, tp
        gain += target - lufs
    raise RuntimeError(f"无法同时满足 {target} LUFS 与 {TP_CEIL} dBTP: lufs={lufs} tp={tp}")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"输出目录已存在, 拒绝覆盖: {out}")
    out.mkdir()
    report = []
    for key, src in SONGS.items():
        h0 = sha16(src)
        x, sr = sf.read(str(src), dtype="float32")
        assert x.ndim == 1, src
        versions = {"orig": x, "v1": restore_quiet(x, chain_v1()(x, sr), sr), "v2": restore_quiet(x, chain_v2(x, sr), sr)}
        st_src = level_stats(x, sr)
        for name, y in versions.items():
            m, gain, lufs, tp = master(y, sr)
            p = out / f"{key}_{name}.wav"
            sf.write(str(p), m, sr, subtype="PCM_16")
            back, sr2 = sf.read(str(p), dtype="float32")
            lufs_b, tp_b = ebur(p)
            st = level_stats(back, sr)
            report.append({
                "song": key, "version": name, "file": p.name, "source": str(src.relative_to(ROOT)),
                "source_sha16": h0, "sr": sr2, "samples": len(back), "samples_match": len(back) == len(x),
                "clipped_samples": int(np.sum(np.abs(back) >= 0.9999)), "lufs": lufs_b, "true_peak_dbtp": tp_b,
                "master_gain_db": round(gain, 2),
                # 尾音/房间声抬升：处理后 (P10−P95) 相对原音 (P10−P95) 的变化，正值=轻声段被抬起
                "tail_lift_db": round((st["p10"] - st["p95"]) - (st_src["p10"] - st_src["p95"]), 2),
                "floor_lift_db": round((st["floor_p05_all"] - st["p95"]) - (st_src["floor_p05_all"] - st_src["p95"]), 2),
            })
        assert sha16(src) == h0, f"源文件被改动: {src}"
    (HERE / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:18s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} clip {r['clipped_samples']} "
              f"len_ok {r['samples_match']} tail {r['tail_lift_db']:+5.1f} floor {r['floor_lift_db']:+5.1f}")


if __name__ == "__main__":
    main()
