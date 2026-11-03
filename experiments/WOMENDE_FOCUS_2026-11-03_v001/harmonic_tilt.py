"""(.venv-f1-adaptive) 我们的歌 定点：浊音段"只抬谐波"的声源倾斜修正。

测量（source_measures.json）：薄音版 alpha ratio −27.4 dB，正常版 −16.1，王力宏 −16.8~−17.7；
H1−A3 37.1 vs 24.4/26.4~27.4。差距主要在高次谐波能量（声带用力的频谱特征），不在气声（H1−H2 不差）。
与以往 EQ 的区别：只在浊音帧、只在谐波峰附近（高斯窗 σ=0.12·F0）施加随频率升高的增益；
谐波之间的气声/噪声、清辅音、齿音不动。斜率 S（dB/倍频程，700 Hz 起）二分求解，使 alpha ratio 达到正常版。
候选：
  C1 = 薄音版 + 谐波倾斜（原调）
  C2 = 薄音版升 310 音分（到正常版的调）+ 谐波倾斜
全部等响 -18.5 LUFS（只施加固定增益），专门排除"只是变响"。
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
TARGET_ALPHA = json.loads((HERE / "source_measures.json").read_text(encoding="utf-8"))["正常版"]["alpha_ratio_dB"]
NFFT, HOP = 2048, 256
F_START, F_FULL, F_FADE = 700.0, 5000.0, 10000.0
SIGMA = 0.12
MAX_GAIN_DB = 15.0
MATCH_LUFS = -18.5          # 所有版本都能在 -1 dBFS 峰值内达到的共同响度


def load_f0(name: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(HERE / f"work/f0_{name}.csv", delimiter=",")
    return a[:, 0], a[:, 1]


def tilt_gain_db(f: np.ndarray, slope: float) -> np.ndarray:
    g = slope * np.clip(np.log2(np.maximum(f, 1) / F_START), 0, np.log2(F_FULL / F_START))
    fade = np.clip((F_FADE - f) / (F_FADE - F_FULL), 0, 1)       # 5–10 kHz 线性收回，避免抬空气噪声
    return np.minimum(np.where(f > F_FULL, g * fade, g), MAX_GAIN_DB)


def process(x: np.ndarray, sr: int, t0: np.ndarray, f0: np.ndarray, slope: float) -> np.ndarray:
    freqs, times, Z = stft(x, sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann", boundary="even", padded=True)
    fz = np.interp(times, t0, f0, left=0, right=0)
    # 插值会在浊/清边界产生假 F0：只保留两侧最近原始帧都是浊音的位置
    near = np.interp(times, t0, (f0 > 0).astype(float), left=0, right=0)
    voiced = (near > 0.99) & (fz > 60)
    w = np.convolve(voiced.astype(float), np.ones(5) / 5, mode="same")      # 约 25 ms 过渡
    g_tilt = tilt_gain_db(freqs, slope)
    G = np.ones_like(np.abs(Z))
    for k in np.nonzero(w > 0)[0]:
        f = fz[k] if fz[k] > 60 else np.max(fz[max(0, k - 3): k + 4])
        if f <= 60:
            continue
        d = freqs - np.round(freqs / f) * f
        m = np.exp(-0.5 * (d / (SIGMA * f)) ** 2) * (freqs > 0.5 * f)
        G[:, k] = 10 ** (g_tilt * m * w[k] / 20)
    _, y = istft(Z * G, sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann", boundary=True)
    return y[: len(x)].astype(np.float32)


def alpha_ratio(x: np.ndarray, sr: int, t0: np.ndarray, f0: np.ndarray) -> float:
    """与 measure_source.py 同口径：响段浊音帧、40 ms Hann、1–5 kHz vs 50 Hz–1 kHz。"""
    win = int(0.04 * sr)
    e = np.array([10 * np.log10(np.mean(x[max(0, int(t * sr) - win // 2): int(t * sr) + win // 2] ** 2) + 1e-12) for t in t0])
    use = (f0 > 0) & (e > np.percentile(e[f0 > 0], 95) - 15)
    fq = np.fft.rfftfreq(8192, 1 / sr)
    out = []
    for t in t0[use]:
        c = int(t * sr)
        fr = x[c - win // 2: c + win // 2]
        if len(fr) < win:
            continue
        p = np.abs(np.fft.rfft(fr * np.hanning(win), 8192)) ** 2
        out.append(10 * np.log10(p[(fq >= 1000) & (fq < 5000)].sum() / p[(fq >= 50) & (fq < 1000)].sum()))
    return float(np.median(out))


def solve(x, sr, t0, f0) -> tuple[np.ndarray, float, float]:
    lo, hi = 0.0, 8.0
    for _ in range(9):
        s = (lo + hi) / 2
        a = alpha_ratio(process(x, sr, t0, f0, s), sr, t0, f0)
        lo, hi = (s, hi) if a < TARGET_ALPHA else (lo, s)
    s = (lo + hi) / 2
    y = process(x, sr, t0, f0, s)
    return y, s, alpha_ratio(y, sr, t0, f0)


def main() -> None:
    sys.path.insert(0, str(ROOT / "experiments/F1_demo_chain_2026-11-02_v001"))
    from run import ebur  # noqa: E402

    out = HERE / "audio"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    out.mkdir()
    x, sr = sf.read(str(SELF / "我们的歌-薄音版.wav"), dtype="float32")
    xk, _ = sf.read(str(HERE / "work/thin_keyup.wav"), dtype="float32")
    report = {"target_alpha_db": TARGET_ALPHA}
    c1, s1, a1 = solve(x, sr, *load_f0("thin"))
    c2, s2, a2 = solve(xk, sr, *load_f0("thin_keyup"))
    report |= {"orig_alpha_db": round(alpha_ratio(x, sr, *load_f0("thin")), 2),
               "keyup_alpha_db": round(alpha_ratio(xk, sr, *load_f0("thin_keyup")), 2),
               "c1": {"slope_db_per_oct": round(s1, 3), "alpha_db": round(a1, 2),
                      "max_gain_db": round(float(tilt_gain_db(np.array([F_FULL]), s1)[0]), 1)},
               "c2": {"slope_db_per_oct": round(s2, 3), "alpha_db": round(a2, 2), "key_up_cents": 310,
                      "max_gain_db": round(float(tilt_gain_db(np.array([F_FULL]), s2)[0]), 1)}}
    for name, y in [("c1_tilt", c1), ("c2_keyup_tilt", c2)]:
        sf.write(str(HERE / f"work/{name}_raw.wav"), y, sr, subtype="FLOAT")
    plan = [("thin", SELF / "我们的歌-薄音版.wav"), ("normal", SELF / "我们的歌正常版.wav"),
            ("c1_tilt", HERE / "work/c1_tilt_raw.wav"), ("c2_keyup_tilt", HERE / "work/c2_keyup_tilt_raw.wav"),
            ("keyup_only", HERE / "work/thin_keyup.wav")]
    levels = []
    for name, p in plan:
        y, s = sf.read(str(p), dtype="float32")
        g = MATCH_LUFS - ebur(p)[0]
        pk = 20 * np.log10(np.max(np.abs(y)) + 1e-12) + g
        if pk > -1.0:
            g -= pk + 1.0
        dst = out / f"{name}.wav"
        sf.write(str(dst), y * 10 ** (g / 20), s, subtype="PCM_16")
        lufs, tp = ebur(dst)
        levels.append({"file": dst.name, "gain_db": round(g, 2), "lufs": lufs, "true_peak": tp, "samples": len(y)})
    report["levels"] = levels
    (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
