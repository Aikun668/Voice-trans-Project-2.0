"""Measure the layer split: original / vocoder_only / a2a / a2p_nomap / a2p.

Read-only. Uses the batch outputs from F1_neuralsvb_layer_split_2026-11-01_v002.
Key questions:
- Does a2a (VAE reconstruction, no latent mapping) already carry the nasal tilt / consonant loss?
- Does a2p add anything over a2a? (mapping-only effect)
- How much does vocoder_only change?
"""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "F1_neuralsvb_layer_split_2026-11-01_v002"
SR = 22050
N, H = 2048, 512
STEP = 0.01
BANDS = {"low_0_500": (0, 500), "mid1_500_1500": (500, 1500),
         "mid2_1500_4000": (1500, 4000), "high_4000_8000": (4000, 8000)}
CLIPS = {
    "A01 平凡之路薄音版": {"dir": "平凡之路薄音版", "a2p": "candidate", "others": "candidate"},
    "A03 我们的歌薄音版": {"dir": "我们的歌-薄音版", "a2p": "candidate", "others": "candidate"},
}
VARIANTS = {
    "vocoder_only": ("candidate", "__vocoder_only.wav"),
    "a2a": ("diagnostics", "__a2a.wav"),
    "a2p_nomap": ("diagnostics", "__a2p_nomap.wav"),
    "a2p": ("candidate", "__a2p.wav"),
}


def read(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float64")
    if sr != SR:
        g = math.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def stft_power(x: np.ndarray):
    w = np.hanning(N)
    n = 1 + max(0, (len(x) - N) // H)
    fr = np.stack([x[i * H:i * H + N] * w for i in range(n)]) if n else np.zeros((0, N))
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2, np.fft.rfftfreq(N, 1.0 / SR)


def band(p, f, lo, hi, sel):
    b = (f >= lo) & (f < hi)
    return float(p[np.ix_(sel, b)].sum()) if sel.any() else 0.0


def rel_bands(p, f, sel):
    vals = {k: band(p, f, lo, hi, sel) for k, (lo, hi) in BANDS.items()}
    ref = max(vals["mid2_1500_4000"], 1e-20)
    return {k: float(10 * np.log10(max(v, 1e-20) / ref)) for k, v in vals.items()}


def env_db(x):
    n = int(round(SR * STEP))
    m = len(x) // n
    fr = x[:m * n].reshape(m, n)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(fr ** 2, axis=1)), 1e-12))


def best_shift(ref, test, max_lag=20):
    m = min(len(ref), len(test))
    a, b = ref[:m] - ref[:m].mean(), test[:m] - test[:m].mean()
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-20
    best = (0, float(np.sum(a * b) / den))
    for lag in range(-max_lag, max_lag + 1):
        bs = np.roll(b, lag)
        if lag > 0:
            bs[:lag] = b[0]
        elif lag < 0:
            bs[lag:] = b[-1]
        c = float(np.sum(a * bs) / den)
        if c > best[1]:
            best = (lag, c)
    return best


def modulation(env):
    e = env - env.mean()
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), STEP)
    total = float(np.sum(spec[(f >= 0.5) & (f <= 20)])) + 1e-20
    return {k: round(float(np.sum(spec[(f >= lo) & (f < hi)]) / total), 4)
            for k, (lo, hi) in {"0.5_2": (0.5, 2), "2_8": (2, 8), "8_20": (8, 20)}.items()}


def noise_frames(x):
    logmag, f = None, None
    w = np.hanning(1024)
    n = 1 + max(0, (len(x) - 1024) // 256)
    fr = np.stack([x[i * 256:i * 256 + 1024] * w for i in range(n)])
    mag = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(1024, 1.0 / SR)
    low = (f >= 100) & (f < 8000)
    hi = (f >= 4000) & (f < 8000)
    e_hi = np.sum(mag[:, hi] ** 2, axis=1)
    e_all = np.sum(mag[:, low] ** 2, axis=1) + 1e-20
    flat = np.exp(np.mean(np.log(np.maximum(mag[:, low], 1e-12)), axis=1)) / \
        np.maximum(np.mean(mag[:, low], axis=1), 1e-12)
    lvl = 10 * np.log10(e_all)
    sel = (flat > 0.35) & (lvl > lvl.max() - 35)
    return int(sel.sum()), round(float(np.sum(e_hi[sel]) / max(np.sum(e_all[sel]), 1e-20)), 4)


results = []
for label, cfg in CLIPS.items():
    stem = cfg["dir"]
    orig = read(EXP / "candidate" / f"{stem}__original.wav")
    p_o, f = stft_power(orig)
    lvl = np.array([band(p_o, f, 100, 4000, np.array([i])) for i in range(len(p_o))])
    ldb = 10 * np.log10(np.maximum(lvl, 1e-20))
    active = ldb > ldb.max() - 30
    ro = rel_bands(p_o, f, active)
    eo = env_db(orig)
    orig_noise, orig_hi = noise_frames(orig)
    row = {"clip": label, "original": {k: round(v, 2) for k, v in ro.items()},
           "original_metrics": {"modulation": modulation(eo), "noise_frames": orig_noise,
                                "high_share_noise": orig_hi},
           "variants": {}}
    for name, (folder, suf) in VARIANTS.items():
        x = read(EXP / folder / f"{stem}{suf}")
        m = min(len(x), len(orig))
        p_x, _ = stft_power(x[:m])
        p_o_m, _ = stft_power(orig[:m])
        lvl = np.array([band(p_o_m, f, 100, 4000, np.array([i])) for i in range(len(p_o_m))])
        ldb = 10 * np.log10(np.maximum(lvl, 1e-20))
        act = ldb > ldb.max() - 30
        rx = rel_bands(p_x, f, act)
        d = {k: round(rx[k] - rel_bands(p_o_m, f, act)[k], 2) for k in BANDS}
        ex = env_db(x[:m])
        lag, corr = best_shift(eo[:len(ex)], ex)
        n_noise, hi_share = noise_frames(x[:m])
        row["variants"][name] = {
            "delta_vs_original": d,
            "mid_emphasis_db": round(-(d["low_0_500"] + d["high_4000_8000"]) / 2, 2),
            "env_lag_ms": round(lag * STEP * 1000, 1), "env_corr": round(corr, 4),
            "modulation": modulation(ex),
            "noise_frames": n_noise, "high_share_noise": hi_share,
        }
    results.append(row)

(EXP / "layer_metrics.json").write_text(json.dumps({"clips": results}, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
for r in results:
    print(f"--- {r['clip']}")
    for k, v in r["variants"].items():
        print(f"  {k:12s} tilt {v['mid_emphasis_db']:+.2f} | mod2-8 {v['modulation']['2_8']:.3f} | "
              f"noise {v['noise_frames']:4d} | corr {v['env_corr']:.3f} | "
              f"low {v['delta_vs_original']['low_0_500']:+.1f} high {v['delta_vs_original']['high_4000_8000']:+.1f}")
