"""A03 词粘连 / 鼻音分层诊断（只读）。

对比 A03 原音、a2p、仅声码器，以及 A04 正常版参考：
1) 时间包络相关性、最佳延迟：模型是否改变了词/音的起止；
2) 包络调制谱 2–8Hz（音节率）相对 0.5–2Hz；
3) 光谱通量（帧间变化）——词边界/辅音瞬态是否被抹平；
4) 低电平间隙数量/时长——是否把字与字之间“粘”起来；
5) 4–8kHz 在噪声型（辅音）帧上的能量；
6) 按 2s 窗找“通量下降最多”的候选片段，供试听。
不生成音频，不改任何文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "experiments" / "F1_neuralsvb_8songs_2026-11-01_v001" / "listening" / "audio"
OUT = ROOT / "experiments" / "F1_neuralsvb_sticky_diag_2026-11-01_v001" / "run-002"
if OUT.exists():
    raise SystemExit(f"{OUT} already exists")
OUT.mkdir(parents=True)

SR = 22050
STEP_S = 0.01          # 包络帧移
N_FFT, HOP = 1024, 256  # 通量/谱特征
WIN_S, WIN_STEP = 2.0, 0.5
SAMPLES = ["A03", "A04"]
KEYS = ["original", "a2p", "vocoder_only"]


def load(sid: str, key: str) -> np.ndarray:
    name = f"{sid}_{key}.wav" if key != "normal" else None
    if key == "normal":
        name = f"{'A04' if sid == 'A03' else 'A03'}_original.wav"
    x, sr = sf.read(str(AUDIO / name), dtype="float64")
    if x.ndim > 1:
        raise ValueError(name)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def env_db(x: np.ndarray) -> np.ndarray:
    n = int(round(SR * STEP_S))
    m = len(x) // n
    frames = x[:m * n].reshape(m, n)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-12))


def best_shift(ref: np.ndarray, test: np.ndarray, max_lag: int = 20):
    """Return (lag, corr_before, corr_after); positive lag shifts `test` right."""
    m = min(len(ref), len(test))
    a, b = ref[:m] - ref[:m].mean(), test[:m] - test[:m].mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-20
    corr0 = float(np.sum(a * b) / denom)
    best = (0, corr0, corr0)
    for lag in range(-max_lag, max_lag + 1):
        bs = np.roll(b, lag)
        if lag > 0:
            bs[:lag] = b[0]
        elif lag < 0:
            bs[lag:] = b[-1]
        c = float(np.sum(a * bs) / denom)
        if c > best[2]:
            best = (lag, corr0, c)
    return best


def modulation_bands(env: np.ndarray) -> dict:
    e = env - env.mean()
    if len(e) < 16:
        return {}
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    freqs = np.fft.rfftfreq(len(e), STEP_S)
    total = float(np.sum(spec[(freqs >= 0.5) & (freqs <= 20)])) + 1e-20
    bands = {"0.5_2_hz": (0.5, 2), "2_8_hz": (2, 8), "8_20_hz": (8, 20)}
    return {k: round(float(np.sum(spec[(freqs >= lo) & (freqs < hi)]) / total), 4)
            for k, (lo, hi) in bands.items()}


def stft_log(x: np.ndarray):
    win = np.hanning(N_FFT)
    n = 1 + max(0, (len(x) - N_FFT) // HOP)
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(n)]) if n else np.zeros((0, N_FFT))
    mag = np.abs(np.fft.rfft(frames, axis=1))
    return 20.0 * np.log10(np.maximum(mag, 1e-8)), np.fft.rfftfreq(N_FFT, 1.0 / SR)


def flux(logmag: np.ndarray) -> np.ndarray:
    if len(logmag) < 2:
        return np.array([])
    d = np.diff(logmag, axis=0)
    return np.sqrt(np.mean(d ** 2, axis=1))


def highband_unvoiced(logmag: np.ndarray, freqs: np.ndarray) -> dict:
    """Noise-like frames (spectral flatness proxy) -> share of 4-8k energy."""
    if not len(logmag):
        return {}
    mag = 10 ** (logmag / 20.0)
    band = (freqs >= 4000) & (freqs < 8000)
    low = (freqs >= 100) & (freqs < 8000)
    e_hi = np.sum(mag[:, band] ** 2, axis=1)
    e_all = np.sum(mag[:, low] ** 2, axis=1) + 1e-20
    flat = np.exp(np.mean(np.log(np.maximum(mag[:, low], 1e-12)), axis=1)) / \
        np.maximum(np.mean(mag[:, low], axis=1), 1e-12)
    level = 10 * np.log10(e_all + 1e-20)
    sel = (flat > 0.35) & (level > level.max() - 35)
    return {"noise_frames": int(sel.sum()),
            "high_share_noise": round(float(np.sum(e_hi[sel]) / max(np.sum(e_all[sel]), 1e-20)), 4)}


def gap_structure(env: np.ndarray, gate_db: float = 35.0) -> dict:
    gate = env.max() - gate_db
    low = env < gate
    edges = np.diff(np.r_[False, low, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    gaps = [(e - s) * STEP_S for s, e in zip(starts, ends) if (e - s) * STEP_S >= 0.05]
    return {"gap_count": len(gaps), "gap_total_s": round(float(sum(gaps)), 3),
            "gap_longest_s": round(float(max(gaps) if gaps else 0.0), 3)}


def window_flux(logmag: np.ndarray, times: np.ndarray, f: np.ndarray, start: float, end: float) -> float:
    sel = (times >= start) & (times < end)
    return float(np.median(f[sel])) if sel.any() else float("nan")


results = []
for sid in SAMPLES:
    x = load(sid, "original")
    a = load(sid, "a2p")
    v = load(sid, "vocoder_only")
    n = load(sid, "normal")
    m = min(len(x), len(a), len(v), len(n))
    x, a, v, n = x[:m], a[:m], v[:m], n[:m]
    ex, ea, ev, en = env_db(x), env_db(a), env_db(v), env_db(n)

    lag_a, c0_a, c_a = best_shift(ex, ea)
    lag_v, c0_v, c_v = best_shift(ex, ev)
    ea_s = np.roll(ea, lag_a)
    ev_s = np.roll(ev, lag_v)

    la, fq = stft_log(x)
    aa, _ = stft_log(a)
    va, _ = stft_log(v)
    if lag_a:
        aa = np.roll(aa, int(round(lag_a * STEP_S * SR / HOP)), axis=0)
    if lag_v:
        va = np.roll(va, int(round(lag_v * STEP_S * SR / HOP)), axis=0)
    times = (np.arange(1, len(la)) * HOP + N_FFT / 2) / SR
    fl_a = flux(la)
    fl_p = flux(aa)
    fl_v = flux(va)
    mm = min(len(times), len(fl_a), len(fl_p), len(fl_v))

    wins = []
    for start in np.arange(0, max(0.1, len(times) * HOP / SR - WIN_S), WIN_STEP):
        end = start + WIN_S
        sel = (times[:mm] >= start) & (times[:mm] < end)
        if sel.sum() < 20:
            continue
        fa = float(np.median(fl_a[:mm][sel]))
        fp = float(np.median(fl_p[:mm][sel]))
        fv = float(np.median(fl_v[:mm][sel]))
        wins.append({"start_s": round(float(start), 2), "end_s": round(float(end), 2),
                     "orig_flux": round(fa, 3), "a2p_flux": round(fp, 3),
                     "voc_flux": round(fv, 3),
                     "a2p_vs_orig_db": round(float(20 * np.log10(max(fp, 1e-6) / max(fa, 1e-6))), 2),
                     "voc_vs_orig_db": round(float(20 * np.log10(max(fv, 1e-6) / max(fa, 1e-6))), 2)})
    wins_sorted = sorted(wins, key=lambda w: w["a2p_vs_orig_db"])
    results.append({
        "sample_id": sid, "duration_s": round(len(x) / SR, 2),
        "envelope": {
            "a2p": {"lag_ms": round(lag_a * STEP_S * 1000, 1), "corr_before": round(c0_a, 4),
                    "corr_after": round(c_a, 4)},
            "vocoder_only": {"lag_ms": round(lag_v * STEP_S * 1000, 1), "corr_before": round(c0_v, 4),
                             "corr_after": round(c_v, 4)},
        },
        "modulation": {"original": modulation_bands(ex), "a2p": modulation_bands(ea_s),
                       "vocoder_only": modulation_bands(ev_s), "normal": modulation_bands(en)},
        "flux_median": {
            "original": round(float(np.median(fl_a[:mm])), 3),
            "a2p": round(float(np.median(fl_p[:mm])), 3),
            "vocoder_only": round(float(np.median(fl_v[:mm])), 3),
        },
        "gaps": {"original": gap_structure(ex), "a2p": gap_structure(ea_s),
                 "vocoder_only": gap_structure(ev_s), "normal": gap_structure(en)},
        "noise_highband": {"original": highband_unvoiced(la, fq),
                           "a2p": highband_unvoiced(aa, fq),
                           "vocoder_only": highband_unvoiced(va, fq)},
        "lowest_flux_windows": wins_sorted[:5],
        "highest_flux_windows": wins_sorted[-3:],
        "windows": wins,
    })

(OUT / "metrics.json").write_text(json.dumps({
    "source": "F1_neuralsvb_8songs_2026-11-01_v001",
    "method": "10 ms RMS envelope; 1024/256 STFT log-magnitude flux; best envelope lag; "
              "modulation bands 0.5-2/2-8/8-20 Hz; gaps >=50 ms below max-35 dB; "
              "noise-like frames (flatness>0.35) 4-8k share. Descriptive only.",
    "results": results,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for r in results:
    print(f"--- {r['sample_id']} ({r['duration_s']}s)")
    print("envelope", r["envelope"])
    print("modulation", r["modulation"])
    print("flux_median", r["flux_median"])
    print("gaps", r["gaps"])
    print("noise_highband", r["noise_highband"])
    print("lowest flux windows", r["lowest_flux_windows"][:3])
    print("highest flux windows", r["highest_flux_windows"])
