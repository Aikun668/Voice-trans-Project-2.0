"""Read-only diagnosis of the "nasal" colour in the 8 NeuralSVB a2p outputs.

只读既有音频，不生成候选。对比 原音 / a2p / 仅声码器 / 同曲正常版，定位：
1) a2p 相对原音的长期频谱倾斜（<0.5k、0.5-1.5k、1.5-4k、4-8k）；
2) 倾斜是否整条稳定，还是随时间变化（2 s 窗，0.5 s 步进）；
3) 倾斜来自声码器还是声学模型（a2p vs vocoder_only）；
4) 用既有 colour_match 估计“如果做输出校准，会加什么 EQ”（不应用）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "F1_neuralsvb_pitch_target_v001"))
import colour_match as cm  # noqa: E402  (既有输出校准模块，仅估计曲线，不应用)

SRC = ROOT / "experiments" / "F1_neuralsvb_8songs_2026-11-01_v001"
AUDIO = SRC / "listening" / "audio"
OUT = ROOT / "experiments" / "F1_neuralsvb_nasal_diag_2026-11-01_v001" / "run-003"
if OUT.exists():
    raise SystemExit(f"{OUT} already exists")
OUT.mkdir(parents=True)

SR = 22050
N_FFT, HOP = 2048, 512
BANDS = {"low_0_500": (0, 500), "mid1_500_1500": (500, 1500),
         "mid2_1500_4000": (1500, 4000), "high_4000_8000": (4000, 8000)}
REF_BAND = "mid2_1500_4000"
WIN_S, STEP_S = 2.0, 0.5

REPORT = json.loads((SRC / "report.json").read_text(encoding="utf-8"))
SAMPLES = {s["sample_id"]: s for s in REPORT["samples"]}
PAIRS = [("A01", "A02"), ("A03", "A04"), ("A05", "A06"), ("A07", "A08")]
PAIR_OF = {thin: normal for thin, normal in PAIRS}


def load(name: str) -> np.ndarray:
    x, sr = sf.read(str(AUDIO / name), dtype="float64")
    if x.ndim > 1:
        raise ValueError(name)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def stft_power(x: np.ndarray):
    win = np.hanning(N_FFT)
    n = 1 + max(0, (len(x) - N_FFT) // HOP)
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(n)]) if n else np.zeros((0, N_FFT))
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2, np.fft.rfftfreq(N_FFT, 1.0 / SR)


def band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float, sel: np.ndarray) -> float:
    band = (freqs >= lo) & (freqs < hi)
    return float(power[np.ix_(sel, band)].sum()) if sel.any() else 0.0


def rel_bands(power: np.ndarray, freqs: np.ndarray, sel: np.ndarray):
    vals = {k: band_power(power, freqs, lo, hi, sel) for k, (lo, hi) in BANDS.items()}
    ref = max(vals[REF_BAND], 1e-20)
    return {k: float(10.0 * np.log10(max(v, 1e-20) / ref)) for k, v in vals.items()}


def ltas(signal_x: np.ndarray):
    power, freqs = stft_power(signal_x)
    level = np.array([band_power(power, freqs, 100, 4000, np.array([i])) for i in range(len(power))])
    level_db = 10.0 * np.log10(np.maximum(level, 1e-20))
    active = level_db > level_db.max() - 30.0
    return rel_bands(power, freqs, active), int(active.sum()), power, freqs, active


def delta(a: dict, b: dict) -> dict:
    return {k: round(a[k] - b[k], 2) for k in BANDS}


def windows(x_orig: np.ndarray, ref_power, freqs, active, others: dict):
    n_frames = len(ref_power)
    step = int(round(STEP_S * SR / HOP))
    win = int(round(WIN_S * SR / HOP))
    rows = []
    for start in range(0, max(1, n_frames - win + 1), step):
        idx = np.arange(start, min(start + win, n_frames))
        sel = idx[active[idx]]
        if len(sel) < 10:
            continue
        ref = rel_bands(ref_power, freqs, sel)
        row = {"start_s": round(start * HOP / SR, 2), "end_s": round(min(start + win, n_frames) * HOP / SR, 2)}
        for name, (power, _freqs) in others.items():
            out = rel_bands(power, freqs, sel)
            d = {k: round(out[k] - ref[k], 2) for k in BANDS}
            row[name] = {"rel": {k: round(v, 2) for k, v in out.items()}, "delta": d}
            row[name]["mid_emphasis_db"] = round(-(d["low_0_500"] + d["high_4000_8000"]) / 2, 2)
        rows.append(row)
    return rows


results = []
for sid, thin in SAMPLES.items():
    nid = PAIR_OF.get(sid)
    r = {
        "sample_id": sid, "song": thin["song"], "variant": thin["variant"],
        "normal_reference": nid,
        "duration_s": round(thin["input_stats"]["duration_seconds"], 2),
        "ltas": {}, "deltas": {}, "windows": [],
    }
    orig = load(f"{sid}_original.wav")
    a2p = load(f"{sid}_a2p.wav")
    voc = load(f"{sid}_vocoder_only.wav")
    norm = load(f"{nid}_original.wav") if nid else orig
    m = min(len(orig), len(a2p), len(voc), len(norm))
    orig, a2p, voc, norm = orig[:m], a2p[:m], voc[:m], norm[:m]

    ref_rel, n_active, ref_power, freqs, active = ltas(orig)
    a_rel, _, a_power, _, _ = ltas(a2p)
    v_rel, _, v_power, _, _ = ltas(voc)
    n_rel, _, n_power, _, _ = ltas(norm)
    r["ltas"] = {"original": {k: round(v, 2) for k, v in ref_rel.items()},
                 "a2p": {k: round(v, 2) for k, v in a_rel.items()},
                 "vocoder_only": {k: round(v, 2) for k, v in v_rel.items()},
                 "normal_reference": {k: round(v, 2) for k, v in n_rel.items()},
                 "active_frames": n_active}
    r["deltas"]["a2p_vs_original"] = delta(a_rel, ref_rel)
    r["deltas"]["vocoder_vs_original"] = delta(v_rel, ref_rel)
    r["deltas"]["a2p_vs_vocoder"] = delta(a_rel, v_rel)
    r["deltas"]["normal_vs_original"] = delta(n_rel, ref_rel)
    r["mid_emphasis_a2p"] = round(-(r["deltas"]["a2p_vs_original"]["low_0_500"]
                                    + r["deltas"]["a2p_vs_original"]["high_4000_8000"]) / 2, 2)
    r["mid_emphasis_vocoder"] = round(-(r["deltas"]["vocoder_vs_original"]["low_0_500"]
                                        + r["deltas"]["vocoder_vs_original"]["high_4000_8000"]) / 2, 2)

    # 既有输出校准模块估计的 EQ 曲线（只估计，不应用）
    _freqs, gain = cm.match_gain_db(orig, a2p, SR, cm.ColourConfig(strength=1.0))
    r["colour_match_estimate"] = {
        f"{lo}-{hi}": round(float(gain[(_freqs >= lo) & (_freqs < hi)].mean()), 2)
        for lo, hi in [(0, 500), (500, 1500), (1500, 4000), (4000, 8000)]}
    r["colour_match_estimate"]["max_abs_db"] = round(float(np.abs(gain).max()), 2)

    r["windows"] = windows(orig, ref_power, freqs, active,
                           {"a2p": (a_power, freqs), "vocoder_only": (v_power, freqs)})
    scores = [w["a2p"]["mid_emphasis_db"] for w in r["windows"]]
    r["window_score_stats"] = {
        "mean": round(float(np.mean(scores)), 2) if scores else None,
        "std": round(float(np.std(scores)), 2) if scores else None,
        "min": round(float(np.min(scores)), 2) if scores else None,
        "max": round(float(np.max(scores)), 2) if scores else None,
    }
    top = sorted(r["windows"], key=lambda w: w["a2p"]["mid_emphasis_db"], reverse=True)[:3]
    r["top_mid_emphasis_windows"] = [
        {"start_s": w["start_s"], "end_s": w["end_s"],
         "mid_emphasis_db": w["a2p"]["mid_emphasis_db"],
         "low_delta": w["a2p"]["delta"]["low_0_500"], "high_delta": w["a2p"]["delta"]["high_4000_8000"],
         "mid1_delta": w["a2p"]["delta"]["mid1_500_1500"]}
        for w in top]
    results.append(r)

(OUT / "metrics.json").write_text(json.dumps({
    "source_experiment": "F1_neuralsvb_8songs_2026-11-01_v001",
    "method": "LTAS over active STFT frames, bands relative to 1.5-4 kHz; deltas are dB; "
              "mid_emphasis = -(low_delta + high_delta)/2 (positive = lows/highs weaker = mid emphasized)",
    "band_edges": {k: list(v) for k, v in BANDS.items()},
    "window": {"seconds": WIN_S, "step_seconds": STEP_S},
    "results": results,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for r in results:
    d = r["deltas"]["a2p_vs_original"]
    v = r["deltas"]["vocoder_vs_original"]
    print(f"{r['sample_id']} {r['song']}·{r['variant']} | "
          f"a2p low {d['low_0_500']:+.1f} mid1 {d['mid1_500_1500']:+.1f} high {d['high_4000_8000']:+.1f} dB | "
          f"voc low {v['low_0_500']:+.1f} mid1 {v['mid1_500_1500']:+.1f} high {v['high_4000_8000']:+.1f} dB | "
          f"mid-emphasis a2p {r['mid_emphasis_a2p']:+.1f} voc {r['mid_emphasis_vocoder']:+.1f} | "
          f"windows mean {r['window_score_stats']['mean']:+.1f} std {r['window_score_stats']['std']:.1f} | "
          f"EQ {r['colour_match_estimate']}")
