"""声源/发声力度相关指标：薄音版、正常版、王力宏原唱对应段（.venv-f2）。
H1−H2：越小越"实"（声带闭合更紧）；H1−A3：越小高次谐波越多（越有力/亮）；
alpha ratio = 1–5 kHz 能量 − 50 Hz–1 kHz 能量：越大越"用力"；CPPS / HNR：越大越"实、不漏气"。
只统计响段浊音帧（相对 P95 −15 dB 以内），避免尾音/气声拉低。"""
import json, sys
from pathlib import Path
import numpy as np
import parselmouth
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "f2"))
from poc_phase1 import read_wav_pcm16  # noqa: E402
call = parselmouth.praat.call
SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
al = json.loads((Path(__file__).parent / "align_ref.json").read_text(encoding="utf-8"))
REF = ROOT / "test_dataset/reference/vocals/我们的歌_王力宏_vocals.wav"
items = {"薄音版": (SELF / "我们的歌-薄音版.wav", None), "正常版": (SELF / "我们的歌正常版.wav", None),
         "王力宏(对应薄音段)": (REF, (al["thin"]["ref_t0"], al["thin"]["ref_t1"])),
         "王力宏(对应正常段)": (REF, (al["normal"]["ref_t0"], al["normal"]["ref_t1"]))}

def harm_db(spec_f, spec_db, f):
    m = (spec_f > f * 0.9) & (spec_f < f * 1.1)
    return spec_db[m].max() if m.any() else np.nan

res = {}
for name, (p, seg) in items.items():
    x, sr = read_wav_pcm16(p)
    if seg: x = x[int(seg[0] * sr): int(seg[1] * sr)]
    s = parselmouth.Sound(x, sr)
    pitch = s.to_pitch(0.01, 75, 600)
    t = pitch.xs(); f0 = pitch.selected_array["frequency"]
    hop = int(0.01 * sr); win = int(0.04 * sr)
    e = np.array([10 * np.log10(np.mean(x[max(0, int(tt * sr) - win // 2): int(tt * sr) + win // 2] ** 2) + 1e-12) for tt in t])
    use = (f0 > 0) & (e > np.percentile(e[f0 > 0], 95) - 15)
    h12, h1a3, alpha = [], [], []
    w = np.hanning(win)
    for tt, f in zip(t[use], f0[use]):
        c = int(tt * sr); fr = x[c - win // 2: c + win // 2]
        if len(fr) < win: continue
        sp = np.abs(np.fft.rfft(fr * w, 8192)); fq = np.fft.rfftfreq(8192, 1 / sr); db = 20 * np.log10(sp + 1e-12)
        H1, H2 = harm_db(fq, db, f), harm_db(fq, db, 2 * f)
        # A3：第三共振峰附近（2.2–3.2 kHz）最强谐波
        m3 = (fq > 2200) & (fq < 3200); A3 = db[m3].max()
        p = sp ** 2
        alpha.append(10 * np.log10(p[(fq >= 1000) & (fq < 5000)].sum() / p[(fq >= 50) & (fq < 1000)].sum()))
        h12.append(H1 - H2); h1a3.append(H1 - A3)
    hnr = call(s.to_harmonicity_cc(0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
    pcg = call(s, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(pcg, "Get CPPS", "no", 0.02, 0.0005, 60, 330, 0.05, "parabolic", 0.001, 0, "Exponential decay", "Robust")
    res[name] = {"F0_median_Hz": round(float(np.median(f0[use])), 1), "H1-H2_dB": round(float(np.median(h12)), 2),
                 "H1-A3_dB": round(float(np.median(h1a3)), 2), "alpha_ratio_dB": round(float(np.median(alpha)), 2),
                 "HNR_dB": round(hnr, 2), "CPPS_dB": round(cpps, 2), "frames": int(len(h12))}
    print(name, res[name])
(Path(__file__).parent / "source_measures.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
