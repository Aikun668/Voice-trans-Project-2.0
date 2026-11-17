"""F1 成品链 v002（2026-11-02）：V3 = 自适应匹配 EQ（向录音室男声音色靠）+ 压缩 + 去底噪/空旷。

依据 demo v001 反馈：4 首都选 V1 或未答，问题一致为"空旷/回音/底噪"；V2 能量链"完全就是录音机播放出来的感觉"。
诊断（见 README）：
- WPE 去混响（nara_wpe）在常规参数下只去掉 <1 dB → 主因不是长混响；短延迟参数会吃掉歌声本身（响段 −12~−18 dB），不可用。
- 长时平均谱：本人手机录音 125–315 Hz 比录音室人声多 8–10 dB（闷/箱体声），2–3 kHz 少约 18 dB；V1 还在 180 Hz +3 dB。
  "录音机感"与该频谱形状一致（低中频箱体共振 + 缺存在感）。
V3：
1. 目标音色 = 4 位男歌手原唱人声轨（Demucs 分离，朴树/王力宏/许嵩/李玉刚）的响段平均谱；每条输入算自己的平均谱，
   差值 1/3 倍频程平滑、取 MATCH 比例，250 Hz 以下只允许提升不允许削减（基频区）、限幅 ±MAX_DB，160 Hz 以下与 6.3 kHz 以上（各一个倍频程内）渐变到 0，避免抬低频轰鸣（手机录音高频几乎无内容，避免抬噪）；
   线性相位 FIR，零延迟。
2. 压缩：门限 = 响段 P95 − 14 dB，3:1，再 2:1 轻压；无谐波激励、无混响、无双叠。
3. 安静段复原（沿用 v001）+ 句间底噪下压：原音中低于响段 32 dB 的部分再压 12 dB（30 ms 平滑），只作用于句间。
V1 原样保留作为上一轮被选中的对照。用 .venv-f1-adaptive 运行；音高粗检用 ../F1_demo_chain_2026-11-02_v001/pitch_check.py 的方法。
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Gain, HighpassFilter, Pedalboard
from scipy.signal import firwin2, oaconvolve, resample_poly, stft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1DIR = ROOT / "experiments/F1_demo_chain_2026-11-02_v001"
sys.path.insert(0, str(V1DIR))
from run import chain_v1, ebur, frame_db, master, restore_quiet, sha16  # noqa: E402

REF_VOCALS = [ROOT / "test_dataset/reference/vocals" / n for n in (
    "平凡之路_朴树_vocals.wav", "我们的歌_王力宏_vocals.wav", "雅俗共赏_许嵩_vocals.wav", "刚好遇见你_李玉刚_录音室_vocals.wav")]
SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
SONGS = {
    "pingfan": SELF / "平凡之路薄音版.wav",
    "womende": SELF / "我们的歌-薄音版.wav",
    "yasu": SELF / "雅俗共赏-清唱版.wav",
    "gyjyn": SELF / "因为刚好遇见你-清唱版.wav",
}
MATCH = 0.7
MAX_DB = 9.0
HPF_HZ = 80.0
F0_GUARD_HZ = 250.0
LO_HZ, HI_HZ = 160.0, 6300.0
CENTRES = 1000.0 * 2.0 ** (np.arange(-14, 15) / 3.0)      # 39 Hz – 25 kHz, 1/3 倍频程
FIR_TAPS = 4097
FLOOR_REL_DB = -32.0
FLOOR_CUT_DB = 12.0


def band_spectrum(x: np.ndarray, sr: int) -> np.ndarray:
    """响段（帧能量高于 P95−15 dB）的 1/3 倍频程平均谱，按总能量归一（dB）。"""
    if sr != 48000:
        x = resample_poly(x, 48000, sr)
        sr = 48000
    f, _, z = stft(x, sr, nperseg=4096, noverlap=3072)
    p = np.abs(z) ** 2
    e = 10 * np.log10(p.sum(0) + 1e-20)
    p = p[:, e > np.percentile(e, 95) - 15].mean(1)
    bands = np.array([p[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() for c in CENTRES])
    db = 10 * np.log10(bands + 1e-20)
    return db - 10 * np.log10(np.sum(10 ** (db / 10)))


def target_spectrum() -> np.ndarray:
    specs = []
    for p in REF_VOCALS:
        x, sr = sf.read(str(p), dtype="float64", always_2d=True)
        specs.append(band_spectrum(x.mean(1), sr))
    return np.mean(specs, axis=0)


def match_curve(src_db: np.ndarray, tgt_db: np.ndarray) -> np.ndarray:
    d = tgt_db - src_db
    d = np.convolve(np.pad(d, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")   # 相邻 1/3 倍频程平滑
    w = np.clip(np.log2(CENTRES / LO_HZ) + 1, 0, 1) * np.clip(np.log2(HI_HZ / CENTRES) + 1, 0, 1)
    c = np.clip(MATCH * d, -MAX_DB, MAX_DB) * w                                   # 整体响度由最后母带统一
    # 250 Hz 以下是本人基频区（F0 136–226 Hz）；录音室参考在此能量少主要因为音区更高，不能据此削减
    return np.where((CENTRES < F0_GUARD_HZ) & (c < 0), 0.0, c)


def match_eq(x: np.ndarray, sr: int, curve_db: np.ndarray) -> np.ndarray:
    nyq = sr / 2
    freqs = np.concatenate([[0.0], CENTRES[CENTRES < nyq], [nyq]])
    gains = np.concatenate([[curve_db[0]], curve_db[CENTRES < nyq], [0.0]])
    h = firwin2(FIR_TAPS, freqs / nyq, 10 ** (gains / 20))
    return oaconvolve(x, h, mode="same").astype(np.float32)


def floor_cut(orig: np.ndarray, proc: np.ndarray, sr: int, hop_s: float = 0.005, smooth_s: float = 0.03) -> np.ndarray:
    """原音中低于门限的句间部分额外下压 FLOOR_CUT_DB；门限上方 4 dB 内渐变。"""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    hop = int(sr * hop_s)
    e = 10 * np.log10(uniform_filter1d(orig.astype(np.float64) ** 2, size=int(0.02 * sr))[::hop] + 1e-12)
    rel = e - np.percentile(e, 95)
    # 门限随录音底噪自适应：底噪（P5）高于 -37 dB 的录音（如雅俗共赏 -28 dB）门限跟着上移，最高 -20 dB
    thr = float(np.clip(np.percentile(rel, 5) + 5.0, FLOOR_REL_DB, -20.0))
    rel = maximum_filter1d(rel, size=int(0.08 / hop_s))                     # 80 ms 保持：不切字尾
    w = np.clip((thr + 4.0 - rel) / 4.0, 0.0, 1.0)
    g_db = uniform_filter1d(-FLOOR_CUT_DB * w, size=int(smooth_s / hop_s))
    t = np.arange(len(proc)) / sr
    return (proc * 10 ** (np.interp(t, np.arange(len(g_db)) * hop_s, g_db) / 20)).astype(np.float32)


def chain_v3(x: np.ndarray, sr: int, tgt_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curve = match_curve(band_spectrum(x, sr), tgt_db)
    y = match_eq(Pedalboard([HighpassFilter(cutoff_frequency_hz=HPF_HZ)])(x, sr), sr, curve)
    db = frame_db(y, sr)
    p95 = float(np.percentile(db[db > db.max() - 40], 95))
    y = Pedalboard([
        Compressor(threshold_db=p95 - 14.0, ratio=3.0, attack_ms=8.0, release_ms=100.0),
        Gain(gain_db=6.0),
        Compressor(threshold_db=p95 - 6.0, ratio=2.0, attack_ms=15.0, release_ms=120.0),
    ])(y, sr)
    return floor_cut(x, restore_quiet(x, y, sr), sr), curve


def process(x: np.ndarray, sr: int, tgt_db: np.ndarray) -> dict:
    v3, curve = chain_v3(x, sr, tgt_db)
    return {"orig": x, "v1": restore_quiet(x, chain_v1()(x, sr), sr), "v3": v3, "_curve": curve}


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"输出目录已存在, 拒绝覆盖: {out}")
    out.mkdir()
    tgt = target_spectrum()
    report, curves = [], {}
    for key, src in SONGS.items():
        h0 = sha16(src)
        x, sr = sf.read(str(src), dtype="float32")
        res = process(x, sr, tgt)
        curves[key] = np.round(res.pop("_curve"), 2).tolist()
        for name, y in res.items():
            m, gain, _, _ = master(y, sr)
            p = out / f"{key}_{name}.wav"
            sf.write(str(p), m, sr, subtype="PCM_16")
            back, _ = sf.read(str(p), dtype="float32")
            lufs, tp = ebur(p)
            report.append({"song": key, "version": name, "file": p.name, "source": str(src.relative_to(ROOT)),
                           "source_sha16": h0, "samples_match": len(back) == len(x),
                           "clipped_samples": int(np.sum(np.abs(back) >= 0.9999)), "lufs": lufs, "true_peak_dbtp": tp,
                           "spectrum_vs_target_rms_db": round(float(np.sqrt(np.mean(
                               ((band_spectrum(back, sr) - tgt)[(CENTRES >= 100) & (CENTRES <= 6300)]) ** 2))), 2)})
        assert sha16(src) == h0
    (HERE / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "match_curves.json").write_text(json.dumps({"centres_hz": np.round(CENTRES, 1).tolist(),
                                                        "target_db": np.round(tgt, 2).tolist(), "curves_db": curves,
                                                        "match": MATCH, "max_db": MAX_DB}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    for r in report:
        print(f"{r['file']:16s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} clip {r['clipped_samples']} "
              f"len_ok {r['samples_match']}  与录音室音色差(100–6300Hz RMS) {r['spectrum_vs_target_rms_db']:5.2f} dB")
    print("curves (dB @", [int(c) for c in CENTRES[::3]], ")")
    for k, c in curves.items():
        print(k, c[::3])


if __name__ == "__main__":
    main()
