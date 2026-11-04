"""《我们的歌》薄音版 v005：同一条 v001 候选2 上，换两个不同的假设。

① 原音·同样响
② 上次有力那版     v001 thin_keyup (+310) → harmonic_tilt（斜率锁报告值）→ master
③ 只补唱出来的中高频  同② → 谐波宽架（800→1600 Hz 升到 +10 dB，持平到 4 kHz，6 kHz 收回；气声不加）→ master
④ 字尾再压低        同② → 衰减段再压 10 dB（不是 A 链）→ master

v004 的 2500 Hz 钟形等响后 2–3 kHz 仍高 4–8 dB，Jade 听不出差别，同时安静段被抬高。
本轮不再做那个钟形，也不把 presence 写成到顶。
不用 chain_dry。不混响、不双叠、不硬吸附、不谐波激励。不削 250 Hz 以下。
F1 不改音高。audio/ 已存在则拒绝覆盖。听感不在这里下结论。
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v001"
V2_RUN = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py"
V1_CHAIN = ROOT / "experiments/F1_demo_chain_2026-11-02_v001/run.py"
KEYUP_WAV = V1 / "work/thin_keyup.wav"
V4_REPLAY = ROOT / "experiments/WOMENDE_FOCUS_2026-11-04_v004/audio/op_replay.wav"
AUDIO = HERE / "audio"
TARGET_LUFS = -13.0
TP_CEIL = -1.0
ALPHA_TOL_DB = 1.0
F0_MOVE_MAX_CENTS = 5.0
LOW_HZ = 250.0
# 宽架：250 Hz 以下增益为 0。800 Hz 起抬，1600 Hz 到顶，4000 Hz 前持平，6000 Hz 收回。
SHELF_LO, SHELF_HI, SHELF_HOLD, SHELF_FADE = 800.0, 1600.0, 4000.0, 6000.0
SHELF_DB = 10.0
TAIL_DEPTH_DB = 10.0
HOP_S = 0.005
TAIL_SMOOTH = 6  # 6 × 5 ms ≈ 30 ms
BANDS = ((125, 200), (50, 250), (400, 600), (800, 1600), (1600, 4000), (2000, 3000), (2400, 3400))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f0_of(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    p = parselmouth.Sound(np.asarray(x, dtype=np.float64), sr).to_pitch(0.01, 75.0, 600.0)
    return np.asarray(p.xs(), dtype=np.float64), np.asarray(p.selected_array["frequency"], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a @ a) * (b @ b)))
    return float((a @ b) / d) if d > 0 else 0.0


def env_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * HOP_S)
    return 10 * np.log10(uniform_filter1d(np.asarray(x, dtype=np.float64) ** 2, size=int(0.02 * sr))[::hop] + 1e-12)


def decay_mask(orig: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """与 v003 相同的字尾定义：相对响段 −4 dB 以下，且下降快于 40 dB/s。"""
    e = env_db(orig, sr)
    rel = e - np.percentile(e, 95)
    slope = np.gradient(uniform_filter1d(e, 6)) / HOP_S
    return (slope < -40.0) & (rel < -4.0), rel


def decay_lift(orig: np.ndarray, proc: np.ndarray, sr: int) -> float:
    mask, rel_o = decay_mask(orig, sr)
    env = env_db(proc, sr)
    n = min(len(env), len(rel_o))
    if int(mask[:n].sum()) < 10:
        raise RuntimeError("字尾衰减段太短")
    delta = (env - np.percentile(env, 95))[:n] - rel_o[:n]
    return float(np.median(delta[mask[:n]]))


def shelf_db(freqs: np.ndarray) -> np.ndarray:
    g = np.zeros_like(freqs, dtype=np.float64)
    rise = (freqs > SHELF_LO) & (freqs < SHELF_HI)
    g[rise] = SHELF_DB * (freqs[rise] - SHELF_LO) / (SHELF_HI - SHELF_LO)
    g[(freqs >= SHELF_HI) & (freqs <= SHELF_HOLD)] = SHELF_DB
    fade = (freqs > SHELF_HOLD) & (freqs < SHELF_FADE)
    g[fade] = SHELF_DB * (SHELF_FADE - freqs[fade]) / (SHELF_FADE - SHELF_HOLD)
    g[freqs < LOW_HZ] = 0.0
    return g


def apply_harmonic_shelf(x: np.ndarray, sr: int, t0: np.ndarray, f0: np.ndarray, ht) -> tuple[np.ndarray, dict]:
    """只在浊音谐波峰上乘宽架。气声、清音、250 Hz 以下保持 1。"""
    freqs, times, Z = stft(
        np.asarray(x, dtype=np.float64), sr, nperseg=ht.NFFT, noverlap=ht.NFFT - ht.HOP,
        window="hann", boundary="even", padded=True,
    )
    fz = np.interp(times, t0, f0, left=0, right=0)
    near = np.interp(times, t0, (f0 > 0).astype(float), left=0, right=0)
    voiced = (near > 0.99) & (fz > 60)
    w = np.convolve(voiced.astype(float), np.ones(5) / 5, mode="same")
    shelf = shelf_db(freqs)
    G = np.ones(Z.shape, dtype=np.float64)
    for k in np.nonzero(w > 0)[0]:
        f = fz[k] if fz[k] > 60 else float(np.max(fz[max(0, k - 3): k + 4]))
        if f <= 60:
            continue
        d = freqs - np.round(freqs / f) * f
        m = np.exp(-0.5 * (d / (ht.SIGMA * f)) ** 2) * (freqs > 0.5 * f)
        G[:, k] = 1.0 + (10 ** (shelf / 20.0) - 1.0) * m * float(w[k])
    G[freqs < LOW_HZ, :] = 1.0
    _, y = istft(Z * G, sr, nperseg=ht.NFFT, noverlap=ht.NFFT - ht.HOP, window="hann", boundary=True)
    y = np.asarray(y, dtype=np.float64)[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    low = G[freqs < LOW_HZ, :]
    info = {
        "shape": "harmonic-only shelf, residual gain 0 dB",
        "lo_hz": SHELF_LO,
        "full_hz": SHELF_HI,
        "hold_hz": SHELF_HOLD,
        "fade_hz": SHELF_FADE,
        "peak_gain_db": SHELF_DB,
        "curve_on_harmonic_peak_db": {
            str(hz): round(float(np.interp(hz, freqs, shelf)), 3)
            for hz in (200, 250, 800, 1200, 1600, 2500, 3000, 4000, 6000, 8000)
        },
        "below_250_max_abs_gain_db": round(float(np.max(np.abs(20.0 * np.log10(np.maximum(low, 1e-12))))), 4),
        "voiced_fraction": round(float(np.mean(voiced)), 3),
    }
    return y, info


def apply_tail_trim(x: np.ndarray, sr: int, depth_db: float) -> tuple[np.ndarray, dict]:
    """衰减段整段再压低。不是均衡，不单独动 250 Hz 以下。也不是 chain_dry。"""
    mask, rel = decay_mask(x, sr)
    g_db = np.zeros_like(rel)
    g_db[mask] = -float(depth_db)
    g_db = uniform_filter1d(g_db, size=TAIL_SMOOTH)
    t = np.arange(len(x)) / sr
    te = np.arange(len(g_db)) * HOP_S
    y = np.asarray(x, dtype=np.float64) * 10 ** (np.interp(t, te, g_db) / 20.0)
    return y, {
        "depth_db": depth_db,
        "smooth_ms": round(TAIL_SMOOTH * HOP_S * 1000.0, 1),
        "decay_frame_fraction": round(float(np.mean(mask)), 3),
        "decay_frames": int(mask.sum()),
        "frames": int(len(mask)),
        "decay_median_rel_db": round(float(np.median(rel[mask])), 2) if mask.any() else None,
        "note": "增益随时间走，全频一起变轻。没有高通，没有 250 Hz 以下的削减曲线。",
    }


def median_shift_cents(ref: np.ndarray, other: np.ndarray, sr: int) -> dict:
    fa = f0_of(ref, sr)[1]
    fb = f0_of(other, sr)[1]
    n = min(len(fa), len(fb))
    voiced = fa[:n] > 0
    both = voiced & (fb[:n] > 0)
    if int(voiced.sum()) < 10 or int(both.sum()) < 10:
        return {"median_cents": None, "voiced_overlap": 0.0, "frames_both": int(both.sum())}
    cents = 1200.0 * np.log2(fb[:n][both] / fa[:n][both])
    return {
        "median_cents": round(float(np.median(cents)), 2),
        "voiced_overlap": round(float(both.sum() / max(int(voiced.sum()), 1)), 4),
        "frames_both": int(both.sum()),
    }


def band_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * w)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (freqs >= lo) & (freqs < hi)
    return float(10.0 * np.log10(float(spec[m].mean()) + 1e-18))


def voiced_band_drop(x: np.ndarray, sr: int, t0: np.ndarray, f0: np.ndarray, ht) -> dict:
    """浊音帧上，400–600 Hz 到 2400–3400 Hz 掉了多少。用来和 Sundberg 的约 6 / 15 dB 对照。"""
    freqs, times, Z = stft(
        np.asarray(x, dtype=np.float64), sr, nperseg=ht.NFFT, noverlap=ht.NFFT - ht.HOP,
        window="hann", boundary="even", padded=True,
    )
    fz = np.interp(times, t0, f0, left=0, right=0)
    near = np.interp(times, t0, (f0 > 0).astype(float), left=0, right=0)
    voiced = (near > 0.99) & (fz > 60)
    mag2 = np.abs(Z) ** 2

    def level(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return 10.0 * np.log10(float(mag2[sel][:, voiced].mean()) + 1e-18)

    low = level(400, 600)
    ring = level(2400, 3400)
    body = level(50, 250)
    # harmonic share in the ring
    harm = np.zeros(int(voiced.sum()), dtype=np.float64)
    res = np.zeros_like(harm)
    sel = (freqs >= 2200) & (freqs < 3600)
    idx = 0
    for k in np.nonzero(voiced)[0]:
        f = fz[k]
        d = freqs - np.round(freqs / f) * f
        m = np.exp(-0.5 * (d / (ht.SIGMA * f)) ** 2) * (freqs > 0.5 * f)
        col = mag2[sel, k]
        mk = m[sel]
        harm[idx] = float((col * mk).sum())
        res[idx] = float((col * (1.0 - mk)).sum())
        idx += 1
    hsum, rsum = float(harm.sum()), float(res.sum())
    return {
        "voiced_400_600_db": round(low, 2),
        "voiced_2400_3400_db": round(ring, 2),
        "voiced_drop_400_to_ring_db": round(low - ring, 2),
        "voiced_drop_50_250_to_ring_db": round(body - ring, 2),
        "ring_2200_3600_harmonic_fraction": round(hsum / (hsum + rsum + 1e-18), 3),
    }


def peak_dbfs(x: np.ndarray) -> float:
    return round(float(20.0 * np.log10(float(np.max(np.abs(x))) + 1e-12)), 2)


def master_round(v1, x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    try:
        y, gain, lufs, tp = v1.master(x, sr)
        info = {
            "path": "v001.master",
            "internal_ceil_dbtp": round(v1.TP_CEIL - 0.3, 2),
            "master_gain_db": round(float(gain), 3),
            "internal_lufs": lufs,
            "internal_true_peak_dbtp": tp,
        }
        return y, info
    except RuntimeError as exc:
        stock_error = str(exc)
    gain = TARGET_LUFS - v1.ebur_array(x, sr)[0]
    for ceil in (-1.25, -1.20, -1.15, -1.10, -1.05):
        g = gain
        for _ in range(8):
            y = v1.tp_limit(x * 10 ** (g / 20.0), sr, ceil)
            lufs, tp = v1.ebur_array(y, sr)
            if tp > TP_CEIL:
                break
            if abs(lufs - TARGET_LUFS) <= 0.15:
                return y, {
                    "path": "v001.tp_limit",
                    "internal_ceil_dbtp": ceil,
                    "master_gain_db": round(float(g), 3),
                    "internal_lufs": lufs,
                    "internal_true_peak_dbtp": tp,
                    "stock_master_error": stock_error,
                    "reason": "v001.master 的内部天花板把响度卡在目标外；同一支前视限幅放宽后仍 ≤ -1 dBTP",
                }
            g += TARGET_LUFS - lufs
        gain = g
    raise RuntimeError(stock_error)


def render_readme(report: dict) -> str:
    rows = {r["id"]: r for r in report["versions"]}
    shelf = report["harmonic_shelf"]
    tail = report["tail_trim"]
    diag = report["diagnosis"]
    lines = [
        "# 我们的歌 · 薄音版第 5 轮（2026-11-04 v005）",
        "",
        "v004 的原话：③④和②差不多，听不出补了什么；依旧空旷，听不出 2234 有啥区别。",
        "那一版的钟形并不是没生效：等响之后 2–3 kHz 仍比②高约 4.2 / 7.9 dB，安静段抬升却从 +2.3 dB 增到 +4.0 / +6.5 dB。",
        "所以这一轮不再做 2500 Hz 钟形，也不把 presence 写成到顶。只换两件不同的事。",
        f"听感：**pending_jade**。工程：**technical_pass = {str(report['technical_pass']).lower()}**。Agent 不能听音。",
        "",
        "没有做老师 demo。P0 没有停。",
        "",
        "## 四个版本",
        "",
        "| 页面上的名字 | 文件 | 处理 |",
        "|---|---|---|",
        "| ① 原音·同样响 | `audio/anchor.wav` | 只增益 + 限幅到约 -13 LUFS |",
        "| ② 上次有力那版 | `audio/op_replay.wav` | v001 的 +310 音分文件 → 同一补谐波 → 母带 |",
        f"| ③ 有力 + 只补唱出来的中高频 | `audio/shelf.wav` | 与②相同 → 谐波宽架 +{SHELF_DB:.0f} dB（{SHELF_LO:.0f}→{SHELF_HI:.0f} Hz 升起，持平到 {SHELF_HOLD:.0f} Hz）→ 母带 |",
        f"| ④ 有力 + 字尾再压低 | `audio/tail.wav` | 与②相同 → 字尾衰减段再压 {TAIL_DEPTH_DB:.0f} dB → 母带 |",
        "",
        "升调是音区开关 F2-K，产品默认关。① 关，②③④ 开 +310 音分。宽架和字尾都不改音高。",
        "没有 chain_dry、混响、双叠、硬吸附、谐波激励。没有 250 Hz 以下的削减。",
        "",
        "试听页：`index.html`。",
        "",
        "## 为什么是这两版",
        "",
        "分开三件事：",
        "",
        "- **Jade 没听出来的**：v004 的②③④。原话是和②差不多，听不出补了什么，空旷依旧。",
        "- **工程上变了的**：钟形曲线在 2500 Hz 是 +6 / +12 dB，250 Hz 以下没有下切。成片等响后，2–3 kHz 仍高约 +4.2 / +7.9 dB。④ 的母带峰值顶到限幅，安静段抬升 +6.5 dB。",
        "- **等响没有把频谱差抹掉**。差还在成片里。被一起带动的是安静段：钟形把该频段里的房间声和谐波一起抬高，限幅再把小声音托起来。ITU-R BS.1770 的 K 计权在 2 kHz 以上有一截抬升，所以这段一响，响度计就变大，拉回 -13 LUFS 时其余部分会一起往下；已经低于门限大约 10 dB 的字尾，响度计基本不算它，压低字尾不会被这次拉回补上。",
        "",
        f"在②的浊音帧上，400–600 Hz 到 2400–3400 Hz 掉了 {diag['c2_voiced']['voiced_drop_400_to_ring_db']} dB；"
        f"50–250 Hz 到同一段掉了 {diag['c2_voiced']['voiced_drop_50_250_to_ring_db']} dB。"
        f"2.2–3.6 kHz 里谐波大约占 {diag['c2_voiced']['ring_2200_3600_harmonic_fraction']:.0%}，不是「那里只有噪声」。",
        f"补谐波的斜率到大约 2.6 kHz 已经顶到 +15 dB 上限。v004 再在 2500 Hz 加钟形，是在已经顶满的谐波上，把气声也一起抬了。",
        "",
        f"③ 换成宽架，而且只乘在谐波峰上：{SHELF_LO:.0f} Hz 开始，{SHELF_HI:.0f} Hz 到 +{SHELF_DB:.0f} dB，"
        f"持平到 {SHELF_HOLD:.0f} Hz，{SHELF_FADE:.0f} Hz 收回。气声增益是 0。250 Hz 以下增益是 0"
        f"（实测最大偏差 {shelf['below_250_max_abs_gain_db']} dB）。",
        "1500 Hz 附近原来的补谐波只有大约 +9 dB，还没到 +15 dB 的顶，所以这一截不是把 v004 的钟形再加一档。",
        "",
        f"④ 不动频谱。v003 的 A 链叠在同一条基线上，字尾抬升是 +2.6 dB，并不低于②的 +2.1 dB：那条链没有把字尾压下去。"
        f"这里沿用 v003 的字尾定义（正在变轻、而且已经比响段低 4 dB 以上），在那些帧上再压 {TAIL_DEPTH_DB:.0f} dB，平滑大约 {tail['smooth_ms']} ms。"
        f"压之前，这段大约占 {tail['decay_frame_fraction']:.0%}，中位比响段低 {abs(tail['decay_median_rel_db']):.0f} dB。"
        "全频一起变轻，没有单独削低频。",
        "",
        "这两版是两个假设：③ 问「只补谐波、不补房间声」能不能听出来、空旷会不会少；④ 问「字尾本身被压低」空旷会不会少，以及字会不会被切掉。",
        "共振峰上移这一次没用：在这条升调后的声音上，第一共振峰中位大约 211 Hz，上移 70 Hz 的增益在 250 Hz 以下会削到大约 −7 dB。那条守护不让破。",
        "",
        "## 文献（只取和这一轮有关的）",
        "",
        "- Sundberg 1974（JASA）：男声歌唱共振峰是大约 2.8 kHz 的一簇（F3–F5 靠拢），不是一个很窄的单峰。https://doi.org/10.1121/1.1914609",
        "- Sundberg 1972（STL-QPSR 13/2–3）：讲话到 3 kHz 常常掉 15 dB 以上，职业歌唱大约只掉 6 dB。本轮浊音帧上 400–600 Hz 到 2.4–3.4 kHz 掉约 11.5 dB，介于两者之间；整段 50–250 Hz 对 2–3 kHz 的大约 18–26 dB，是另一把尺子，不要当成「元音上还缺 18 dB」。",
        "- Sundberg 等 1995（STL-QPSR）：歌唱共振峰是 F3–F5 聚在一起；比预期的第三共振峰高出大约 6 dB，被当作「有没有这簇」的一条下限。v004 的成片差已经到这个量级，Jade 仍听不出，所以本轮不把「再加一档 2.5 kHz」当下一刀。",
        "- ITU-R BS.1770-5：响度是 K 计权再加门限。2 kHz 以上的抬升会把读数抬高，等响时其余频段一起往下；比未加门限响度低 10 dB 以上的块不算进去。字尾压低因此不会被拉回 -13 LUFS 时补上。",
        "- 《Journal of Speech, Language, and Hearing Research》2024，*Validity of Acoustic Measures…Smartphones…*：文中两只手机内置麦在大约 0.1–2.5 kHz 相对平，2.5 kHz 附近有一个小峰，没有近讲低频隆起。https://pubs.asha.org/doi/full/10.1044/2024_JSLHR-23-00759 所以「125–200 Hz 比录音室多」更像是录音室人声做过处理，不是这只手机麦的低音凸起。250 Hz 以下仍然不削。",
        "",
        "## 和 v001 候选2 的参数对照",
        "",
        "补谐波没有重新求解。宽架和字尾都加在补谐波之后。",
        "",
        "| 项 | v001 候选2 | 本轮 ②③④ |",
        "|---|---|---|",
        "| 模块 | `harmonic_tilt.py` `process` | 同一个函数 |",
        f"| 升调 | `shift_uniform` +310 音分 | 直接读 `work/thin_keyup.wav`。相关系数 {report['v001_c2_lock']['shift_lock']['correlation_vs_fresh_shift']} |",
        f"| 斜率 | {report['v001_c2_lock']['slope_db_per_oct']} dB/倍频程 | 同一值 |",
        f"| 5 kHz 增益上限 | {report['v001_c2_lock']['max_gain_db']} dB | 同一值。约 2.6 kHz 起已经顶到这个上限 |",
        f"| alpha ratio | {report['v001_c2_lock']['alpha']['v001_c2_alpha_db']} dB | 补谐波之后、③④之前 {report['v001_c2_lock']['alpha']['replay_310_alpha_db']} dB |",
        "",
        f"和 v004 的②相关系数 {report['v004_replay_lock']['correlation']:.6f}。",
        "",
        "## 工程结果",
        "",
        f"源 SHA-256 `{report['source']['sha256']}`，处理前后相同。{report['source']['sample_rate']} Hz，{report['source']['samples']} 样本。",
        "",
        "| 版本 | LUFS | 真峰值 dBTP | 中位音分 | 浊音重合 | 相对②的 1600–4000 Hz | 字尾抬升 | 通过 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("anchor", "op_replay", "shelf", "tail"):
        r = rows[key]
        lines.append(
            f"| {r['name']} | {r['lufs']:.1f} | {r['true_peak_dbtp']:.1f} | "
            f"{r['pitch_vs_source']['median_cents']:+.2f} | {r['pitch_vs_source']['voiced_overlap']:.3f} | "
            f"{r['band_delta_vs_replay_db']['1600-4000']:+.2f} dB | {r['decay_lift_db']:+.2f} dB | "
            f"{'是' if r['technical_pass'] else '否'} |"
        )
    lines += [
        "",
        f"③ 相对补谐波、进母带之前，F0 中位变化 {report['f0_move_vs_tilt']['shelf']['median_cents']} 音分；"
        f"④ 是 {report['f0_move_vs_tilt']['tail']['median_cents']} 音分。门槛 ±{F0_MOVE_MAX_CENTS:.0f} 音分。",
        "",
        "字尾抬升是相对原音、在 v003 那个衰减段上的中位数。② 没有去压字尾。"
        f"④ 要比②至少低 6 dB，这一轮是 {rows['tail']['decay_lift_db'] - rows['op_replay']['decay_lift_db']:+.2f} dB。",
        "安静段相对抬升（v001 `level_stats`，相对原音）："
        + "、".join(f"{rows[k]['name']} {rows[k]['floor_lift_db']:+.2f} dB" for k in ("anchor", "op_replay", "shelf", "tail"))
        + "。③ 不应当再走出 v004 那种「中高频更高、安静段也更高」。",
        "",
        "母带用 v001 的 `master`（前视真峰值限幅）。真峰值用 ffmpeg ebur128 读回，门槛 ≤ -1 dBTP。",
        "",
        "## 复用",
        "",
        "| 部件 | 地址 | 版本 / 许可证 | 本轮用的接口 |",
        "|---|---|---|---|",
        "| Parselmouth | https://github.com/YannickJadoul/Parselmouth | "
        f"{report['reuse']['parselmouth']} / GPL-3.0 | Praat `to_pitch`；升调仍走 `f2/poc_phase1.shift_uniform` |",
        "| 补谐波 | `experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py` | 仓库内 | `process` / `alpha_ratio`，斜率锁 v001 c2。宽架沿用它的谐波窗，不新造谐波 |",
        "| 字尾定义 | `experiments/F1_demo_chain_2026-11-02_v003/run.py` `decay_mask` | 仓库内 | 同一条阈值。增益改成「再压 10 dB」，不用 `chain_dry` |",
        "| 母带 | `experiments/F1_demo_chain_2026-11-02_v001/run.py` | 仓库内 | `master` / `tp_limit` / `ebur` |",
        "| SciPy STFT | https://scipy.org | "
        f"{report['reuse']['scipy']} / BSD-3-Clause | `stft` / `istft`，只做谐波窗上的增益 |",
        "",
        "自写部分是宽架的增益曲线、250 Hz 以下保持 1 的守护，以及字尾上那一截固定压低。没有新的激励器。",
        "",
        "## 重跑",
        "",
        "`audio/` 已存在时脚本拒绝覆盖。依赖见 `requirements.txt`。不要装进已有的 `.venv-*`。",
        "",
        "```bash",
        "python3 -m venv .venv-womende-v005",
        ".venv-womende-v005/bin/pip install -r experiments/WOMENDE_FOCUS_2026-11-04_v005/requirements.txt",
        ".venv-womende-v005/bin/python experiments/WOMENDE_FOCUS_2026-11-04_v005/run.py",
        "```",
        "",
        "## 听感",
        "",
        "pending_jade。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if AUDIO.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {AUDIO}")
    page = (HERE / "index.html").read_text(encoding="utf-8")
    for token in ("10 分贝", "只补唱出来的中高频", "字尾再压低", "原音·同样响", "上次有力那版", "生成答案", "250 Hz"):
        if token not in page:
            raise SystemExit(f"试听页缺少与参数一致的文字: {token}")

    v2 = load_module("womende_v002", V2_RUN)
    ht = load_module("harmonic_tilt_v001", V1 / "harmonic_tilt.py")
    v1 = load_module("f1_v001_run", V1_CHAIN)
    if not v2.SRC.is_file():
        raise SystemExit(f"输入不存在: {v2.SRC}")
    if not KEYUP_WAV.is_file():
        raise SystemExit(f"v001 升调文件不存在: {KEYUP_WAV}")

    locked = json.loads((V1 / "report.json").read_text(encoding="utf-8"))["c2"]
    slope = float(locked["slope_db_per_oct"])
    constants_ok = (
        ht.F_START == 700.0 and ht.F_FULL == 5000.0 and ht.F_FADE == 10000.0
        and ht.SIGMA == 0.12 and ht.MAX_GAIN_DB == 15.0 and ht.NFFT == 2048 and ht.HOP == 256
        and abs(slope - 7.992) < 1e-9 and abs(float(locked["max_gain_db"]) - 15.0) < 1e-9
        and int(locked["key_up_cents"]) == 310
    )
    gain_at_5k = float(ht.tilt_gain_db(np.array([ht.F_FULL]), slope)[0])
    if not constants_ok or abs(gain_at_5k - 15.0) > 0.05:
        raise SystemExit(f"补谐波参数没有锁上 v001 候选2: slope={slope}")

    hash_before = v2.sha256_file(v2.SRC)
    x, sr = v2.read_wav_pcm16(v2.SRC)
    n = len(x)
    if sr != 48000:
        raise SystemExit(f"采样率应为 48000，实际 {sr}")

    pre = v2.precheck(x, sr)
    print("pitch precheck", json.dumps(pre, ensure_ascii=False))
    if not pre["ok"]:
        raise SystemExit(2)

    fresh310 = v2.fit_len(v2.to_array(v2.shift_uniform(parselmouth.Sound(x, sr), 310.0)), n)
    k310, sr310 = sf.read(str(KEYUP_WAV), dtype="float32")
    if sr310 != sr or len(k310) != n:
        raise SystemExit(f"v001 thin_keyup 尺寸不符: sr={sr310} n={len(k310)}")
    shift_lock = {
        "file": str(KEYUP_WAV.relative_to(ROOT)),
        "max_abs_diff_vs_fresh_shift": round(float(np.max(np.abs(fresh310 - k310))), 6),
        "correlation_vs_fresh_shift": round(corr(fresh310, k310), 6),
    }
    print("shift lock", shift_lock)
    if shift_lock["correlation_vs_fresh_shift"] < 0.999:
        raise SystemExit("重新升 310 音分与 v001 thin_keyup.wav 对不上")

    t310, f310 = f0_of(k310, sr)
    print("tilt +310")
    tilt310 = v2.fit_len(ht.process(np.asarray(k310, dtype=np.float64), sr, t310, f310, slope), n)
    alpha310 = round(ht.alpha_ratio(tilt310, sr, t310, f310), 2)
    alpha_v001 = float(locked["alpha_db"])
    alpha_lock = {
        "v001_c2_alpha_db": alpha_v001,
        "replay_310_alpha_db": alpha310,
        "delta_db": round(alpha310 - alpha_v001, 2),
        "tolerance_db": ALPHA_TOL_DB,
        "stage": "harmonic_tilt output, before shelf / tail",
    }
    print("alpha lock", alpha_lock)
    if abs(alpha310 - alpha_v001) > ALPHA_TOL_DB:
        raise SystemExit(f"补谐波 alpha {alpha310} 与 v001 候选2 {alpha_v001} 差超过 {ALPHA_TOL_DB} dB")

    c2_voiced = voiced_band_drop(tilt310, sr, t310, f310, ht)
    print("c2 voiced", c2_voiced)

    print("shelf + tail")
    shelf_audio, shelf_info = apply_harmonic_shelf(tilt310, sr, t310, f310, ht)
    tail_audio, tail_info = apply_tail_trim(tilt310, sr, TAIL_DEPTH_DB)
    if shelf_info["below_250_max_abs_gain_db"] > 0.05:
        raise SystemExit(f"宽架动了 250 Hz 以下: {shelf_info}")
    if abs(shelf_info["curve_on_harmonic_peak_db"]["250"]) > 0.05:
        raise SystemExit("250 Hz 谐波峰增益不是 0")
    if shelf_info["curve_on_harmonic_peak_db"]["1600"] < SHELF_DB - 0.2:
        raise SystemExit("1600 Hz 没有到设定增益")
    if tail_info["decay_frame_fraction"] < 0.10:
        raise SystemExit(f"字尾太短，压低没有对象: {tail_info}")

    f0_move = {
        "shelf": median_shift_cents(tilt310, shelf_audio, sr),
        "tail": median_shift_cents(tilt310, tail_audio, sr),
        "max_abs_cents": F0_MOVE_MAX_CENTS,
    }
    print("f0 move", f0_move)
    for name, row in f0_move.items():
        if name == "max_abs_cents":
            continue
        # 中位音分必须不动。字尾压低会让最轻的尾音掉出 Praat，重合率以母带之后、相对原音的那一次为准。
        if row["median_cents"] is None or abs(row["median_cents"]) > F0_MOVE_MAX_CENTS:
            raise SystemExit(f"{name} 改变了 F0 中位: {row}")

    print("master")
    mastered, master_info = {}, {}
    for key, src in (
        ("anchor", x),
        ("op_replay", tilt310),
        ("shelf", shelf_audio),
        ("tail", tail_audio),
    ):
        y, info = master_round(v1, v2.as_f32(src, n), sr)
        mastered[key] = y
        info["pre_master_peak_dbfs"] = peak_dbfs(src)
        info["limited_peak_dbfs"] = peak_dbfs(y)
        master_info[key] = info
        print(key, info["path"], info.get("internal_lufs"), info.get("internal_true_peak_dbtp"), "pre", info["pre_master_peak_dbfs"])

    versions = [
        {"id": "anchor", "file": "anchor.wav", "name": "原音·同样响", "f2k_cents": 0.0,
         "chain": "master_only", "uses_tilt": False, "lever": None},
        {"id": "op_replay", "file": "op_replay.wav", "name": "上次有力那版", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → master",
         "uses_tilt": True, "lever": None},
        {"id": "shelf", "file": "shelf.wav", "name": "有力 + 只补唱出来的中高频", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt → harmonic shelf → master",
         "uses_tilt": True, "lever": "shelf"},
        {"id": "tail", "file": "tail.wav", "name": "有力 + 字尾再压低", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt → tail trim → master",
         "uses_tilt": True, "lever": "tail"},
    ]

    x32 = v2.as_f32(x, n)
    st_src = v1.level_stats(x32, sr)
    rows = []
    written = {}
    with tempfile.TemporaryDirectory(prefix="womende_v005_") as tmp:
        tmp_path = Path(tmp)
        for spec in versions:
            y = v2.as_f32(mastered[spec["id"]], n)
            path = tmp_path / spec["file"]
            sf.write(str(path), y, sr, subtype="PCM_16")
            back, sr_back = sf.read(str(path), dtype="float32")
            written[spec["id"]] = back
            lufs, tp = v1.ebur(path)
            clipped = int(np.sum(np.abs(back) >= 0.9999))
            finite = bool(np.isfinite(back).all())
            delta = v2.pitch_delta(v2.SRC, path, spec["f2k_cents"])
            st = v1.level_stats(back, sr)
            floor_lift = (st["floor_p05_all"] - st["p95"]) - (st_src["floor_p05_all"] - st_src["p95"])
            lift = decay_lift(x32, back, sr)
            bands = {f"{lo}-{hi}": round(band_db(back, sr, lo, hi), 3) for lo, hi in BANDS}
            failures = []
            if sr_back != sr or len(back) != n:
                failures.append(f"尺寸 sr={sr_back} n={len(back)}")
            if not finite:
                failures.append("NaN 或 Inf")
            if clipped:
                failures.append(f"削波样本 {clipped}")
            if abs(lufs - TARGET_LUFS) > v2.LUFS_TOL:
                failures.append(f"LUFS {lufs}")
            if tp > TP_CEIL:
                failures.append(f"真峰值 {tp}")
            if delta["median_cents"] is None or abs(delta["median_cents"] - spec["f2k_cents"]) >= v2.SHIFT_TOL_CENTS:
                failures.append(f"中位音分 {delta.get('median_cents')} 目标 {spec['f2k_cents']}")
            if delta["voiced_overlap"] < v2.OVERLAP_MIN:
                failures.append(f"浊音重合 {delta['voiced_overlap']}")
            if spec["lever"] is not None:
                moved = f0_move[spec["lever"]]["median_cents"]
                if moved is None or abs(moved) > F0_MOVE_MAX_CENTS:
                    failures.append(f"杠杆改变 F0 中位 {moved} 音分")
            row = {
                "id": spec["id"], "file": spec["file"], "name": spec["name"],
                "f2k_cents": spec["f2k_cents"], "f2k_switch": "off" if spec["f2k_cents"] == 0 else "on",
                "chain": spec["chain"], "uses_harmonic_tilt": spec["uses_tilt"],
                "uses_a_chain": False, "uses_presence_bell": False, "lever": spec["lever"],
                "samples": int(len(back)), "samples_match": len(back) == n, "sample_rate": int(sr_back),
                "finite": finite, "clipped_samples": clipped, "lufs": lufs, "true_peak_dbtp": tp,
                "lufs_meter": "ffmpeg ebur128",
                "master": master_info[spec["id"]],
                "floor_lift_db": round(float(floor_lift), 2),
                "decay_lift_db": round(float(lift), 2),
                "band_power_db": bands,
                "pitch_vs_source": delta,
                "technical_pass": not failures, "failures": failures, "listening": "pending_jade",
            }
            rows.append(row)
            print(f"{spec['file']:16s} LUFS {lufs:6.2f} TP {tp:5.2f} cents {delta['median_cents']} "
                  f"overlap {delta['voiced_overlap']} decay {lift:+.2f} pass {not failures} {failures}")

        ref = next(r for r in rows if r["id"] == "op_replay")
        for row in rows:
            delta_bands = {k: round(row["band_power_db"][k] - ref["band_power_db"][k], 3) for k in row["band_power_db"]}
            row["band_delta_vs_replay_db"] = delta_bands
            row["correlation_vs_replay"] = round(corr(written[row["id"]], written["op_replay"]), 6)
            row["decay_lift_minus_replay_db"] = round(row["decay_lift_db"] - ref["decay_lift_db"], 2)
        shelf_row = next(r for r in rows if r["id"] == "shelf")
        tail_row = next(r for r in rows if r["id"] == "tail")
        if shelf_row["band_delta_vs_replay_db"]["1600-4000"] < 3.0:
            shelf_row["failures"].append("1600–4000 Hz 没有比②高至少 3 dB")
        if shelf_row["correlation_vs_replay"] > 0.995:
            shelf_row["failures"].append("和②几乎一样，宽架没有起作用")
        if shelf_row["floor_lift_db"] > ref["floor_lift_db"] + 2.5:
            shelf_row["failures"].append(
                f"安静段抬升 {shelf_row['floor_lift_db']} 比②高超过 2.5 dB，又走到 v004 限幅托底"
            )
        if tail_row["decay_lift_minus_replay_db"] > -6.0:
            tail_row["failures"].append(
                f"字尾抬升只比②低 {tail_row['decay_lift_minus_replay_db']} dB，没有压下去"
            )
        if tail_row["correlation_vs_replay"] > 0.999:
            tail_row["failures"].append("和②几乎一样，字尾没有压")
        if corr(written["shelf"], written["tail"]) > 0.995:
            shelf_row["failures"].append("③和④几乎一样")
        for row in rows:
            row["technical_pass"] = not row["failures"]

        v4_audio, v4_sr = sf.read(str(V4_REPLAY), dtype="float32")
        v4_lock = {
            "file": str(V4_REPLAY.relative_to(ROOT)),
            "sample_rate": int(v4_sr),
            "samples": int(len(v4_audio)),
            "correlation": round(corr(written["op_replay"], v4_audio), 6),
        }
        print("v004 replay lock", v4_lock)
        if v4_sr != sr or len(v4_audio) != n or v4_lock["correlation"] < 0.99:
            ref["failures"].append(f"与 v004 ② 不一致 corr={v4_lock['correlation']}")
            ref["technical_pass"] = False

        if v2.sha256_file(v2.SRC) != hash_before:
            raise SystemExit("源文件哈希变了")

        report = {
            "round": HERE.name,
            "listening": "pending_jade",
            "agent_cannot_hear": True,
            "technical_pass": all(r["technical_pass"] for r in rows),
            "source": {
                "path": str(v2.SRC.relative_to(ROOT)),
                "sha256": hash_before,
                "sha16": hash_before[:16],
                "hash_unchanged": True,
                "samples": n,
                "sample_rate": sr,
                "duration_s": round(n / sr, 3),
            },
            "diagnosis": {
                "v004_heard": "和②差不多，听不出补了什么；依旧空旷",
                "v004_measured_after_rematch": "2–3 kHz vs ② about +4.2 dB and +7.9 dB; floor lift +3.98 / +6.48 dB vs anchor's reference",
                "not_exhausted": "一轮两档，不按 §7.5 写成 presence 到顶。本轮换形状，不再加 2500 Hz 钟形。",
                "c2_voiced": c2_voiced,
                "why_not_f1": "F1 中位约 211 Hz。+70 Hz 的增益在 250 Hz 以下会削到约 -7 dB，守护不允许。",
                "why_tail": "v003 A 链的字尾抬升不低于②。这一轮把衰减段再压低，等响门限不会把已经偏低的字尾拉回来。",
            },
            "v001_c2_lock": {
                "module": "experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py",
                "slope_db_per_oct": slope,
                "max_gain_db": gain_at_5k,
                "key_up_cents": 310,
                "alpha": alpha_lock,
                "shift_lock": shift_lock,
            },
            "v004_replay_lock": v4_lock,
            "harmonic_shelf": shelf_info,
            "tail_trim": tail_info,
            "f0_move_vs_tilt": f0_move,
            "pitch_precheck": pre,
            "not_applied": [
                "chain_dry", "presence_bell_2500", "reverb", "double_track", "hard_pitch_snap",
                "harmonic_exciter", "cut_below_250_hz", "formant_shift",
            ],
            "f1_does_not_change_pitch": True,
            "f2k_switch": "① 关。②③④ +310 音分（v001 文件）。产品默认仍是关。宽架和字尾不改音高。",
            "literature": [
                "Sundberg 1974 JASA doi:10.1121/1.1914609",
                "Sundberg 1972 STL-QPSR 13(2-3) 61-63",
                "Sundberg et al. 1995 STL-QPSR, singer's formant revisited",
                "ITU-R BS.1770-5",
                "JSLHR 2024 10.1044/2024_JSLHR-23-00759",
            ],
            "targets": {
                "lufs": TARGET_LUFS,
                "true_peak_dbtp_max": TP_CEIL,
                "lufs_abs_tolerance": v2.LUFS_TOL,
                "shift_error_cents_max": v2.SHIFT_TOL_CENTS,
                "voiced_overlap_min": v2.OVERLAP_MIN,
                "lever_f0_median_abs_cents_max": F0_MOVE_MAX_CENTS,
                "tail_decay_lift_vs_replay_db_max": -6.0,
                "shelf_1600_4000_vs_replay_db_min": 3.0,
            },
            "ffmpeg": v2.ffmpeg_version(),
            "reuse": {
                "parselmouth": pkg_version("praat-parselmouth"),
                "numpy": pkg_version("numpy"),
                "scipy": pkg_version("scipy"),
                "soundfile": pkg_version("soundfile"),
                "pedalboard": pkg_version("pedalboard"),
            },
            "versions": rows,
            "listening_material_s": round(4 * n / sr, 1),
        }
        (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (HERE / "README.md").write_text(render_readme(report), encoding="utf-8")
        print("technical_pass", report["technical_pass"])
        if not report["technical_pass"]:
            raise SystemExit(1)
        AUDIO.mkdir()
        for spec in versions:
            sf.write(str(AUDIO / spec["file"]), written[spec["id"]], sr, subtype="PCM_16")
            row = next(r for r in rows if r["id"] == spec["id"])
            row["sha256"] = v2.sha256_file(AUDIO / spec["file"])
        (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", AUDIO)


if __name__ == "__main__":
    main()
