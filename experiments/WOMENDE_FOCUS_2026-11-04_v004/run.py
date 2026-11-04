"""《我们的歌》薄音版 v004：v001 候选2 之后只补 2–3 kHz。

① 原音·同样响        master
② 上次有力那版       v001 thin_keyup (+310) → harmonic_tilt（斜率锁报告值）→ master
③ 有力 + 补中高频    同② → presence EQ（2500 Hz +6 dB，Q=1.4）→ master
④ 有力 + 更强补中高  同② → presence EQ（2500 Hz +12 dB，Q=1.4）→ master

原因 A（ROADMAP §3.1）：2–3 kHz 少约 18 dB。守护：EQ 不削 250 Hz 以下。
不用 chain_dry / decay_guard。不混响、不双叠、不硬吸附、不谐波激励。
F1 不改音高。audio/ 已存在则拒绝覆盖。听感不在这里下结论。
"""
from __future__ import annotations

import importlib.util
import json
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from pedalboard import Pedalboard, PeakFilter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v001"
V2_RUN = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py"
V1_CHAIN = ROOT / "experiments/F1_demo_chain_2026-11-02_v001/run.py"
KEYUP_WAV = V1 / "work/thin_keyup.wav"
V3_REPLAY = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v003/audio/op_replay.wav"
AUDIO = HERE / "audio"
TARGET_LUFS = -13.0
TP_CEIL = -1.0
ALPHA_TOL_DB = 1.0
F0_EQ_MAX_CENTS = 5.0
LOW_HZ = 250.0
LOW_MIN_DB = -0.25
LOW_MAX_DB = 0.50
CURVE_MIN_DB = -0.25
# 原因 A 的 2–3 kHz 缺口约 18 dB。③ 取约 1/3，④ 取约 2/3。Q=1.4 的钟形两侧就是 1–2 / 3–4 kHz 过渡。
PRESENCE = {
    "moderate": [{"hz": 2500.0, "gain_db": 6.0, "q": 1.4, "role": "2–3 kHz 主抬升"}],
    "strong": [{"hz": 2500.0, "gain_db": 12.0, "q": 1.4, "role": "2–3 kHz 主抬升，增益为③的两倍"}],
}
CURVE_HZ = (100, 200, 250, 500, 1000, 1500, 1600, 2000, 2500, 3000, 3500, 3600, 4000, 8000)
BANDS = ((125, 200), (50, 250), (1000, 2000), (2000, 3000), (3000, 4000))


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


def make_board(bands: list[dict]) -> Pedalboard:
    plugs = []
    for b in bands:
        hz, gain, q = float(b["hz"]), float(b["gain_db"]), float(b["q"])
        if hz < LOW_HZ or gain <= 0.0 or q <= 0.0:
            raise SystemExit(f"presence 参数越界: {b}")
        plugs.append(PeakFilter(cutoff_frequency_hz=hz, gain_db=gain, q=q))
    return Pedalboard(plugs)


def apply_presence(x: np.ndarray, sr: int, bands: list[dict], n: int) -> np.ndarray:
    y = np.asarray(make_board(bands)(np.asarray(x, dtype=np.float32), sr), dtype=np.float64).reshape(-1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def magnitude(bands: list[dict], sr: int, n: int = 65536) -> tuple[np.ndarray, np.ndarray, int]:
    board = make_board(bands)
    impulse = np.zeros(n, dtype=np.float32)
    impulse[0] = 1.0
    y = np.asarray(board(impulse, sr), dtype=np.float64).reshape(-1)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = y[:n]
    delay = int(np.argmax(np.abs(y)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    gain = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(y)), 1e-12))
    return freqs, gain, delay


def at_hz(freqs: np.ndarray, gain: np.ndarray, hz: float) -> float:
    return float(np.interp(hz, freqs, gain))


def db3(value: float) -> float:
    rounded = round(float(value), 3)
    return 0.0 if abs(rounded) < 0.0005 else rounded


def tone_gain_db(bands: list[dict], sr: int, freq: float) -> float:
    n = sr
    t = np.arange(n) / sr
    x = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    y = np.asarray(make_board(bands)(x, sr), dtype=np.float64).reshape(-1)[:n]
    a, b = int(0.1 * sr), int(0.9 * sr)
    num = float(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-18)
    den = float(np.sqrt(np.mean(np.asarray(x[a:b], dtype=np.float64) ** 2)) + 1e-18)
    return float(20.0 * np.log10(num / den))


def curve_report(name: str, bands: list[dict], sr: int) -> dict:
    freqs, gain, delay = magnitude(bands, sr)
    samples = {str(hz): db3(at_hz(freqs, gain, hz)) for hz in CURVE_HZ}
    low = gain[freqs <= LOW_HZ]
    set_gain = float(bands[0]["gain_db"])
    g2500 = at_hz(freqs, gain, 2500.0)
    failures = []
    if delay > 48:
        failures.append(f"均衡器延迟 {delay} 样本")
    if float(np.min(low)) < LOW_MIN_DB:
        failures.append(f"250 Hz 以下有削减 {float(np.min(low)):.3f} dB")
    if float(np.max(low)) > LOW_MAX_DB:
        failures.append(f"250 Hz 以下有抬升 {float(np.max(low)):.3f} dB")
    if float(np.min(gain)) < CURVE_MIN_DB:
        failures.append(f"曲线有下切 {float(np.min(gain)):.3f} dB")
    if abs(g2500 - set_gain) > 0.35:
        failures.append(f"2500 Hz 实测 {g2500:.2f} dB，设定 {set_gain:.1f} dB")
    if g2500 < at_hz(freqs, gain, 1500.0) + 2.5 or g2500 < at_hz(freqs, gain, 4000.0) + 2.5:
        failures.append("2–3 kHz 不是曲线的主峰")
    tone = {str(hz): round(tone_gain_db(bands, sr, hz), 3) for hz in (200.0, 2500.0)}
    if abs(tone["200.0"] - samples["200"]) > 0.4 or tone["200.0"] < LOW_MIN_DB:
        failures.append(f"200 Hz 正弦增益 {tone}")
    if abs(tone["2500.0"] - set_gain) > 0.4:
        failures.append(f"2500 Hz 正弦增益 {tone['2500.0']}")
    return {
        "name": name,
        "bands": bands,
        "sample_rate": sr,
        "impulse_delay_samples": delay,
        "curve_db": samples,
        "below_250_min_db": db3(float(np.min(low))),
        "below_250_max_db": db3(float(np.max(low))),
        "curve_min_db": db3(float(np.min(gain))),
        "sine_gain_db": tone,
        "failures": failures,
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


def peak_dbfs(x: np.ndarray) -> float:
    return round(float(20.0 * np.log10(float(np.max(np.abs(x))) + 1e-12)), 2)


def master_round(v1, x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """先走 v001 master。它把前视限幅的内部天花板固定在 -1.3 dBTP。

    ④ 的峰值更高，响度会停在大约 -13.2，进不了它自己的 ±0.15 窗口，但真峰值仍低于 -1。
    这时改用同一支 tp_limit，只把内部天花板从 -1.3 放到不超过 -1.05，让响度进入 ±0.15。
    """
    try:
        y, gain, lufs, tp = v1.master(x, sr)
        info = {"path": "v001.master", "internal_ceil_dbtp": round(v1.TP_CEIL - 0.3, 2),
                "master_gain_db": round(float(gain), 3), "internal_lufs": lufs, "internal_true_peak_dbtp": tp}
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
                info = {
                    "path": "v001.tp_limit",
                    "internal_ceil_dbtp": ceil,
                    "master_gain_db": round(float(g), 3),
                    "internal_lufs": lufs,
                    "internal_true_peak_dbtp": tp,
                    "stock_master_error": stock_error,
                    "reason": "v001.master 的内部天花板 -1.3 dBTP 把响度卡在目标外 0.2 LU 以内；同一支前视限幅放宽到该值后仍 ≤ -1 dBTP",
                }
                return y, info
            g += TARGET_LUFS - lufs
        gain = g
    raise RuntimeError(stock_error)


def render_readme(report: dict) -> str:
    rows = {r["id"]: r for r in report["versions"]}
    eq = report["presence_eq"]
    c3, c4 = eq["moderate"], eq["strong"]
    b3, b4 = c3["bands"][0], c4["bands"][0]
    lines = [
        "# 我们的歌 · 薄音版第 4 轮（2026-11-04 v004）",
        "",
        "v003 的听感：②③④力度分不出来；③还是挺空，和②差不多；④不嗲，问题仍是空旷。",
        "本轮不再把 A 链当空旷的药。在 v001 候选2（升约 310 音分 + 补谐波）后面，只做 2500 Hz 的 presence 峰值。",
        f"听感：**pending_jade**。工程：**technical_pass = {str(report['technical_pass']).lower()}**。Agent 不能听音。",
        "",
        "没有做老师 demo。P0 没有停。",
        "",
        "## 四个版本",
        "",
        "| 页面上的名字 | 文件 | 处理 |",
        "|---|---|---|",
        "| ① 原音·同样响 | `audio/anchor.wav` | 只增益 + 限幅到约 -13 LUFS |",
        "| ② 上次有力那版 | `audio/op_replay.wav` | v001 的 +310 音分文件 → 同一补谐波 → 母带。没有 presence EQ |",
        f"| ③ 有力 + 补中高频 | `audio/presence.wav` | 与②相同 → 2500 Hz {b3['gain_db']:+.0f} dB，Q={b3['q']} → 母带 |",
        f"| ④ 有力 + 更强补中高 | `audio/presence_strong.wav` | 与②相同 → 2500 Hz {b4['gain_db']:+.0f} dB，Q={b4['q']} → 母带 |",
        "",
        "升调是音区开关 F2-K，产品默认关。① 关，②③④ 开 +310 音分。补谐波和 presence EQ 不改音高。",
        "没有 chain_dry、decay_guard、混响、双叠、硬吸附、谐波激励。",
        "",
        "试听页：`index.html`。",
        "",
        "## ③④相对②改了什么",
        "",
        f"③ 在②的声音上多了一个峰值均衡：{b3['hz']:.0f} Hz、{b3['gain_db']:+.0f} dB、Q={b3['q']}。",
        "没有第二段削减，也没有 250 Hz 以下的滤波器。Q=1.4 的钟形自己带出过渡：",
        f"实测大约 2000 Hz {c3['curve_db']['2000']:+.2f} dB、2500 Hz {c3['curve_db']['2500']:+.2f} dB、3000 Hz {c3['curve_db']['3000']:+.2f} dB，",
        f"1500 Hz {c3['curve_db']['1500']:+.2f} dB、4000 Hz {c3['curve_db']['4000']:+.2f} dB。",
        f"250 Hz 以下最高 {c3['below_250_max_db']:+.3f} dB，最低 {c3['below_250_min_db']:+.3f} dB。",
        "",
        f"④ 频率和 Q 不变，增益改成 {b4['gain_db']:+.0f} dB。",
        f"实测大约 2000 Hz {c4['curve_db']['2000']:+.2f} dB、2500 Hz {c4['curve_db']['2500']:+.2f} dB、3000 Hz {c4['curve_db']['3000']:+.2f} dB，",
        f"1500 Hz {c4['curve_db']['1500']:+.2f} dB、4000 Hz {c4['curve_db']['4000']:+.2f} dB。",
        f"250 Hz 以下最高 {c4['below_250_max_db']:+.3f} dB，最低 {c4['below_250_min_db']:+.3f} dB。",
        "",
        "上面这些分贝是均衡器自己的曲线，发生在母带之前。",
        "母带再把四版拉到同一响度，所以成片里各频段的绝对电平还会一起挪，不会正好等于曲线上的数字。",
        "2–3 kHz 相对 250 Hz 以下的比例，见下面的测量。这只说明频谱变了，不说明听起来更好。",
        "",
        "§3.1 写手机录音的 2–3 kHz 比录音室人声少约 18 dB。",
        "+6 dB 大约是这个缺口的三分之一，+12 dB 大约是三分之二，用来听两档，不是把频谱配平到录音室。",
        "",
        "## 和 v001 候选2 的参数对照",
        "",
        "补谐波没有重新求解。斜率、上限、窗长都从 v001 的脚本和 `report.json` 读出来用。presence EQ 加在补谐波之后。",
        "",
        "| 项 | v001 候选2 | 本轮 ②③④ |",
        "|---|---|---|",
        "| 模块 | `harmonic_tilt.py` `process` | 同一个函数 |",
        f"| 升调 | `shift_uniform` +310 音分，`work/thin_keyup.wav` | 直接读这个文件。相关系数 {report['v001_c2_lock']['shift_lock']['correlation_vs_fresh_shift']} |",
        f"| 斜率 | {report['v001_c2_lock']['slope_db_per_oct']} dB/倍频程 | 同一值，来自 v001 `report.json` 的 `c2` |",
        f"| 5 kHz 增益上限 | {report['v001_c2_lock']['max_gain_db']} dB | {report['v001_c2_lock']['max_gain_db']} dB |",
        "| 谐波窗 | σ = 0.12·F0，700 Hz 起，5 kHz 封顶，10 kHz 收回 | 模块常量未改 |",
        f"| alpha ratio | {report['v001_c2_lock']['alpha']['v001_c2_alpha_db']} dB | 补谐波输出、加 EQ 之前 {report['v001_c2_lock']['alpha']['replay_310_alpha_db']} dB，差 {report['v001_c2_lock']['alpha']['delta_db']} dB |",
        "",
        f"和 v003 的②（`op_replay.wav`）相关系数 {report['v003_replay_lock']['correlation']:.6f}。",
        "母带增益如果因响度计略有差别，相关系数仍然只看形状。",
        "",
        "## 工程结果",
        "",
        f"源 SHA-256 `{report['source']['sha256']}`，处理前后相同。{report['source']['sample_rate']} Hz，{report['source']['samples']} 样本。",
        "",
        "| 版本 | LUFS | 真峰值 dBTP | 中位音分 | 浊音重合 | 相对②的 2–3 kHz 减 50–250 Hz | 通过 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("anchor", "op_replay", "presence", "presence_strong"):
        r = rows[key]
        ratio = r["band_delta_vs_replay_db"]["2000-3000_minus_50-250"]
        lines.append(
            f"| {r['name']} | {r['lufs']:.1f} | {r['true_peak_dbtp']:.1f} | "
            f"{r['pitch_vs_source']['median_cents']:+.2f} | {r['pitch_vs_source']['voiced_overlap']:.3f} | "
            f"{ratio:+.2f} dB | {'是' if r['technical_pass'] else '否'} |"
        )
    lines += [
        "",
        f"presence 之前到之后，F0 中位变化：③ {report['presence_f0_vs_tilt']['moderate']['median_cents']} 音分，"
        f"④ {report['presence_f0_vs_tilt']['strong']['median_cents']} 音分。"
        f"门槛是 ±{F0_EQ_MAX_CENTS:.0f} 音分。",
        "",
        "成片相对②：③ 的 2–3 kHz "
        + f"{rows['presence']['band_delta_vs_replay_db']['2000-3000']:+.2f} dB，50–250 Hz "
        + f"{rows['presence']['band_delta_vs_replay_db']['50-250']:+.2f} dB；④ 的 2–3 kHz "
        + f"{rows['presence_strong']['band_delta_vs_replay_db']['2000-3000']:+.2f} dB，50–250 Hz "
        + f"{rows['presence_strong']['band_delta_vs_replay_db']['50-250']:+.2f} dB。",
        "低段这点下降来自整段被拉回同一响度，不是 250 Hz 以下的削减。均衡器曲线在 250 Hz 以下没有下切。",
        "安静段相对响段（v001 `level_stats` 的 floor 抬升，相对原音）："
        + "、".join(
            f"{rows[key]['name']} {rows[key]['floor_lift_db']:+.2f} dB"
            for key in ("anchor", "op_replay", "presence", "presence_strong")
        )
        + "。④ 更高，和它的峰值更吃限幅一致。这不是 A 链的字尾指标，本轮没有用那条链。",
        "",
        "句间降噪没做。本轮也不使用 A 链的字尾守护。",
        "",
        "母带先调用 v001 的 `master`（前视真峰值限幅，内部天花板 -1.3 dBTP）。",
        "④ 的峰值更高，这个内部天花板会把响度留在大约 -13.2 LUFS，进不了它自己的 ±0.15 窗口。",
        "④ 改用同一支 `tp_limit`，内部天花板放到 "
        + str(rows["presence_strong"]["master"]["internal_ceil_dbtp"])
        + " dBTP。成片的真峰值仍用 ffmpeg 读回，门槛仍是 ≤ -1 dBTP。①②③ 没有改这个天花板。",
        "",
        "## 复用",
        "",
        "| 部件 | 地址 | 版本 / 许可证 | 本轮用的接口 |",
        "|---|---|---|---|",
        "| Pedalboard PeakFilter | https://github.com/spotify/pedalboard | "
        f"{report['reuse']['pedalboard']} / GPL-3.0 | `PeakFilter` + `Pedalboard` |",
        "| Parselmouth | https://github.com/YannickJadoul/Parselmouth | "
        f"{report['reuse']['parselmouth']} / GPL-3.0 | Praat `to_pitch`；升调仍走 `f2/poc_phase1.shift_uniform` |",
        "| 补谐波 | `experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py` | 仓库内 | `process` / `alpha_ratio`，斜率锁 v001 c2 |",
        "| 母带 | `experiments/F1_demo_chain_2026-11-02_v001/run.py` | 仓库内 | `master` / `tp_limit` / `ebur` |",
        "| 测量辅助 | `experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py` | 仓库内 | 音高预检、`pitch_delta`、哈希、长度对齐 |",
        "",
        "自写部分只是把这些接口串起来，加上 250 Hz 以下不得削减的曲线检查，以及 F0 中位是否被 EQ 带走的检查。",
        "",
        "## 重跑",
        "",
        "`audio/` 已存在时脚本拒绝覆盖。依赖见 `requirements.txt`。不要装进已有的 `.venv-*`。",
        "",
        "```bash",
        "python3 -m venv .venv-womende-v004",
        ".venv-womende-v004/bin/pip install -r experiments/WOMENDE_FOCUS_2026-11-04_v004/requirements.txt",
        ".venv-womende-v004/bin/python experiments/WOMENDE_FOCUS_2026-11-04_v004/run.py",
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
    for token in ("6 分贝", "12 分贝", "2500", "有力 + 补中高频", "有力 + 更强补中高", "原音·同样响", "上次有力那版", "生成答案"):
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
        raise SystemExit(f"补谐波参数没有锁上 v001 候选2: slope={slope} gain5k={gain_at_5k} constants_ok={constants_ok}")

    hash_before = v2.sha256_file(v2.SRC)
    x, sr = v2.read_wav_pcm16(v2.SRC)
    n = len(x)
    if sr != 48000:
        raise SystemExit(f"采样率应为 48000，实际 {sr}")

    print("presence curve")
    curves = {
        "moderate": curve_report("moderate", PRESENCE["moderate"], sr),
        "strong": curve_report("strong", PRESENCE["strong"], sr),
    }
    if curves["strong"]["curve_db"]["2500"] < curves["moderate"]["curve_db"]["2500"] + 5.0:
        raise SystemExit("④ 的 2500 Hz 增益没有明显高于③")
    curve_fail = [f"{name}: {row['failures']}" for name, row in curves.items() if row["failures"]]
    if curve_fail:
        raise SystemExit("presence 曲线不通过: " + " | ".join(curve_fail))
    print(json.dumps({k: {"curve": v["curve_db"], "low_max": v["below_250_max_db"]} for k, v in curves.items()}, ensure_ascii=False))

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
        raise SystemExit("重新升 310 音分与 v001 thin_keyup.wav 对不上，拒绝继续")

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
        "stage": "harmonic_tilt output, before presence EQ",
    }
    print("alpha lock", alpha_lock)
    if abs(alpha310 - alpha_v001) > ALPHA_TOL_DB:
        raise SystemExit(f"补谐波 alpha {alpha310} 与 v001 候选2 {alpha_v001} 差超过 {ALPHA_TOL_DB} dB")

    print("presence")
    pres3 = apply_presence(tilt310, sr, PRESENCE["moderate"], n)
    pres4 = apply_presence(tilt310, sr, PRESENCE["strong"], n)
    f0_eq = {
        "moderate": median_shift_cents(tilt310, pres3, sr),
        "strong": median_shift_cents(tilt310, pres4, sr),
        "max_abs_cents": F0_EQ_MAX_CENTS,
    }
    print("presence f0", f0_eq)
    for name, row in f0_eq.items():
        if name == "max_abs_cents":
            continue
        if row["median_cents"] is None or abs(row["median_cents"]) > F0_EQ_MAX_CENTS or row["voiced_overlap"] < v2.OVERLAP_MIN:
            raise SystemExit(f"presence EQ 改变了 F0: {name} {row}")

    print("master")
    mastered = {}
    master_info = {}
    for key, src in (
        ("anchor", x),
        ("op_replay", tilt310),
        ("presence", pres3),
        ("presence_strong", pres4),
    ):
        y, info = master_round(v1, v2.as_f32(src, n), sr)
        mastered[key] = y
        info["pre_master_peak_dbfs"] = peak_dbfs(src)
        info["limited_peak_dbfs"] = peak_dbfs(y)
        master_info[key] = info
        print(key, info["path"], info["internal_ceil_dbtp"], info.get("internal_lufs"), info.get("internal_true_peak_dbtp"))

    versions = [
        {"id": "anchor", "file": "anchor.wav", "name": "原音·同样响", "f2k_cents": 0.0,
         "chain": "master_only", "uses_tilt": False, "presence": None},
        {"id": "op_replay", "file": "op_replay.wav", "name": "上次有力那版", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → master",
         "uses_tilt": True, "presence": None},
        {"id": "presence", "file": "presence.wav", "name": "有力 + 补中高频", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → presence_eq moderate → master",
         "uses_tilt": True, "presence": "moderate"},
        {"id": "presence_strong", "file": "presence_strong.wav", "name": "有力 + 更强补中高", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → presence_eq strong → master",
         "uses_tilt": True, "presence": "strong"},
    ]

    AUDIO.mkdir()
    x32 = v2.as_f32(x, n)
    st_src = v1.level_stats(x32, sr)
    written = {}
    rows = []
    for spec in versions:
        y = v2.as_f32(mastered[spec["id"]], n)
        path = AUDIO / spec["file"]
        sf.write(str(path), y, sr, subtype="PCM_16")
        back, sr_back = sf.read(str(path), dtype="float32")
        written[spec["id"]] = back
        lufs, tp = v1.ebur(path)
        clipped = int(np.sum(np.abs(back) >= 0.9999))
        finite = bool(np.isfinite(back).all())
        delta = v2.pitch_delta(v2.SRC, path, spec["f2k_cents"])
        st = v1.level_stats(back, sr)
        floor_lift = (st["floor_p05_all"] - st["p95"]) - (st_src["floor_p05_all"] - st_src["p95"])
        tail_p10 = (st["p10"] - st["p95"]) - (st_src["p10"] - st_src["p95"])
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
        if spec["presence"] is not None:
            moved = f0_eq[spec["presence"]]["median_cents"]
            if moved is None or abs(moved) > F0_EQ_MAX_CENTS:
                failures.append(f"presence 改变 F0 中位 {moved} 音分")
        row = {
            "id": spec["id"], "file": spec["file"], "name": spec["name"],
            "f2k_cents": spec["f2k_cents"], "f2k_switch": "off" if spec["f2k_cents"] == 0 else "on",
            "chain": spec["chain"], "uses_harmonic_tilt": spec["uses_tilt"],
            "uses_a_chain": False, "uses_presence_eq": spec["presence"] is not None,
            "presence_setting": spec["presence"],
            "presence_bands": PRESENCE[spec["presence"]] if spec["presence"] else [],
            "samples": int(len(back)), "samples_match": len(back) == n, "sample_rate": int(sr_back),
            "finite": finite, "clipped_samples": clipped, "lufs": lufs, "true_peak_dbtp": tp,
            "lufs_meter": "ffmpeg ebur128",
            "master": master_info[spec["id"]],
            "floor_lift_db": round(float(floor_lift), 2),
            "tail_p10_lift_db": round(float(tail_p10), 2),
            "band_power_db": bands,
            "pitch_vs_source": delta, "sha256": v2.sha256_file(path),
            "technical_pass": not failures, "failures": failures, "listening": "pending_jade",
        }
        rows.append(row)
        print(f"{spec['file']:22s} LUFS {lufs:6.2f} TP {tp:5.2f} cents {delta['median_cents']} "
              f"overlap {delta['voiced_overlap']} pass {not failures} {failures}")

    replay = written["op_replay"]
    for row, audio in zip(rows, (written[r["id"]] for r in rows)):
        bands = row["band_power_db"]
        ref = rows[1]["band_power_db"]
        delta_bands = {k: round(bands[k] - ref[k], 3) for k in bands}
        ratio = bands["2000-3000"] - bands["50-250"]
        ref_ratio = ref["2000-3000"] - ref["50-250"]
        delta_bands["2000-3000_minus_50-250"] = round(ratio - ref_ratio, 3)
        row["band_delta_vs_replay_db"] = delta_bands
        row["correlation_vs_replay"] = round(corr(audio, replay), 6)
        if row["id"] == "presence" and delta_bands["2000-3000_minus_50-250"] < 2.0:
            row["failures"].append("2–3 kHz 相对低频没有比②抬起来")
        if row["id"] == "presence_strong":
            mild = next(r for r in rows if r["id"] == "presence")
            if delta_bands["2000-3000_minus_50-250"] < mild["band_delta_vs_replay_db"]["2000-3000_minus_50-250"] + 2.0:
                row["failures"].append("④ 的 2–3 kHz 比例没有明显高于③")
        if row["id"] in ("presence", "presence_strong") and row["correlation_vs_replay"] > 0.999:
            row["failures"].append("和②几乎一样，presence 没有起作用")
        row["technical_pass"] = not row["failures"]

    v3_audio, v3_sr = sf.read(str(V3_REPLAY), dtype="float32")
    v3_lock = {
        "file": str(V3_REPLAY.relative_to(ROOT)),
        "sample_rate": int(v3_sr),
        "samples": int(len(v3_audio)),
        "correlation": round(corr(written["op_replay"], v3_audio), 6),
    }
    print("v003 replay lock", v3_lock)
    if v3_sr != sr or len(v3_audio) != n or v3_lock["correlation"] < 0.99:
        for row in rows:
            if row["id"] == "op_replay":
                row["failures"].append(f"与 v003 ② 不一致 corr={v3_lock['correlation']}")
                row["technical_pass"] = False

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
            "cause": "ROADMAP §3.1 原因A：手机录音相对录音室人声，125–200 Hz 多 7–10 dB，2–3 kHz 少约 18 dB",
            "guard": "EQ 不削 250 Hz 以下。本轮滤波器中心都在 2500 Hz，增益为正。",
            "not_the_lever": "不用 v003 的 chain_dry / decay_guard 当空旷主药",
            "partial_step_db": {"moderate_over_about_18db": 6.0, "strong_over_about_18db": 12.0},
        },
        "v001_c2_lock": {
            "module": "experiments/WOMENDE_FOCUS_2026-11-03_v001/harmonic_tilt.py",
            "report": "experiments/WOMENDE_FOCUS_2026-11-03_v001/report.json",
            "slope_db_per_oct": slope,
            "max_gain_db": gain_at_5k,
            "key_up_cents": 310,
            "f_start_hz": ht.F_START,
            "f_full_hz": ht.F_FULL,
            "f_fade_hz": ht.F_FADE,
            "sigma_f0": ht.SIGMA,
            "nfft": ht.NFFT,
            "hop": ht.HOP,
            "alpha": alpha_lock,
            "shift_lock": shift_lock,
            "note": "②③④ 的升调直接用 v001 work/thin_keyup.wav。斜率用 v001 report 的 c2，不重新求解。alpha 在 presence EQ 之前核对。",
        },
        "v003_replay_lock": v3_lock,
        "presence_eq": curves,
        "presence_f0_vs_tilt": f0_eq,
        "pitch_precheck": pre,
        "not_applied": [
            "chain_dry", "decay_guard", "reverb", "double_track", "hard_pitch_snap",
            "harmonic_exciter", "cut_below_250_hz",
        ],
        "f1_does_not_change_pitch": True,
        "f2k_switch": "① 关。②③④ +310 音分（v001 文件）。产品默认仍是关。presence EQ 不改音高。",
        "targets": {
            "lufs": TARGET_LUFS,
            "true_peak_dbtp_max": TP_CEIL,
            "lufs_abs_tolerance": v2.LUFS_TOL,
            "shift_error_cents_max": v2.SHIFT_TOL_CENTS,
            "voiced_overlap_min": v2.OVERLAP_MIN,
            "presence_f0_median_abs_cents_max": F0_EQ_MAX_CENTS,
            "alpha_abs_delta_max_db": ALPHA_TOL_DB,
            "below_250_hz_gain_db": [LOW_MIN_DB, LOW_MAX_DB],
        },
        "ffmpeg": v2.ffmpeg_version(),
        "reuse": {
            "pedalboard": pkg_version("pedalboard"),
            "parselmouth": pkg_version("praat-parselmouth"),
            "numpy": pkg_version("numpy"),
            "scipy": pkg_version("scipy"),
            "soundfile": pkg_version("soundfile"),
            "pedalboard_project": "https://github.com/spotify/pedalboard",
            "pedalboard_license": "GPL-3.0",
            "parselmouth_project": "https://github.com/YannickJadoul/Parselmouth",
            "parselmouth_license": "GPL-3.0",
            "interface": "pedalboard.PeakFilter; F1 v001 master/tp_limit/ebur; v001 harmonic_tilt.process",
        },
        "versions": rows,
        "listening_material_s": round(4 * n / sr, 1),
    }
    (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "README.md").write_text(render_readme(report), encoding="utf-8")
    print("technical_pass", report["technical_pass"])
    if not report["technical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
