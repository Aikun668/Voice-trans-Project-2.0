"""《我们的歌》薄音版 v003：接回 v001 候选2（升调 + 补谐波），再叠 A 链。

① 原音·同样响          master
② 上次有力那版         F2-K +310 → v001 harmonic_tilt（斜率锁报告值）→ master
③ 上次有力那版+收字尾  同上 → chain_dry → master
④ 再高一点             F2-K +400 → 同一补谐波斜率 → chain_dry → master

F1 不改音高。不混响、不双叠、不硬吸附、不谐波激励。
audio/ 已存在则拒绝覆盖。听感不在这里下结论。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v001"
V2_RUN = ROOT / "experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py"
KEYUP_WAV = V1 / "work/thin_keyup.wav"
AUDIO = HERE / "audio"
TARGET_LUFS = -13.0
TP_CEIL = -1.0
DECAY_LIFT_MAX_DB = 3.0
ALPHA_TOL_DB = 1.0


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


def main() -> None:
    if AUDIO.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {AUDIO}")
    v2 = load_module("womende_v002", V2_RUN)
    ht = load_module("harmonic_tilt_v001", V1 / "harmonic_tilt.py")
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
    sys.path.insert(0, str(ROOT / "f2"))
    from poc_phase1 import read_wav_pcm16, shift_uniform, to_array  # noqa: E402
    x, sr = read_wav_pcm16(v2.SRC)
    n = len(x)
    pre = v2.precheck(x, sr)
    print("pitch precheck", json.dumps(pre, ensure_ascii=False))
    if not pre["ok"]:
        raise SystemExit(2)

    fresh310 = v2.fit_len(to_array(shift_uniform(parselmouth.Sound(x, sr), 310.0)), n)
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

    k400 = v2.fit_len(to_array(shift_uniform(parselmouth.Sound(x, sr), 400.0)), n)
    t310, f310 = f0_of(k310, sr)
    t400, f400 = f0_of(k400, sr)
    print("tilt +310")
    tilt310 = v2.fit_len(ht.process(np.asarray(k310, dtype=np.float64), sr, t310, f310, slope), n)
    print("tilt +400")
    tilt400 = v2.fit_len(ht.process(np.asarray(k400, dtype=np.float64), sr, t400, f400, slope), n)
    alpha310 = round(ht.alpha_ratio(tilt310, sr, t310, f310), 2)
    alpha400 = round(ht.alpha_ratio(tilt400, sr, t400, f400), 2)
    alpha_v001 = float(locked["alpha_db"])
    alpha_lock = {
        "v001_c2_alpha_db": alpha_v001,
        "replay_310_alpha_db": alpha310,
        "delta_db": round(alpha310 - alpha_v001, 2),
        "k400_same_slope_alpha_db": alpha400,
        "tolerance_db": ALPHA_TOL_DB,
    }
    print("alpha lock", alpha_lock)

    v3 = v2.load_v003_chain()
    v1 = v3._v1
    if abs(float(v3.DECAY_ALLOW_DB) - DECAY_LIFT_MAX_DB) > 1e-9:
        raise SystemExit(f"A 链字尾允许值是 {v3.DECAY_ALLOW_DB}，不是 {DECAY_LIFT_MAX_DB}")

    print("chains")
    anchor = v1.master(v2.as_f32(x, n), sr)[0]
    replay = v1.master(v2.as_f32(tilt310, n), sr)[0]
    with_a = v1.master(v3.chain_dry(v2.as_f32(tilt310, n), sr), sr)[0]
    limit = v1.master(v3.chain_dry(v2.as_f32(tilt400, n), sr), sr)[0]

    versions = [
        {"id": "anchor", "file": "anchor.wav", "name": "原音·同样响", "f2k_cents": 0.0,
         "chain": "master_only", "uses_tilt": False, "uses_a": False, "audio": anchor},
        {"id": "op_replay", "file": "op_replay.wav", "name": "上次有力那版", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → master",
         "uses_tilt": True, "uses_a": False, "audio": replay},
        {"id": "op_a", "file": "op_a.wav", "name": "上次有力那版 + 收字尾", "f2k_cents": 310.0,
         "chain": "v001 thin_keyup → harmonic_tilt(slope locked) → chain_dry → master",
         "uses_tilt": True, "uses_a": True, "audio": with_a},
        {"id": "k400_tilt_a", "file": "k400_tilt_a.wav", "name": "再高一点", "f2k_cents": 400.0,
         "chain": "shift_uniform +400 → harmonic_tilt(same slope) → chain_dry → master",
         "uses_tilt": True, "uses_a": True, "audio": limit},
    ]

    AUDIO.mkdir()
    x32 = v2.as_f32(x, n)
    st_src = v1.level_stats(x32, sr)
    rows = []
    alpha_fail = abs(alpha310 - alpha_v001) > ALPHA_TOL_DB
    for spec in versions:
        y = v2.as_f32(spec["audio"], n)
        path = AUDIO / spec["file"]
        sf.write(str(path), y, sr, subtype="PCM_16")
        back, sr_back = sf.read(str(path), dtype="float32")
        lufs, tp = v1.ebur(path)
        clipped = int(np.sum(np.abs(back) >= 0.9999))
        finite = bool(np.isfinite(back).all())
        lift = v2.decay_lift_raw(x32, back, sr, v3)
        loud = v3.lift_stats(x32, back, sr)
        delta = v2.pitch_delta(v2.SRC, path, spec["f2k_cents"])
        st = v1.level_stats(back, sr)
        floor_lift = (st["floor_p05_all"] - st["p95"]) - (st_src["floor_p05_all"] - st_src["p95"])
        tail_p10 = (st["p10"] - st["p95"]) - (st_src["p10"] - st_src["p95"])
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
        if spec["uses_a"] and lift > DECAY_LIFT_MAX_DB:
            failures.append(f"字尾抬升 {lift:.2f} dB")
        if spec["uses_tilt"] and alpha_fail:
            failures.append(f"补谐波 alpha {alpha310} 与 v001 候选2 {alpha_v001} 差超过 {ALPHA_TOL_DB} dB")
        row = {
            "id": spec["id"], "file": spec["file"], "name": spec["name"],
            "f2k_cents": spec["f2k_cents"], "f2k_switch": "off" if spec["f2k_cents"] == 0 else "on",
            "chain": spec["chain"], "uses_harmonic_tilt": spec["uses_tilt"], "uses_a_chain": spec["uses_a"],
            "samples": int(len(back)), "samples_match": len(back) == n, "sample_rate": int(sr_back),
            "finite": finite, "clipped_samples": clipped, "lufs": lufs, "true_peak_dbtp": tp,
            "lufs_meter": "ffmpeg ebur128", "decay_lift_db": round(lift, 2),
            "loud_lift_db": loud["loud_lift_db"], "floor_lift_db": round(float(floor_lift), 2),
            "tail_p10_lift_db": round(float(tail_p10), 2),
            "sample_identity_outside_correction": "not_applicable",
            "pitch_vs_source": delta, "sha256": v2.sha256_file(path),
            "technical_pass": not failures, "failures": failures, "listening": "pending_jade",
        }
        rows.append(row)
        print(f"{spec['file']:20s} LUFS {lufs:6.2f} TP {tp:5.2f} lift {lift:+5.2f} "
              f"cents {delta['median_cents']} overlap {delta['voiced_overlap']} pass {not failures} {failures}")

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
            "note": "②③ 的升调直接用 v001 work/thin_keyup.wav。斜率用 v001 report 的 c2，不重新求解。",
        },
        "pitch_precheck": pre,
        "floor_cut": {
            "status": "identity_stub",
            "applied": False,
            "reason": "与 v002 相同：F1_demo_chain v002/run.py 不在本快照，chain_dry 的句间降噪未做",
        },
        "not_applied": ["reverb", "double_track", "hard_pitch_snap", "harmonic_exciter"],
        "f1_does_not_change_pitch": True,
        "f2k_switch": "① 关。②③ +310 音分（v001 文件）。④ +400 音分。产品默认仍是关。",
        "targets": {
            "lufs": TARGET_LUFS,
            "true_peak_dbtp_max": TP_CEIL,
            "shift_error_cents_max": v2.SHIFT_TOL_CENTS,
            "voiced_overlap_min": v2.OVERLAP_MIN,
            "decay_lift_db_max_on_a_chain": DECAY_LIFT_MAX_DB,
            "alpha_abs_delta_max_db": ALPHA_TOL_DB,
        },
        "ffmpeg": v2.ffmpeg_version(),
        "versions": rows,
        "listening_material_s": round(4 * n / sr, 1),
    }
    (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["technical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
