"""《我们的歌》薄音版定点第 2 轮。

音区开关（F2-K，默认产品态是关；本轮按 §6 打开）与 F1 分开：
  ① 原音·同样响     只增益 + 前视真峰值限幅到 -13 LUFS
  ② K310 + A        +310 音分 → chain_dry → master
  ③ K400 + A        +400 音分 → chain_dry → master
  ④ K310 + 共振峰   +310 音分 → formant +70 Hz → chain_effort（即 A）→ master

F1 不改音高。不混响、不双叠、不硬吸附、不谐波激励、不用 harmonic_tilt。
输出目录 audio/ 已存在则拒绝运行。听感不在本脚本里下结论。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "f2"))
from poc_phase1 import read_wav_pcm16, shift_uniform, to_array, write_wav_pcm16  # noqa: E402

SRC = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav/我们的歌-薄音版.wav"
AUDIO = HERE / "audio"
WORK = HERE / "work"
TARGET_LUFS = -13.0
TP_CEIL = -1.0
PITCH_LIMIT_HZ = 600.0
FORMANT_HZ = 70.0  # §6 给出的 +45~70 Hz 上沿；v003 chain_effort 的 SHIFT_HZ
SHIFT_TOL_CENTS = 5.0
OVERLAP_MIN = 0.95
LUFS_TOL = 0.4
DECAY_LIFT_MAX_DB = 3.0

V3_RUN = ROOT / "experiments/F1_demo_chain_2026-11-02_v003/run.py"
V2_STUB = HERE / "floor_cut_stub.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_v003_chain():
    """按路径加载 v003。v002/run.py 不在本快照，导入行改指到本目录的恒等 floor_cut。"""
    text = V3_RUN.read_text(encoding="utf-8")
    needle = '_v2 = _load("f1_v002_run", ROOT / "experiments/F1_demo_chain_2026-11-02_v002/run.py")'
    if needle not in text:
        raise SystemExit("v003/run.py 的 v002 加载行与预期不符，拒绝猜测 chain_dry")
    text = text.replace(
        needle,
        f'_v2 = _load("f1_v002_stub", Path({str(V2_STUB)!r}))',
    )
    mod = types.ModuleType("f1_v003_chain")
    mod.__dict__["__file__"] = str(V3_RUN)
    mod.__dict__["__name__"] = "f1_v003_chain"
    exec(compile(text, str(V3_RUN), "exec"), mod.__dict__)
    return mod


def fit_len(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2) + 1e-18))


def match_rms(y: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    g = rms(ref) / rms(y)
    return y * g, float(20.0 * np.log10(max(g, 1e-12)))


def as_f32(y: np.ndarray, n: int) -> np.ndarray:
    out = np.ascontiguousarray(fit_len(y, n), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("出现 NaN 或 Inf")
    return out


def pitch_hz(x: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    return parselmouth.Sound(np.asarray(x, dtype=np.float64), sr).to_pitch(
        0.01, 75.0, ceiling
    ).selected_array["frequency"]


def track_stats(f: np.ndarray) -> dict:
    v = f[f > 0]
    if v.size == 0:
        raise RuntimeError("Praat 没有浊音帧")
    return {
        "voiced_frames": int(v.size),
        "median_hz": round(float(np.median(v)), 2),
        "p95_hz": round(float(np.percentile(v, 95)), 2),
        "p99_hz": round(float(np.percentile(v, 99)), 2),
        "max_hz": round(float(np.max(v)), 2),
    }


def exceed_count(f: np.ndarray, cents: float, limit: float) -> tuple[int, int, int]:
    factor = 2.0 ** (cents / 1200.0)
    voiced = f > 0
    over = voiced & (f * factor > limit)
    longest = run = 0
    for flag in over:
        if flag:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return int(over.sum()), int(longest), int(voiced.sum())


def precheck(x: np.ndarray, sr: int) -> dict:
    tracks = {}
    for ceiling in (600.0, 800.0):
        tracks[str(int(ceiling))] = track_stats(pitch_hz(x, sr, ceiling))
    f = pitch_hz(x, sr, PITCH_LIMIT_HZ)
    shifts = []
    ok = True
    for cents in (310.0, 400.0):
        factor = 2.0 ** (cents / 1200.0)
        n_over, longest, n_voiced = exceed_count(f, cents, PITCH_LIMIT_HZ)
        p99 = tracks["600"]["p99_hz"] * factor
        mx = tracks["600"]["max_hz"] * factor
        # 孤立单帧尖峰不算“最高音”；p99 或连续 ≥3 帧（30 ms）越过 600 Hz 则停。
        blocked = p99 > PITCH_LIMIT_HZ or (n_over > 0 and (longest >= 3 or n_over / max(n_voiced, 1) > 0.01))
        if blocked:
            ok = False
        shifts.append({
            "cents": cents,
            "factor": round(factor, 4),
            "projected_p99_hz": round(p99, 2),
            "projected_max_hz": round(mx, 2),
            "frames_over_600": n_over,
            "longest_run_frames": longest,
            "blocked": blocked,
        })
    ceilings_agree = tracks["600"]["max_hz"] == tracks["800"]["max_hz"]
    return {
        "ok": bool(ok and ceilings_agree),
        "ceilings_agree": ceilings_agree,
        "tracks": tracks,
        "shifts": shifts,
        "note": "分析天花板 600 Hz 与 800 Hz 的最高音不一致" if not ceilings_agree else "",
    }


def shift_keyed(x: np.ndarray, sr: int, cents: float) -> tuple[np.ndarray, dict]:
    y = to_array(shift_uniform(parselmouth.Sound(x, sr), cents))
    y = fit_len(y, len(x))
    y, gain_db = match_rms(y, x)
    info = {
        "cents": cents,
        "rms_match_db": round(gain_db, 3),
        "peak_dbfs_after_match": round(float(20.0 * np.log10(max(np.max(np.abs(y)), 1e-12))), 2),
    }
    return y, info


def f0_file(path: Path) -> np.ndarray:
    x, sr = read_wav_pcm16(path)
    return pitch_hz(x, sr, PITCH_LIMIT_HZ)


def pitch_delta(ref: Path, other: Path, target_cents: float | None = None) -> dict:
    fa, fb = f0_file(ref), f0_file(other)
    n = min(len(fa), len(fb))
    voiced = fa[:n] > 0
    both = voiced & (fb[:n] > 0)
    if int(voiced.sum()) < 10 or int(both.sum()) < 10:
        return {"voiced_overlap": 0.0, "median_cents": None, "p95_abs_cents": None, "frames_both": int(both.sum())}
    cents = 1200.0 * np.log2(fb[:n][both] / fa[:n][both])
    out = {
        "voiced_overlap": round(float(both.sum() / voiced.sum()), 4),
        "median_cents": round(float(np.median(cents)), 2),
        "cents_p05": round(float(np.percentile(cents, 5)), 2),
        "cents_p95": round(float(np.percentile(cents, 95)), 2),
        # p95_abs_cents 是移调量本身的绝对值分位，不是相对目标的误差。
        "p95_abs_cents": round(float(np.percentile(np.abs(cents), 95)), 2),
        "frames_both": int(both.sum()),
        "frames_ref_voiced": int(voiced.sum()),
    }
    if target_cents is not None:
        err = cents - target_cents
        out["abs_error_p95_cents"] = round(float(np.percentile(np.abs(err), 95)), 2)
        out["frames_abs_error_over_20_cents"] = round(float(np.mean(np.abs(err) > 20)), 4)
    return out


def decay_lift_raw(orig: np.ndarray, proc: np.ndarray, sr: int, v3) -> float:
    mask, rel_o = v3.decay_mask(orig, sr)
    env = v3.env_db(proc, sr)
    n = min(len(env), len(rel_o))
    if int(mask[:n].sum()) == 0:
        raise RuntimeError("字尾衰减段为空，无法检查抬升")
    delta = (env - np.percentile(env, 95))[:n] - rel_o[:n]
    return float(np.median(delta[mask[:n]]))


def ffmpeg_version() -> str:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return r.stdout.splitlines()[0]


def write_pcm(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="PCM_16")


def main() -> None:
    if AUDIO.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {AUDIO}")
    if not SRC.is_file():
        raise SystemExit(f"输入不存在: {SRC}")
    if not V2_STUB.is_file():
        raise SystemExit(f"缺少 floor_cut 占位: {V2_STUB}")

    hash_before = sha256_file(SRC)
    x, sr = read_wav_pcm16(SRC)
    if sr != 48000:
        raise SystemExit(f"采样率应为 48000，实际 {sr}")
    n = len(x)
    pre = precheck(x, sr)
    print("pitch precheck", json.dumps(pre, ensure_ascii=False))
    if not pre["ok"]:
        report = {
            "round": HERE.name,
            "listening": "pending_jade",
            "technical_pass": False,
            "pitch_precheck": pre,
            "failures": ["升调后最高音会越过约 600 Hz 跟踪上限，未生成音频"],
        }
        (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(2)

    v3 = load_v003_chain()
    v1 = v3._v1
    if abs(float(v3.DECAY_ALLOW_DB) - DECAY_LIFT_MAX_DB) > 1e-9:
        raise SystemExit(f"v003 DECAY_ALLOW_DB={v3.DECAY_ALLOW_DB}，与本轮 ≤{DECAY_LIFT_MAX_DB} 的前提不一致")
    if abs(float(v3.SHIFT_HZ) - FORMANT_HZ) > 1e-9:
        raise SystemExit(f"v003 SHIFT_HZ={v3.SHIFT_HZ}，本轮要用 {FORMANT_HZ}")

    k310, info310 = shift_keyed(x, sr, 310.0)
    k400, info400 = shift_keyed(x, sr, 400.0)
    WORK.mkdir(exist_ok=True)
    write_wav_pcm16(WORK / "k310_level.wav", k310, sr)
    write_wav_pcm16(WORK / "k400_level.wav", k400, sr)

    print("chain: anchor / k310 / k400 / formant")
    anchor = v1.master(as_f32(x, n), sr)[0]
    a310 = v1.master(v3.chain_dry(as_f32(k310, n), sr), sr)[0]
    a400 = v1.master(v3.chain_dry(as_f32(k400, n), sr), sr)[0]
    formant_audio, formant_stats = v3.chain_effort(as_f32(k310, n), sr)
    a_formant = v1.master(formant_audio, sr)[0]

    versions = [
        {
            "id": "anchor",
            "file": "anchor.wav",
            "name": "原音·同样响",
            "f2k_cents": 0.0,
            "formant_hz": 0.0,
            "chain": "master_only",
            "audio": anchor,
            "f2k_info": None,
        },
        {
            "id": "k310_a",
            "file": "k310_a.wav",
            "name": "升一点",
            "f2k_cents": 310.0,
            "formant_hz": 0.0,
            "chain": "shift_uniform → rms_match → chain_dry → master",
            "audio": a310,
            "f2k_info": info310,
            "f2k_wav": "k310_level.wav",
        },
        {
            "id": "k400_a",
            "file": "k400_a.wav",
            "name": "再升一点",
            "f2k_cents": 400.0,
            "formant_hz": 0.0,
            "chain": "shift_uniform → rms_match → chain_dry → master",
            "audio": a400,
            "f2k_info": info400,
            "f2k_wav": "k400_level.wav",
        },
        {
            "id": "k310_formant70_a",
            "file": "k310_formant70_a.wav",
            "name": "升一点 + 更厚实",
            "f2k_cents": 310.0,
            "formant_hz": FORMANT_HZ,
            "chain": "shift_uniform → rms_match → chain_effort(formant 70 Hz + A) → master",
            "audio": a_formant,
            "f2k_info": info310,
            "f2k_wav": "k310_level.wav",
            "formant": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in formant_stats.items()},
        },
    ]

    AUDIO.mkdir()
    rows = []
    x32 = as_f32(x, n)
    st_src = v1.level_stats(x32, sr)
    for spec in versions:
        y = as_f32(spec["audio"], n)
        if len(y) != n:
            raise RuntimeError(f"{spec['id']} 样本数 {len(y)} != {n}")
        path = AUDIO / spec["file"]
        write_pcm(path, y, sr)
        back, sr_back = sf.read(str(path), dtype="float32")
        lufs, tp = v1.ebur(path)
        clipped = int(np.sum(np.abs(back) >= 0.9999))
        finite = bool(np.isfinite(back).all())
        lift = decay_lift_raw(x32, back, sr, v3)
        loud = v3.lift_stats(x32, back, sr)
        delta = pitch_delta(SRC, path, spec["f2k_cents"])
        f2k_delta = None
        if spec.get("f2k_wav"):
            f2k_delta = pitch_delta(SRC, WORK / spec["f2k_wav"], spec["f2k_cents"])
        st = v1.level_stats(back, sr)
        floor_lift = (st["floor_p05_all"] - st["p95"]) - (st_src["floor_p05_all"] - st_src["p95"])
        tail_p10 = (st["p10"] - st["p95"]) - (st_src["p10"] - st_src["p95"])
        failures = []
        if sr_back != sr:
            failures.append(f"采样率 {sr_back}")
        if len(back) != n:
            failures.append(f"样本数 {len(back)} != {n}")
        if not finite:
            failures.append("NaN 或 Inf")
        if clipped:
            failures.append(f"削波样本 {clipped}")
        if abs(lufs - TARGET_LUFS) > LUFS_TOL:
            failures.append(f"LUFS {lufs} 偏离 {TARGET_LUFS} 超过 {LUFS_TOL}")
        if tp > TP_CEIL:
            failures.append(f"真峰值 {tp} dBTP 高于 {TP_CEIL}")
        if delta["median_cents"] is None:
            failures.append("音高偏差无法计算")
        else:
            err = abs(delta["median_cents"] - spec["f2k_cents"])
            if err >= SHIFT_TOL_CENTS:
                failures.append(f"成片中位音高偏差 {delta['median_cents']} 与目标 {spec['f2k_cents']} 差 {err:.2f} 音分")
        if delta["voiced_overlap"] < OVERLAP_MIN:
            failures.append(f"浊音帧重合 {delta['voiced_overlap']} < {OVERLAP_MIN}")
        if f2k_delta is not None:
            if f2k_delta["median_cents"] is None or abs(f2k_delta["median_cents"] - spec["f2k_cents"]) >= SHIFT_TOL_CENTS:
                failures.append(f"F2-K 实际移调 {f2k_delta.get('median_cents')} 与目标 {spec['f2k_cents']} 差 ≥ {SHIFT_TOL_CENTS} 音分")
            if f2k_delta["voiced_overlap"] < OVERLAP_MIN:
                failures.append(f"F2-K 浊音帧重合 {f2k_delta['voiced_overlap']} < {OVERLAP_MIN}")
        if spec["chain"] != "master_only" and lift > DECAY_LIFT_MAX_DB:
            failures.append(f"字尾抬升 {lift:.2f} dB > {DECAY_LIFT_MAX_DB}")
        row = {
            "id": spec["id"],
            "file": spec["file"],
            "name": spec["name"],
            "f2k_cents": spec["f2k_cents"],
            "formant_hz": spec["formant_hz"],
            "chain": spec["chain"],
            "f2k_switch": "off" if spec["f2k_cents"] == 0 else "on",
            "f2k_level_match": spec["f2k_info"],
            "samples": int(len(back)),
            "samples_match": len(back) == n,
            "sample_rate": int(sr_back),
            "finite": finite,
            "clipped_samples": clipped,
            "lufs": lufs,
            "true_peak_dbtp": tp,
            "lufs_meter": "ffmpeg ebur128",
            "decay_lift_db": round(lift, 2),
            "loud_lift_db": loud["loud_lift_db"],
            "floor_lift_db": round(float(floor_lift), 2),
            "tail_p10_lift_db": round(float(tail_p10), 2),
            "sample_identity_outside_correction": "not_applicable",
            "pitch_vs_source": delta,
            "pitch_f2k_vs_source": f2k_delta,
            "sha256": sha256_file(path),
            "technical_pass": not failures,
            "failures": failures,
            "listening": "pending_jade",
        }
        if "formant" in spec:
            row["formant"] = spec["formant"]
        rows.append(row)
        print(
            f"{spec['file']:24s} LUFS {lufs:6.2f} TP {tp:5.2f} clip {clipped} "
            f"lift {lift:+5.2f} cents {delta['median_cents']} overlap {delta['voiced_overlap']} "
            f"pass {not failures} {failures}"
        )

    hash_after = sha256_file(SRC)
    if hash_after != hash_before:
        raise SystemExit("源文件哈希在处理后变了")

    passed = all(r["technical_pass"] for r in rows)
    report = {
        "round": HERE.name,
        "listening": "pending_jade",
        "agent_cannot_hear": True,
        "technical_pass": passed,
        "source": {
            "path": str(SRC.relative_to(ROOT)),
            "sha256": hash_before,
            "sha16": hash_before[:16],
            "sha256_after": hash_after,
            "hash_unchanged": True,
            "samples": n,
            "sample_rate": sr,
            "duration_s": round(n / sr, 3),
        },
        "pitch_precheck": pre,
        "floor_cut": {
            "status": "identity_stub",
            "reason": "experiments/F1_demo_chain_2026-11-02_v002/run.py 不在本快照，未做句间底噪下压",
            "applied": False,
        },
        "sibilance_qa_extra": {
            "status": "not_measured",
            "reason": "experiments/F1_demo_chain_2026-11-02_v002/qa_extra.py 不在本快照，未另写齿音指标",
        },
        "sample_identity_note": "F2-K 是整段 PSOLA，没有修正段外可逐样本对照的区间。① 只有增益和限幅。",
        "not_applied": ["reverb", "double_track", "hard_pitch_snap", "harmonic_exciter", "harmonic_tilt"],
        "f1_does_not_change_pitch": True,
        "f2k_switch": "独立步骤。① 关闭。②④ +310 音分。③ +400 音分。产品默认仍是关。",
        "targets": {
            "lufs": TARGET_LUFS,
            "lufs_tolerance": LUFS_TOL,
            "true_peak_dbtp_max": TP_CEIL,
            "shift_error_cents_max": SHIFT_TOL_CENTS,
            "voiced_overlap_min": OVERLAP_MIN,
            "decay_lift_db_max_on_a_chain": DECAY_LIFT_MAX_DB,
        },
        "formant_hz": FORMANT_HZ,
        "ffmpeg": ffmpeg_version(),
        "versions": rows,
        "listening_material_s": round(4 * n / sr, 1),
    }
    (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
