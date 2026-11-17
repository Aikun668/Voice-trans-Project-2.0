"""组装 2026-11-02 demo 页音频：F1 成品链、F2 重对齐修音、F2→F1 组合。

- F1：直接复制 experiments/F1_demo_chain_2026-11-02_v001/audio（已统一 -13 LUFS）。
- F2：只改音高，比较必须等响：统一 -16 LUFS，只施加固定增益（峰值超 -1 dBFS 则整体降低并记录）。
- 组合：F2 输出先还原到原音电平（F1 链的压缩门限是绝对电平），再走 F1 V1/V2 链 + 安静段复原 + 母带到 -13 LUFS。
用 .venv-f1-adaptive 运行。
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
F1 = ROOT / "experiments/F1_demo_chain_2026-11-02_v001"
sys.path.insert(0, str(F1))
from run import chain_v1, chain_v2, ebur, master, restore_quiet, sha16  # noqa: E402

F2 = ROOT / "f2/outputs"
GYJYN = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav/因为刚好遇见你-清唱版.wav"
F2_LUFS = -16.0
PEAK_CEIL = -1.0


def fixed_gain(src: Path, dst: Path, target: float = F2_LUFS) -> dict:
    x, sr = sf.read(str(src), dtype="float32")
    gain = target - ebur(src)[0]
    pk = 20 * np.log10(np.max(np.abs(x)) + 1e-12) + gain
    note = ""
    if pk > PEAK_CEIL:
        gain -= pk - PEAK_CEIL
        note = "peak-limited"
    sf.write(str(dst), x * 10 ** (gain / 20), sr, subtype="PCM_16")
    lufs, tp = ebur(dst)
    return {"file": dst.name, "source": str(src.relative_to(ROOT)), "source_sha16": sha16(src), "gain_db": round(gain, 2),
            "lufs": lufs, "true_peak_dbtp": tp, "samples": len(x), "note": note}


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    out.mkdir()
    report = []
    for p in sorted((F1 / "audio").glob("*.wav")):
        shutil.copy2(p, out / f"f1_{p.name}")
        lufs, tp = ebur(out / f"f1_{p.name}")
        report.append({"file": f"f1_{p.name}", "source": str(p.relative_to(ROOT)), "lufs": lufs, "true_peak_dbtp": tp})

    f2_plan = [
        ("f2_real_orig.wav", GYJYN),
        ("f2_real_p3e_old.wav", F2 / "phase3e_gyjyn_v002/02_corrected.wav"),
        ("f2_real_p4_new.wav", F2 / "phase4_gyjyn_v002/02_corrected.wav"),
        ("f2_inj_orig.wav", GYJYN),
        ("f2_inj_detuned.wav", F2 / "phase4_gyjyn_inject_v001/input_detuned.wav"),
        ("f2_inj_repaired.wav", F2 / "phase4_gyjyn_inject_v001/repair/02_corrected.wav"),
        ("f2_reference.wav", F2 / "phase4_gyjyn_v002/03_reference_matched_segment.wav"),
    ]
    report += [fixed_gain(src, out / name) for name, src in f2_plan]

    # 组合：phase4 输出与 01_target_original 共用同一增益，按此比值还原到原音电平
    x0, sr = sf.read(str(GYJYN), dtype="float32")
    t1, _ = sf.read(str(F2 / "phase4_gyjyn_v002/01_target_original.wav"), dtype="float32")
    corr, _ = sf.read(str(F2 / "phase4_gyjyn_v002/02_corrected.wav"), dtype="float32")
    g = np.sqrt(np.mean(t1**2) / np.mean(x0**2))
    corr = (corr / g).astype(np.float32)
    for tag, y in [("v1", restore_quiet(corr, chain_v1()(corr, sr), sr)), ("v2", restore_quiet(corr, chain_v2(corr, sr), sr))]:
        m, gain, lufs, tp = master(y, sr)
        p = out / f"combo_gyjyn_f2_f1{tag}.wav"
        sf.write(str(p), m, sr, subtype="PCM_16")
        lufs, tp = ebur(p)
        report.append({"file": p.name, "source": "phase4_gyjyn_v002/02_corrected.wav -> F1 " + tag, "lufs": lufs,
                       "true_peak_dbtp": tp, "samples": len(m), "samples_match": len(m) == len(x0)})
    (HERE / "levels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:34s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} {r.get('note', '')}")


if __name__ == "__main__":
    main()
