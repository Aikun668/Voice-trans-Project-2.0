"""组装 demo v002：F1 V3（匹配 EQ 链）×4 首 + F2 phase4b（真正有跑调的录音）+ F2→F1 组合 ×4 首。

- F1：复制 experiments/F1_demo_chain_2026-11-02_v002/audio（-13 LUFS）。
- F2 单独：原音 / phase4b 修正 / 原唱对应段，统一 -16 LUFS 固定增益（只改音高，必须等响比）。
- 组合：phase4b 修正版先还原到原音电平，再走 F1 V3 → 母带 -13 LUFS；同时给"原音·同响度"和"只 F1"。
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
F1V2 = ROOT / "experiments/F1_demo_chain_2026-11-02_v002"


def _load(name: str, path: Path):
    """各实验目录的脚本同名（run.py / build_demo.py），按路径加载避免 sys.path 冲突。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_f1 = _load("f1_v002_run", F1V2 / "run.py")   # 其内部再按 sys.path 导入 v001 的 run（chain_v1/master 等）
_demo1 = _load("demo_v001_build", ROOT / "experiments/DEMO_2026-11-02_v001/build_demo.py")
chain_v3, ebur, master, sha16, target_spectrum = _f1.chain_v3, _f1.ebur, _f1.master, _f1.sha16, _f1.target_spectrum
fixed_gain = _demo1.fixed_gain

F2 = ROOT / "f2/outputs"
MY = ROOT / "test_dataset/my_singing"
COMBOS = {  # key: (原始录音, phase4b 输出目录, 说明)
    "zuixuan_mild": (MY / "zuixuan_mild.wav", F2 / "phase4b_zuixuan_mild_v001", "最炫民族风·轻度跑调"),
    "shige_severe": (MY / "shige_severe.wav", F2 / "phase4b_shige_severe_v001", "自由飞翔（是谁在唱歌）·严重跑调"),
    "zuixuan_severe": (MY / "zuixuan_severe.wav", F2 / "phase4b_zuixuan_severe_v001", "最炫民族风·严重跑调"),
    "gyjyn": (ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav/因为刚好遇见你-清唱版.wav",
              F2 / "phase4b_gyjyn_v001", "因为刚好遇见你·清唱版"),
}


def restore_level(f2dir: Path, src: Path) -> tuple[np.ndarray, int]:
    x0, sr = sf.read(str(src), dtype="float32")
    t1, _ = sf.read(str(f2dir / "01_target_original.wav"), dtype="float32")
    corr, _ = sf.read(str(f2dir / "02_corrected.wav"), dtype="float32")
    g = np.sqrt(np.mean(t1**2) / np.mean(x0**2))
    return (corr / g).astype(np.float32), sr


def write_master(y: np.ndarray, sr: int, p: Path) -> dict:
    m, _, _, _ = master(y, sr)
    sf.write(str(p), m, sr, subtype="PCM_16")
    lufs, tp = ebur(p)
    return {"file": p.name, "lufs": lufs, "true_peak_dbtp": tp, "samples": len(m),
            "clipped_samples": int(np.sum(np.abs(m) >= 0.9999))}


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    out.mkdir()
    report = []
    for p in sorted((F1V2 / "audio").glob("*.wav")):
        if p.stem.endswith("_v1"):
            continue
        shutil.copy2(p, out / f"f1_{p.name}")
        lufs, tp = ebur(out / f"f1_{p.name}")
        report.append({"file": f"f1_{p.name}", "source": str(p.relative_to(ROOT)), "lufs": lufs, "true_peak_dbtp": tp})

    tgt = target_spectrum()
    for key, (src, f2dir, _) in COMBOS.items():
        h0 = sha16(src)
        report.append(fixed_gain(src, out / f"f2_{key}_orig.wav"))
        report.append(fixed_gain(f2dir / "02_corrected.wav", out / f"f2_{key}_fixed.wav"))
        report.append(fixed_gain(f2dir / "03_reference_matched_segment.wav", out / f"f2_{key}_reference.wav"))
        x, sr = sf.read(str(src), dtype="float32")
        corr, _ = restore_level(f2dir, src)
        assert len(corr) == len(x), key
        report.append(write_master(x, sr, out / f"combo_{key}_orig.wav") | {"source": str(src.relative_to(ROOT))})
        report.append(write_master(chain_v3(x, sr, tgt)[0], sr, out / f"combo_{key}_f1.wav") | {"source": "F1 V3"})
        report.append(write_master(chain_v3(corr, sr, tgt)[0], sr, out / f"combo_{key}_f2f1.wav")
                      | {"source": f"{f2dir.relative_to(ROOT)} -> F1 V3"})
        assert sha16(src) == h0
    (HERE / "levels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:34s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} {r.get('note', '')}")


if __name__ == "__main__":
    main()
