"""组装 demo v003：F1 A/B（去字尾空响 / +力度）×4 首；F2 phase4c（拖音 + 连续衔接）×3 首跑调录音；组合 ×4 首。

- F1：复制 experiments/F1_demo_chain_2026-11-02_v003/audio（-13 LUFS）。
- F2 单独：原音 / 上一轮 phase4b / 新 phase4c / 原唱对应段，统一 -16 LUFS 固定增益。
- 组合：phase4c 修正版还原到原音电平 → F1 A 或 B → -13 LUFS；附"原音·同样响"。
用 .venv-f1-adaptive 运行。
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F1V3 = ROOT / "experiments/F1_demo_chain_2026-11-02_v003"
_f1 = _load("f1_v003_run", F1V3 / "run.py")
_d1 = _load("demo_v001_build", ROOT / "experiments/DEMO_2026-11-02_v001/build_demo.py")
_d2 = _load("demo_v002_build", ROOT / "experiments/DEMO_2026-11-02_v002/build_demo.py")
fixed_gain, restore_level, write_master = _d1.fixed_gain, _d2.restore_level, _d2.write_master

F2 = ROOT / "f2/outputs"
MY = ROOT / "test_dataset/my_singing"
GYJYN = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav/因为刚好遇见你-清唱版.wav"
SONGS = {  # key: (原始录音, phase4b 目录, phase4c 目录)
    "zuixuan_mild": (MY / "zuixuan_mild.wav", "phase4b_zuixuan_mild_v001", "phase4c_zuixuan_mild_v001"),
    "shige_severe": (MY / "shige_severe.wav", "phase4b_shige_severe_v001", "phase4c_shige_severe_v001"),
    "zuixuan_severe": (MY / "zuixuan_severe.wav", "phase4b_zuixuan_severe_v001", "phase4c_zuixuan_severe_v001"),
    "gyjyn": (GYJYN, "phase4b_gyjyn_v001", "phase4c_gyjyn_v001"),
}


def main() -> None:
    out = HERE / "audio"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    out.mkdir()
    report = []
    for p in sorted((F1V3 / "audio").glob("*.wav")):
        if p.stem.endswith("_v3"):
            continue
        shutil.copy2(p, out / f"f1_{p.name}")
        lufs, tp = _f1.ebur(out / f"f1_{p.name}")
        report.append({"file": f"f1_{p.name}", "source": str(p.relative_to(ROOT)), "lufs": lufs, "true_peak_dbtp": tp})
    for key, (src, d4b, d4c) in SONGS.items():
        h0 = _f1.sha16(src)
        if key != "gyjyn":
            report.append(fixed_gain(src, out / f"f2_{key}_orig.wav"))
            report.append(fixed_gain(F2 / d4b / "02_corrected.wav", out / f"f2_{key}_4b.wav"))
            report.append(fixed_gain(F2 / d4c / "02_corrected.wav", out / f"f2_{key}_4c.wav"))
            report.append(fixed_gain(F2 / d4c / "03_reference_matched_segment.wav", out / f"f2_{key}_reference.wav"))
        x, sr = sf.read(str(src), dtype="float32")
        corr, _ = restore_level(F2 / d4c, src)
        assert len(corr) == len(x), key
        report.append(write_master(x, sr, out / f"combo_{key}_orig.wav") | {"source": str(src.relative_to(ROOT))})
        report.append(write_master(_f1.chain_dry(corr, sr), sr, out / f"combo_{key}_f2a.wav") | {"source": f"{d4c} -> F1 A"})
        report.append(write_master(_f1.chain_effort(corr, sr)[0], sr, out / f"combo_{key}_f2b.wav") | {"source": f"{d4c} -> F1 B"})
        assert _f1.sha16(src) == h0
    (HERE / "levels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report:
        print(f"{r['file']:34s} {r['lufs']:6.1f} LUFS TP {r['true_peak_dbtp']:5.1f} {r.get('note', '')}")


if __name__ == "__main__":
    main()
