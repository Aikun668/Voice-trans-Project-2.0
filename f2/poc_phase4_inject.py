"""F2 Phase-4 受控跑调注入：把本人在调的录音里 5 个稳定音人为推偏约 2 个半音，
再用 poc_phase4_realign.py 以原唱为参考修复。用于在"本人几乎不跑调"的素材上验证修复能力。

选音依据：outputs/phase4_gyjyn_v001/notes.csv 中 status=ok（参考段稳定、偏差 <60 音分）的长音。
用法: python poc_phase4_inject.py <target.wav> <out.wav>
"""
import json
import sys
from pathlib import Path

import parselmouth

from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase3b import apply_tapered

INJECT = [  # (t0, t1, cents)
    (2.96, 3.70, +200.0),
    (4.33, 4.66, -200.0),
    (5.84, 6.25, +180.0),
    (7.97, 8.46, -200.0),
    (10.74, 11.13, +200.0),
]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f"拒绝覆盖: {dst}")
    x, sr = read_wav_pcm16(src)
    y = apply_tapered(parselmouth.Sound(x, sr), INJECT)[: len(x)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(dst, y, sr)
    dst.with_suffix(".json").write_text(json.dumps({"source": str(src), "inject": INJECT}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"注入 {len(INJECT)} 处 -> {dst}")


if __name__ == "__main__":
    main()
