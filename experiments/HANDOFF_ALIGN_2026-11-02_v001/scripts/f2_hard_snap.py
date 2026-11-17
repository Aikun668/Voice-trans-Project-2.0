"""对齐探针：硬吸附修音（每个音高点贴到最近的十二平均律半音，A4=440）。

只用于"你说的修音是不是这种方向"的夸张对照，不是候选方案。
复用 f2/poc_phase1.py 的读写与 Praat Manipulation 参数。用 .venv-f2 运行。
用法: python f2_hard_snap.py <input.wav> <output.wav>
"""
import math
import sys
from pathlib import Path

import parselmouth

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "f2"))
from poc_phase1 import PITCH_CEIL, PITCH_FLOOR, TIME_STEP, read_wav_pcm16, to_array, write_wav_pcm16  # noqa: E402

call = parselmouth.praat.call


def snap_hz(f):
    return 440.0 * 2.0 ** (round(12.0 * math.log2(f / 440.0)) / 12.0)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    x, sr = read_wav_pcm16(src)
    sound = parselmouth.Sound(x, sampling_frequency=sr)
    manip = call(sound, "To Manipulation", TIME_STEP, PITCH_FLOOR, PITCH_CEIL)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    points = [(call(tier, "Get time from index", i), call(tier, "Get value at index", i)) for i in range(1, n + 1)]
    dur = sound.get_total_duration()
    new_tier = call("Create PitchTier", "snap", 0.0, dur)
    shifts = []
    for t, f in points:
        g = snap_hz(f)
        shifts.append(1200.0 * math.log2(g / f))
        call(new_tier, "Add point", t, g)
    call([manip, new_tier], "Replace pitch tier")
    y = to_array(call(manip, "Get resynthesis (overlap-add)"))[: len(x)]
    write_wav_pcm16(dst, y, sr)
    mean_abs = sum(abs(s) for s in shifts) / max(len(shifts), 1)
    print(f"{src.name}: {n} pitch points, mean |shift| {mean_abs:.1f} cents, max {max(map(abs, shifts)):.1f}")


if __name__ == "__main__":
    main()
