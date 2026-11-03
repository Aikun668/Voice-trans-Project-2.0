"""从人声轨截取指定区间, 作为参考片段。"""
import sys
from pathlib import Path

from poc_phase1 import read_wav_pcm16, write_wav_pcm16

src, start, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
x, sr = read_wav_pcm16(Path(src))
a, b = int(start * sr), int((start + dur) * sr)
write_wav_pcm16(Path(out), x[a:b], sr)
print(f"截取 {start:.1f}s 起 {dur:.1f}s -> {out} ({len(x[a:b]) / sr:.2f}s)")
