"""打印两版音符序列, 人工核对是否同一首歌、对齐是否合理。"""
import sys

from self_ref_align import note_name, notes_of

for tag, p in [("severe", r"test_dataset\my_singing\shige_severe.wav"),
               ("normal", r"test_dataset\my_singing\shige_normal.wav")]:
    ns = notes_of(p)
    print(f"{tag}: {len(ns)} 个稳定音")
    items = [f"{note_name(n['cents'])}@{n['t0']:.1f}" for n in ns]
    for i in range(0, len(items), 9):
        print("   " + "  ".join(items[i:i + 9]))
    print()
