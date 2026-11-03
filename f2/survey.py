"""普查: 6 首歌的参考修音候选, 哪些修正落在"听得见"的范围。

感知显著性代理 = 音符时长(s) × 修正量(音分)。校准实验表明:
约 0.21s@103 音分不可辨, 0.61s@247 音分可辨 -> 阈值大致在 150 附近。
"""

import tempfile
from pathlib import Path

import numpy as np

from poc_phase1 import read_wav_pcm16, write_wav_pcm16
from poc_phase3b import FIX_THRESHOLD, align_notes, note_name, notes_of

PAIRS = [
    ("自由飞翔", r"test_dataset\my_singing\shige_medium.wav",
     r"test_dataset\reference\vocals\自由飞翔_凤凰传奇_vocals.wav", 73.7),
    ("最炫民族风", r"test_dataset\my_singing\zuixuan_normal.wav",
     r"test_dataset\reference\vocals\最炫民族风_凤凰传奇_vocals.wav", 113.2),
    ("平凡之路", r"test_dataset\f1_dataset_v2\A_thin\self\wav\平凡之路正常版.wav",
     r"test_dataset\reference\vocals\平凡之路_朴树_vocals.wav", 203.8),
    ("雅俗共赏", r"test_dataset\f1_dataset_v2\A_thin\self\wav\雅俗共赏-正常版.wav",
     r"test_dataset\reference\vocals\雅俗共赏_许嵩_vocals.wav", 59.2),
    ("我们的歌", r"test_dataset\f1_dataset_v2\A_thin\self\wav\我们的歌正常版.wav",
     r"test_dataset\reference\vocals\我们的歌_王力宏_vocals.wav", 107.9),
    ("刚好遇见你", r"test_dataset\f1_dataset_v2\A_thin\self\wav\因为刚好遇见你-原版.wav",
     r"test_dataset\reference\vocals\刚好遇见你_李玉刚_录音室_vocals.wav", 150.2),
]

print(f"{'歌曲':<10} {'配对':>4} {'待修':>4} {'显著性>150':>10} {'最大显著性':>10}  最显眼的一处")
for song, user, ref, start in PAIRS:
    ux, usr = read_wav_pcm16(Path(user))
    rx, rsr = read_wav_pcm16(Path(ref))
    dur = len(ux) / usr
    a = max(0, int((start - 1.0) * rsr))
    b = min(len(rx), int((start + dur * 1.6 + 3.0) * rsr))
    tmp = Path(tempfile.gettempdir()) / f"_ref_{song}.wav"
    write_wav_pcm16(tmp, rx[a:b], rsr)

    t = notes_of(user)
    r = notes_of(tmp)
    pairs, _ = align_notes(t, r)
    if not pairs:
        print(f"{song:<10} 无配对")
        continue
    rows = []
    for i, j in pairs:
        rows.append({"t0": t[i]["t0"], "t1": t[i]["t1"], "note": note_name(t[i]["cents"]),
                     "ref": note_name(r[j]["cents"]), "dev": t[i]["cents"] - r[j]["cents"]})
    off = float(np.median([x["dev"] for x in rows]))
    for x in rows:
        x["rel"] = x["dev"] - off
        x["shift"] = -x["rel"] if abs(x["rel"]) > FIX_THRESHOLD else 0.0
    fix = [x for x in rows if x["shift"] != 0]
    sal = [(x["t1"] - x["t0"]) * abs(x["shift"]) for x in fix]
    best = max(fix, key=lambda z: (z["t1"] - z["t0"]) * abs(z["shift"])) if fix else None
    n_hi = sum(1 for s in sal if s > 150)
    desc = (f"{best['t0']:.2f}s {best['note']}->{best['ref']} "
            f"{best['shift']:+.0f}音分/{(best['t1'] - best['t0']):.2f}s") if best else "-"
    print(f"{song:<10} {len(rows):>4} {len(fix):>4} {n_hi:>10} "
          f"{(max(sal) if sal else 0):>10.0f}  {desc}")
