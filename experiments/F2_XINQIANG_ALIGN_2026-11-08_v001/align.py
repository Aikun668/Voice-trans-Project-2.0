"""Measure the marked words in the 《心墙》 take and the confirmed reference region."""

import hashlib
import json
from pathlib import Path


DEST = Path(__file__).resolve().parent
REPO = DEST.parents[1]
DATA = REPO / "test_dataset/my_singing/rerecord_2026-11-04_f2"
SOURCE = DATA / "wav/心墙01-对照微微跑调.wav"
REFERENCE = DATA / "ref_vocals/心墙_郭静_vocals.wav"
MODEL = Path("D:/My_Grand_Projects2026/Videos-Master-Project/Qwen3-ForcedAligner-0.6B")
TRANSCRIPT = "我学着不去担心得太远"
WINDOWS = {
    "take": (SOURCE, ((0.00, 6.40), (0.08, 7.60))),
    "reference": (REFERENCE, ((111.90, 119.80), (112.40, 120.20))),
}


def main():
    output = DEST / "alignment.json"
    if output.exists() or (DEST / "audio").exists():
        raise SystemExit("实验输出已存在，拒绝覆盖")
    print("Loading the local forced aligner", flush=True)
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner

    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE, REFERENCE)}
    torch.set_num_threads(4)
    model = Qwen3ForcedAligner.from_pretrained(
        str(MODEL), dtype=torch.float32, device_map="cpu", local_files_only=True
    )
    results = {}
    for name, (path, windows) in WINDOWS.items():
        rows = []
        for start, end in windows:
            with sf.SoundFile(path) as source:
                sr = source.samplerate
                source.seek(round(start * sr))
                clip = source.read(round((end - start) * sr), dtype="float32")
            if clip.ndim != 1:
                raise RuntimeError(f"需要单声道人声：{path.name}")
            units = model.align(audio=(clip, sr), text=TRANSCRIPT, language="Chinese")[0]
            words = [
                {"text": u.text, "start": round(start + u.start_time, 3), "end": round(start + u.end_time, 3)}
                for u in units
            ]
            if "".join(u["text"] for u in words) != TRANSCRIPT:
                raise RuntimeError(f"对齐文本不符：{name}")
            rows.append({"search_window_seconds": [start, end], "words": words})
        results[name] = rows
        print(name, json.dumps(rows, ensure_ascii=False), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("对齐期间源文件发生变化")
    with output.open("x", encoding="utf-8") as file:
        json.dump({"model": str(MODEL), "inputs": hashes, "results": results}, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
