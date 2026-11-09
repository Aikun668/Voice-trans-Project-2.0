"""Measure within-phrase timing on the confirmed cuts with the local Qwen aligner."""
import json
import os
from pathlib import Path
import wave

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
DEST = Path(__file__).resolve().parent
MODEL = Path("D:/My_Grand_Projects2026/Videos-Master-Project/Qwen3-ForcedAligner-0.6B")

def main():
    out = DEST / "word_probes.json"
    if out.exists():
        raise SystemExit("Refusing to overwrite word probes.")
    import numpy as np
    import torch
    from qwen_asr import Qwen3ForcedAligner
    torch.set_num_threads(4)
    print("Loading local Qwen", flush=True)
    model = Qwen3ForcedAligner.from_pretrained(str(MODEL), dtype=torch.float32, device_map="cpu", local_files_only=True)
    result = {}
    for row in json.loads((DEST / "confirmed_pairs.json").read_text(encoding="utf-8")):
        result[row["id"]] = {}
        for side in ("self", "reference"):
            rec = row[side]
            text = row[side + "_text"]
            sr = rec["rate"]
            probes = []
            for pad in (0, .06):
                with wave.open(rec["path"], "rb") as f:
                    start = max(0, rec["frames"][0] - round(pad * sr))
                    end = min(f.getnframes(), rec["frames"][1] + round(pad * sr))
                    f.setpos(start)
                    clip = np.frombuffer(f.readframes(end - start), dtype="<i2").astype(np.float32) / 32768
                units = model.align(audio=(clip, sr), text=text, language="Chinese")[0]
                words = [{"word": unit.text, "start": round(start / sr + unit.start_time, 4),
                          "end": round(start / sr + unit.end_time, 4)} for unit in units]
                if "".join(w["word"] for w in words) != text or len(words) != len(text):
                    raise ValueError(f"Qwen returned different tokens for {row['id']} {side}")
                probes.append(words)
            result[row["id"]][side] = probes
        print(f"#{row['id']:02d} {row['phrase']} : {row['self_text']} / {row['reference_text']}", flush=True)
    with out.open("x", encoding="utf-8") as file:
        json.dump({"model": str(MODEL), "probe_padding_seconds": [0, .06], "pairs": result}, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

