"""Step 2 validation: run the official a2p path on the official PopBuTFy preview pairs.

The full packed test set (binary/PopBuTFyENSpkEM, phone_set.json, train_f0s_mean_std.npy) is not
released, so this reproduces the same a2p computation from the 3 amateur/professional preview clips.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import neuralsvb_infer as n  # noqa: E402

PREVIEW = n.SRC / "data" / "preview"
OUT = Path(__file__).parent / "outputs" / "official_preview_pairs"


def clip(role: str, i: int) -> str:
    d = f"Male1#singing#five_hundred_Miles_{role}"
    return str(PREVIEW / d / f"{d}_{i}.mp3")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Proxy for the unreleased train_f0s_mean_std.npy: pooled voiced F0 of the 6 preview clips.
    n._bootstrap(None)
    probe = n.NeuralSVB()
    pooled = np.concatenate([probe.features(clip(r, i))[2] for r in ("Amateur", "Professional") for i in range(3)])
    pooled = pooled[pooled > 0]
    f0n = n.F0Norm(float(pooled.mean()), float(pooled.std()))
    probe.f0_norm = f0n
    probe.hp["f0_mean"], probe.hp["f0_std"] = f0n.mean, f0n.std
    report = {"f0_norm_proxy": [f0n.mean, f0n.std], "pairs": []}
    for i in range(3):
        wavs, mels, gt, align = probe.run_pair(clip("Amateur", i), clip("Professional", i))
        # mel distance to the professional target (same length by construction)
        pm = gt["p"]
        d = {k: float(np.abs(mels[k][: len(pm)] - pm[: len(mels[k])]).mean()) for k in ("a2a", "p2p", "a2p")}
        d["amateur_aligned_vs_prof"] = float(np.abs(gt["a"][align] - pm).mean())
        for k, w in wavs.items():
            sf.write(OUT / f"pair{i}_{k}.wav", w, n.SAMPLE_RATE, subtype="PCM_16")
        report["pairs"].append({"pair": i, "frames_a": len(gt["a"]), "frames_p": len(pm), "mel_l1_to_prof": d})
        print(report["pairs"][-1])
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
