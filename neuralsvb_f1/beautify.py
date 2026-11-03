"""CLI: python neuralsvb_f1/beautify.py INPUT.wav OUTPUT.wav [--mode a2p] [--seed 0]

Single-WAV F1 beautification with the official NeuralSVB weights. No reference audio needed.
Output is 22.05 kHz mono PCM16 (the model's native rate); loudness is NOT matched here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from neuralsvb_infer import beautify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_wav")
    ap.add_argument("output_wav")
    ap.add_argument("--mode", default="a2p", choices=["a2p", "a2p_nomap", "a2a", "vocoder"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if Path(args.output_wav).exists():
        sys.exit(f"{args.output_wav} exists; refusing to overwrite")
    print(json.dumps(beautify(args.input_wav, args.output_wav, mode=args.mode, seed=args.seed),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
