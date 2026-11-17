"""Shared helpers for the pitch-target study (read-only use of the official NeuralSVB code path).

All paths are absolute: neuralsvb_infer._bootstrap() chdirs into the official repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "official_all"
sys.path.insert(0, str(ROOT / "neuralsvb_f1"))
sys.path.insert(0, str(HERE))

SR = 22050
HOP = 128
FRAME_S = HOP / SR
LANGS = {"zh": "Chinese", "en": "English"}


def clip_ids(lang_dir: str) -> list[int]:
    """Clip ids present in all of gt_a / gt_p / NSVB for one language, sorted."""
    sets = [{int(p.stem) for p in (DATA / lang_dir / d).glob("*.wav")} for d in ("gt_a", "gt_p", "NSVB")]
    return sorted(set.intersection(*sets))


def clip_path(lang_dir: str, kind: str, cid: int) -> Path:
    return DATA / lang_dir / kind / f"{cid}.wav"


def all_clips() -> list[tuple[str, int]]:
    """[(tag, id)] for every complete official clip, e.g. ('zh', 7)."""
    return [(tag, cid) for tag, d in LANGS.items() for cid in clip_ids(d)]


def split_of(tag: str, cid: int) -> str:
    """Fixed, content-independent split: every third clip (by id) is held out for the final test."""
    return "test" if cid % 3 == 0 else "dev"


_ENGINE = {}


def feature_engine():
    """Bootstrap the official hparams once; returns the neuralsvb_infer module."""
    if "n" not in _ENGINE:
        import neuralsvb_infer as n

        n._bootstrap(None)
        _ENGINE["n"] = n
    return _ENGINE["n"]


def features(wav_path: str | Path):
    """(wav, mel, f0): official PWG mel + official Praat F0 (same as the F1 pipeline, no speaker embedding)."""
    n = feature_engine()  # must run first: it puts the official repo on sys.path
    from vocoders.pwg import PWG

    wav, mel = PWG.wav2spec(str(wav_path))
    f0 = n._get_pitch(wav, mel, _hp())
    return wav, mel, f0


def _hp():
    from utils.hparams import hparams

    return hparams


def cents(f_hz: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    """Hz -> cents re ref; NaN where f <= 0."""
    f = np.asarray(f_hz, dtype=float)
    out = np.full(f.shape, np.nan)
    ok = f > 0
    out[ok] = 1200.0 * np.log2(f[ok] / ref_hz)
    return out


def warp_pro_to_amateur(f0_pro: np.ndarray, alignment: np.ndarray, n_amateur: int) -> np.ndarray:
    """Pro F0 on the amateur frame axis. alignment[i] = amateur frame matched to pro frame i (official EHSADTW).

    Each amateur frame takes the median of the voiced pro frames mapped to it; 0 where none (unvoiced / unmatched).
    """
    buckets: list[list[float]] = [[] for _ in range(n_amateur)]
    for i, j in enumerate(np.asarray(alignment).astype(int)):
        if 0 <= j < n_amateur and f0_pro[i] > 0:
            buckets[j].append(float(f0_pro[i]))
    return np.array([np.median(b) if b else 0.0 for b in buckets])


def _shape_dist_chunked(src: np.ndarray, tgt: np.ndarray, chunk: int = 64):
    """Same numbers as the official cal_shape_dist, built in row chunks (the official one allocates
    a [T1, T2, 48] float64 tensor three times over and crashed with an access violation on ~1000-frame clips)."""
    import torch
    from modules.voice_conversion.dtw import enhance_sadtw as E

    src_hist = torch.tensor(E.cal_hist_of_f0(src, normalize_hist=True))
    tgt_hist = torch.tensor(E.cal_hist_of_f0(tgt, normalize_hist=True, scale_factor=len(tgt) / len(src)))
    rows = [E.cal_hist_dist(src_hist[i:i + chunk], tgt_hist, None, None) for i in range(0, len(src_hist), chunk)]
    return torch.cat(rows)  # [S, T]


def dtw_alignment(f0_amateur: np.ndarray, f0_pro: np.ndarray) -> np.ndarray:
    """Official shape-aware DTW alignment (len == len(f0_pro)), clipped into the amateur range."""
    feature_engine()
    from modules.voice_conversion.dtw.align import align_from_distances

    costs = _shape_dist_chunked(f0_amateur, f0_pro)
    alignment = np.asarray(align_from_distances(costs.T.cpu().detach().numpy()))
    return np.clip(alignment[: len(f0_pro)], 0, len(f0_amateur) - 1)


def dtw_alignment_official(f0_amateur: np.ndarray, f0_pro: np.ndarray) -> np.ndarray:
    """Unmodified official call; only used to verify dtw_alignment on clips short enough not to crash."""
    feature_engine()
    from modules.voice_conversion.dtw.enhance_sadtw import EHSADTW

    _, alignment = EHSADTW(f0_amateur, f0_pro, f0_amateur)
    return np.clip(np.asarray(alignment)[: len(f0_pro)], 0, len(f0_amateur) - 1)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average with edge replication (width forced odd)."""
    width = max(1, int(width) | 1)
    pad = width // 2
    padded = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")
