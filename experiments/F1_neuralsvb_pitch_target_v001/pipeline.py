"""Single-recording NeuralSVB beautifier, candidate v3 (official weights, no training).

Differences from the earlier F1 single-WAV path, each measured on official demo clips / the user's recordings:
  1. F0 time axis: Praat frame centres mapped onto mel frames, delayed one frame (the position the vocoder reproduces best).
  2. Pitch target: the singer's own F0 with detector mistakes repaired (pitch_target.make_target). No melody change.
  3. Pitch inputs follow the shipped config (normalize_pitch: false): raw F0, no standardisation to invented statistics.
  4. Optional output calibration: dip_repair (model collapses quiet unvoiced stretches) and colour_match (global tilt).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "F1_neuralsvb_contract_v001"))
import common as c  # noqa: E402
from colour_match import ColourConfig, colour_match  # noqa: E402
from level_match import LevelConfig, level_match  # noqa: E402
from octave_evidence import OctaveConfig, fix_octaves  # noqa: E402
from pitch_target import TargetConfig, make_target  # noqa: E402

SR = c.SR
HOP = c.HOP
F0_DELAY_FRAMES = 1


@dataclass
class Analysis:
    wav: np.ndarray        # 22.05 kHz, length count*HOP
    mel: np.ndarray        # [count, 80]
    f0: np.ndarray         # [count] Praat F0 on the mel frame axis (Hz, 0 = unvoiced)
    hint: np.ndarray       # [count] bool, independent voicing decision (pYIN) on the same axis
    ref_f0: np.ndarray     # [count] independent pitch (pYIN, Hz, 0 = unvoiced) on the same axis
    spk: np.ndarray        # 256-d speaker embedding
    count: int


class Beautifier:
    def __init__(self) -> None:
        self.n = c.feature_engine()
        self.eng = self.n.NeuralSVB()
        self.eng.hp["f0_mean"], self.eng.hp["f0_std"] = 300.0, 100.0  # norm->denorm round trip is identity for the vocoder branch

    # ---- analysis -----------------------------------------------------------------------------------------
    def analyse(self, wav22: np.ndarray, f0_mode: str = "aligned") -> Analysis:
        import librosa
        import torch  # noqa: F401  (ensures torch is initialised before the official modules)
        from vocoders.pwg import PWG

        wav, mel = PWG.wav2spec(np.asarray(wav22, dtype=np.float32))
        count = len(mel)
        if f0_mode == "legacy":
            f0 = self.n._get_pitch(wav, mel, c._hp())
        else:
            import feature_contract as fc

            f0, _ = fc.aligned_f0(wav, count, frame_offset_seconds=-F0_DELAY_FRAMES * HOP / SR)
        hint = np.zeros(count, dtype=bool)
        ref_f0 = np.zeros(count)
        if f0_mode != "legacy":  # independent tracker, only used by the target repair
            yp, vf, _ = librosa.pyin(wav, fmin=75, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
            m = min(count, len(vf))
            hint[:m] = vf[:m]
            ref_f0[:m] = np.where(vf[:m], yp[:m], 0.0)
            hint, ref_f0 = np.roll(hint, F0_DELAY_FRAMES), np.roll(ref_f0, F0_DELAY_FRAMES)
            hint[:F0_DELAY_FRAMES], ref_f0[:F0_DELAY_FRAMES] = False, 0.0
        spk = self.eng.voice_encoder.embed_utterance(wav)
        return Analysis(wav=wav, mel=mel, f0=np.asarray(f0, dtype=float), hint=hint, ref_f0=ref_f0, spk=spk, count=count)

    # ---- model + vocoder ----------------------------------------------------------------------------------
    def _pitch_ids_and_vocoder_f0(self, f0: np.ndarray):
        from utils.pitch_utils import denorm_f0, f0_to_coarse, norm_interp_f0
        import torch

        hp = self.eng.hp
        ids = f0_to_coarse(f0.copy())
        f0n, uv = norm_interp_f0(f0.copy(), hp)
        voc = denorm_f0(torch.FloatTensor(f0n), torch.FloatTensor(uv), hp).numpy()
        return ids, voc

    def synthesise(self, a: Analysis, target_f0: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
        """a2p output and the vocoder-only resynthesis of the same mel, both driven by `target_f0`."""
        import torch

        pad = (-a.count) % 4
        pe = lambda x, kind: np.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")  # noqa: E731
        ids_self, _ = self._pitch_ids_and_vocoder_f0(pe(a.f0, 0))
        ids_tgt, f0_voc = self._pitch_ids_and_vocoder_f0(pe(target_f0, 0))
        mel_t = torch.tensor(pe(a.mel, 0), dtype=torch.float32)[None]
        spk_t = torch.tensor(a.spk, dtype=torch.float32)[None]
        align = torch.arange(mel_t.shape[1], dtype=torch.long)[None]
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.eng.model(
                amateur_mel=mel_t, prof_mel=mel_t,
                amateur_pitch=torch.tensor(ids_self, dtype=torch.long)[None],
                prof_pitch=torch.tensor(ids_tgt, dtype=torch.long)[None],
                amateur_spk_id=spk_t, prof_spk_id=spk_t,
                a2p_alignment=align, p2a_alignment=None, infer=False,
                concurrent_ways=["a2a", "p2p", "a2p"], disable_map=False)
        mel_out = out["a2p"]["mel_out"][0].detach().numpy()[: a.count]
        f0v = f0_voc[: a.count]
        torch.manual_seed(seed)
        a2p = self.eng.vocoder.spec2wav(mel_out, f0=f0v)
        torch.manual_seed(seed)
        voc = self.eng.vocoder.spec2wav(a.mel, f0=f0v)
        n = len(a.wav)
        fit = lambda w: np.pad(w, (0, max(0, n - len(w))))[:n]  # noqa: E731
        return {"a2p": fit(a2p), "vocoder": fit(voc), "f0_vocoder": f0v}


def build_target(a: Analysis, use_target: bool, config: TargetConfig | None = None,
                 octave: OctaveConfig | None = None):
    """Own F0 with detector mistakes repaired: octave errors (decided on the waveform), then spikes / short gaps."""
    if not use_target:
        return a.f0.copy(), {}
    f0, oct_rep = fix_octaves(a.f0, a.ref_f0, a.wav, SR, HOP, octave, time_offset_frames=F0_DELAY_FRAMES)
    target, rep = make_target(f0, a.hint, config)
    rep["octave_evidence"] = oct_rep
    return target, rep


def postprocess(a: Analysis, base: dict[str, np.ndarray], level: bool, colour: float,
                level_config: LevelConfig | None = None, colour_config: ColourConfig | None = None):
    """Output calibration on the 22.05 kHz a2p output: loudness contour follows the input, then the model's global
    spectral tilt (estimated on sung frames only) is undone with strength `colour` (0..1). Returns (wav, report)."""
    wav, report = base["a2p"], {}
    if level:
        wav, report["level"] = level_match(a.wav[: len(wav)], wav, SR, level_config)
    if colour > 0:
        cfg = colour_config or ColourConfig()
        wav, report["colour"] = colour_match(a.wav[: len(wav)], wav, SR, ColourConfig(**{**cfg.__dict__, "strength": colour}),
                                             voiced_f0=a.f0)
    return wav, report
