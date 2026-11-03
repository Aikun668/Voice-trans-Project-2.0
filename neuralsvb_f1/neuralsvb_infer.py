"""NeuralSVB (official code + official weights) single-WAV inference wrapper for F1.

Reuses, without modification: MleSVBVAE (acoustic model + latent mapping + VCASR/PPG
encoder), HifiGAN-Singing vocoder, the official mel / F0 / speaker-embedding feature code.
Nothing here is trained or fine-tuned. See neuralsvb_f1/README.md for the known deviations.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party_neuralsvb" / "src"
VAE_DIR = SRC / "checkpoints" / "1030_vae_mle"
VOC_DIR = SRC / "checkpoints" / "1012_hifigan_all_songs_nsf"
MAX_FRAMES = 5000  # official hparams['max_frames']
FRAMES_MULTIPLE = 4  # official hparams['frames_multiple']
SAMPLE_RATE = 22050


@dataclass(frozen=True)
class F0Norm:
    """Dataset F0 statistics used by official `normalize_pitch: true` (train_f0s_mean_std.npy).

    The file is NOT shipped with the release. If no F0Norm is given we use identity
    (utterance's own mean/std), which preserves F0 but is an unverified deviation.
    """
    mean: float
    std: float


def _bootstrap(f0_norm: Optional[F0Norm]):
    """Make the official repo importable and load its hparams (cwd-relative paths inside)."""
    os.chdir(SRC)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from utils.hparams import hparams, set_hparams

    set_hparams(config=str(VAE_DIR / "config.yaml"), exp_name="", print_hparams=False)
    hparams["vocoder_ckpt"] = "checkpoints/1012_hifigan_all_songs_nsf"  # relative: official loader builds a regex from it
    hparams["infer"] = True
    if f0_norm is not None:
        hparams["f0_mean"], hparams["f0_std"] = f0_norm.mean, f0_norm.std
    return hparams


def praat_f0(wav: np.ndarray, hp: dict) -> np.ndarray:
    """Official Praat F0 for an arbitrary 22.05 kHz waveform (mel length implied by hop size)."""
    return _get_pitch(wav, np.zeros((len(wav) // hp["hop_size"] + 1, 1)), hp)


def _get_pitch(wav: np.ndarray, mel: np.ndarray, hp: dict) -> np.ndarray:
    """data_gen.tts.data_gen_utils.get_pitch (hop 128 branch) with ONE change: rpad is clamped
    to >= 0. Praat in parselmouth 0.4.7 yields a few more frames than 0.3.3 (the pinned version),
    making rpad negative; the official code truncates to len(mel) right after anyway."""
    import parselmouth

    step = hp["hop_size"] / hp["audio_sample_rate"]
    f0 = parselmouth.Sound(wav, hp["audio_sample_rate"]).to_pitch_ac(
        time_step=step, voicing_threshold=0.6, pitch_floor=80, pitch_ceiling=750
    ).selected_array["frequency"]
    assert hp["hop_size"] == 128
    lpad = 4 * 2
    rpad = max(len(mel) - len(f0) - lpad, 0)
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")
    delta = len(mel) - len(f0)
    assert abs(delta) <= 8, f"f0/mel length gap too large: {delta}"
    if delta > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta], 0)
    return f0[: len(mel)]


def _build_vocoder(cls):
    """Official HifiGAN.__init__ regex-matches glob() output with '/'; on Windows glob returns '\'.
    Normalise separators for the duration of the constructor only (official file untouched)."""
    import glob

    orig = glob.glob
    glob.glob = lambda *a, **k: [x.replace("\\", "/") for x in orig(*a, **k)]
    try:
        return cls()
    finally:
        glob.glob = orig


class NeuralSVB:
    def __init__(self, f0_norm: Optional[F0Norm] = None, device: str = "cpu"):
        self.hp = _bootstrap(f0_norm)
        self.f0_norm = f0_norm
        from modules.voice_conversion.svb_vae import MleSVBVAE
        from vocoders.hifigan import HifiGAN
        from resemblyzer import VoiceEncoder

        ckpt = torch.load(VAE_DIR / "model_ckpt_steps_200000.ckpt", map_location="cpu")
        sd = ckpt["state_dict"]["model"]
        dict_size = sd["vc_asr.token_embed.weight"].shape[0]  # 88 = len(phone_set)+10
        self.model = MleSVBVAE(dict_size)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(device)
        self.device = device
        self.global_step = ckpt["global_step"]
        self.vocoder = _build_vocoder(HifiGAN)
        self.voice_encoder = VoiceEncoder(device="cpu")

    # ---- features (official code paths) -------------------------------------------------
    def features(self, wav_path: str):
        from vocoders.pwg import PWG

        wav, mel = PWG.wav2spec(wav_path)  # official: 22.05 kHz, hop 128, log10-mel(80)
        f0 = _get_pitch(wav, mel, self.hp)  # official: praat autocorrelation, 80-750 Hz
        spk = self.voice_encoder.embed_utterance(wav)  # official: Resemblyzer, 256-d
        return wav, mel, f0, spk

    def _pitch_inputs(self, f0: np.ndarray):
        """Replicates FastSpeechDataset.__getitem__ for `normalize_pitch: true`.

        Returns (coarse pitch ids fed to the model, f0 fed to the vocoder, uv).
        In the official pipeline the SAME standardised f0 is fed to the vocoder.
        """
        from utils.pitch_utils import f0_to_coarse, norm_interp_f0, denorm_f0

        hp = self.hp
        f0 = f0.copy()
        voiced = f0 > 0
        if voiced.any() and f0[voiced].std() > 0:
            if self.f0_norm is None:
                # F1 keeps F0: standardise to the utterance's own mean/std -> identity transform.
                hp["f0_mean"], hp["f0_std"] = float(f0[voiced].mean()), float(f0[voiced].std())
            f0[voiced] = (f0[voiced] - f0[voiced].mean()) / f0[voiced].std() * hp["f0_std"] + hp["f0_mean"]
            f0[voiced] = f0[voiced].clip(min=60, max=900)
        pitch = f0_to_coarse(f0)
        f0n, uv = norm_interp_f0(f0, hp)
        f0_for_voc = denorm_f0(torch.FloatTensor(f0n), torch.FloatTensor(uv), hp).numpy()
        return pitch, f0_for_voc

    # ---- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def run(self, wav_path: str, mode: str = "a2p", seed: int = 0, keep_f0: bool = False):
        """mode: a2p (beautify) | a2p_nomap (latent mapping disabled) | a2a (VAE recon only)
        | vocoder (original mel straight through HiFi-GAN, no acoustic model).
        keep_f0: feed the vocoder the ORIGINAL f0 instead of the standardised one.
        """
        wav, mel, f0, spk = self.features(wav_path)
        n = min(len(mel), MAX_FRAMES)
        n -= n % FRAMES_MULTIPLE
        mel, f0 = mel[:n], f0[:n]
        pitch, f0_voc = self._pitch_inputs(f0)
        if keep_f0:
            f0_voc = f0
        if mode == "vocoder":
            return self.vocoder.spec2wav(mel, f0=f0), f0, f0

        dev = self.device
        mel_t = torch.FloatTensor(mel)[None].to(dev)
        pitch_t = torch.LongTensor(pitch)[None].to(dev)
        spk_t = torch.FloatTensor(spk)[None].to(dev)
        # Identity alignment: the "professional-side" F0 contour is the input's own contour.
        align = torch.arange(n, dtype=torch.long)[None].to(dev)
        torch.manual_seed(seed)
        out = self.model(
            amateur_mel=mel_t, prof_mel=mel_t,  # prof_mel only feeds the p2p branch -> MLE loss
            amateur_pitch=pitch_t, prof_pitch=pitch_t,
            amateur_spk_id=spk_t, prof_spk_id=spk_t,
            a2p_alignment=align, p2a_alignment=None,
            infer=False, concurrent_ways=["a2a", "p2p", "a2p"],
            disable_map=(mode == "a2p_nomap"),
        )
        way = "a2a" if mode == "a2a" else "a2p"
        mel_out = out[way]["mel_out"][0].cpu().numpy()
        return self.vocoder.spec2wav(mel_out, f0=f0_voc), f0, f0_voc


    @torch.no_grad()
    def run_pair(self, amateur_wav: str, prof_wav: str, seed: int = 0):
        """Official-style a2p on a real amateur/professional pair (PopBuTFy preview).

        Uses the official EHSADTW alignment on the two F0 contours and the professional pitch
        contour as target, exactly like the packed-test-set path. Requires F0Norm statistics.
        """
        from modules.voice_conversion.dtw.enhance_sadtw import EHSADTW

        assert self.f0_norm is not None, "pair mode needs dataset F0 statistics"
        _, a_mel, a_f0, a_spk = self.features(amateur_wav)
        _, p_mel, p_f0, _ = self.features(prof_wav)
        a_n = len(a_mel) - len(a_mel) % FRAMES_MULTIPLE
        p_n = len(p_mel) - len(p_mel) % FRAMES_MULTIPLE
        a_mel, a_f0, p_mel, p_f0 = a_mel[:a_n], a_f0[:a_n], p_mel[:p_n], p_f0[:p_n]
        _, alignment = EHSADTW(a_f0, p_f0, a_f0)  # official: len == len(prof)
        alignment = np.clip(alignment[:p_n], 0, a_n - 1)
        a_pitch, a_fv = self._pitch_inputs(a_f0)
        p_pitch, p_fv = self._pitch_inputs(p_f0)
        t = lambda x, d: torch.as_tensor(x, dtype=d)[None].to(self.device)  # noqa: E731
        torch.manual_seed(seed)
        out = self.model(
            amateur_mel=t(a_mel, torch.float32), prof_mel=t(p_mel, torch.float32),
            amateur_pitch=t(a_pitch, torch.long), prof_pitch=t(p_pitch, torch.long),
            amateur_spk_id=t(a_spk, torch.float32), prof_spk_id=t(a_spk, torch.float32),
            a2p_alignment=t(alignment, torch.long), p2a_alignment=None,
            infer=False, concurrent_ways=["a2a", "p2p", "a2p"], disable_map=False,
        )
        res = {w: out[w]["mel_out"][0].cpu().numpy() for w in ("a2a", "p2p", "a2p")}
        wavs = {
            "gt_a": self.vocoder.spec2wav(a_mel, f0=a_fv), "gt_p": self.vocoder.spec2wav(p_mel, f0=p_fv),
            "a2a": self.vocoder.spec2wav(res["a2a"], f0=a_fv), "p2p": self.vocoder.spec2wav(res["p2p"], f0=p_fv),
            "a2p": self.vocoder.spec2wav(res["a2p"], f0=p_fv),
        }
        return wavs, res, {"a": a_mel, "p": p_mel}, alignment


def beautify(input_wav: str, output_wav: str, f0_norm: Optional[F0Norm] = None, mode: str = "a2p",
             keep_f0: bool = False, seed: int = 0) -> dict:
    """Public F1 entry: read input WAV, write beautified 22.05 kHz WAV, return a run record."""
    input_wav, output_wav = str(Path(input_wav).resolve()), str(Path(output_wav).resolve())
    engine = NeuralSVB(f0_norm=f0_norm)  # chdirs into the official repo (relative ckpt paths)
    wav_out, f0_in, f0_voc = engine.run(input_wav, mode=mode, seed=seed, keep_f0=keep_f0)
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, wav_out, SAMPLE_RATE, subtype="PCM_16")
    v_in, v_out = f0_in[f0_in > 0], f0_voc[f0_voc > 0]
    return {
        "input": str(input_wav), "output": str(output_wav), "mode": mode, "keep_f0": keep_f0,
        "ckpt_global_step": engine.global_step,
        "f0_median_in_hz": float(np.median(v_in)) if len(v_in) else None,
        "f0_median_vocoder_hz": float(np.median(v_out)) if len(v_out) else None,
    }
