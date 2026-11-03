# NeuralSVB single-WAV F1 PoC

Wraps the **official** NeuralSVB code and weights (https://github.com/MoonInTheRiver/NeuralSVB,
commit `7a80bc5`, GPL-3.0). No training, no fine-tuning, no DSP effects. Official files are untouched.
Listening status of every output: **pending_jade** (the agent cannot listen; measurements are not "sounds better").

## Setup (isolated; does not touch other Python envs)
```
git clone --depth 1 https://github.com/MoonInTheRiver/NeuralSVB third_party_neuralsvb/src
uv venv --python 3.10 third_party_neuralsvb/venv && source third_party_neuralsvb/venv/Scripts/activate
uv pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy==1.23.5 "scipy<1.12" librosa==0.9.2 pyyaml tqdm matplotlib praat-parselmouth h5py \
  scikit-learn soundfile resemblyzer pyloudnorm numba six tensorboardX tensorboard "protobuf<5" \
  scikit-image chardet pycwt "setuptools<81"
```
numpy must stay 1.23.x: the official `f0_to_coarse` uses `np.int`.
Weights (release `pre-release`) unzipped into `third_party_neuralsvb/src/checkpoints/`:
`1030_vae_mle` (acoustic model + Latent Mapping, step 200000), `1009_pretrain_asr_english` (PPG/ASR encoder, step 136000;
its weights are bit-identical to `vc_asr.*` inside 1030), `1012_hifigan_all_songs_nsf` (vocoder, step 1170000).
Note: the ASR/vocoder zips have no top-level folder - unzip into a folder of the same name.

## Run
```
python neuralsvb_f1/beautify.py IN.wav OUT.wav            # a2p beautification, 22.05 kHz mono PCM16
python neuralsvb_f1/run_poc_batch.py --out experiments/NEW_DIR a.wav b.wav c.wav   # 4 variants + checks
python neuralsvb_f1/validate_official_pair.py              # official a2p path on PopBuTFy preview pairs
```
CPU: ~13 s for a 5 s clip (model load included). Max 5000 mel frames (~29 s) per call (official `max_frames`).

## What the model actually needs (from code, `svb_vae.py::MleSVBVAE`, `svb_vae_task.py`)
- amateur mel -> PPG-like content (`vc_asr`), amateur pitch, 256-d Resemblyzer speaker embedding.
- a2p decode = `decoder(z_mapping(z_amateur), cond = [professional pitch contour, amateur content gathered by
  a2p alignment, amateur speaker emb])`. The professional **mel** only feeds the p2p branch / MLE loss, not the a2p output.
- So a professional *audio* reference is NOT required, but a professional **pitch contour + alignment** is.
  Single-WAV mode uses the input's own F0 and an identity alignment (F0 kept; no "professional" target contour exists).
  This is an adaptation, not an official inference mode (the official README says "Inference from raw inputs: WIP").
- The latent is a single global vector per utterance (GlobalFVAE), so mapping acts as a global style shift.

## Deviations / unverified (no guessing)
1. `train_f0s_mean_std.npy` (dataset F0 stats), `phone_set.json`, the packed binary test set are not released.
   Single-WAV mode standardises F0 to the utterance's own mean/std (identity). `validate_official_pair.py` uses the pooled
   preview-clip F0 as a proxy. dict_size=88 recovered from the checkpoint.
2. Praat via parselmouth 0.4.7 (pinned 0.3.3 unavailable): returns a few extra frames; `rpad` clamped to >=0 (official code
   truncates to len(mel) anyway). Possible 1-2 frame (6-12 ms) F0 offset is unverified.
3. Windows: official vocoder loader regex needs '/' paths; a glob shim is used during construction only.
4. Models are trained on English PopBuTFy (24 singers); your clips are Chinese - PPG/content quality there is untested.
5. `z_q` is sampled in the encoder; seed fixed to 0 (output is stochastic across seeds).
