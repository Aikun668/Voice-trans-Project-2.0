# F1 NeuralSVB PoC - technical report (v002)  | listening: pending_jade | technical: BLOCKED

Inputs: test_dataset/my_singing/{molihua,shige,zuixuan}_normal.wav (mono 44.1 kHz, untouched).
Model: NeuralSVB MleSVBVAE 1030_vae_mle@200k + official ASR/PPG 1009@136k + HiFi-GAN-NSF 1012@1.17M; seed 0.
F0 modified: **no** (model pitch input and vocoder F0 = input's own contour; F0 median identical in/out).
All files loudness-matched to a common target (-18 LUFS; original copy included in candidate/). v001 (first run, matched
to the hot original) is kept as-is: it failed true-peak/clipping because inputs are ~-11.5 LUFS / -1 dBTP.

Folders: `candidate/` = original, vocoder_only, a2p.  `diagnostics/` = a2a, a2p_nomap.
- vocoder_only: original mel -> HiFi-GAN only (shows what vocoder re-synthesis alone does)
- a2a / a2p_nomap: acoustic model without the Latent Mapping (identical to each other, as expected)
- a2p: full NeuralSVB.

See metrics_table.md and report.json. Findings (measured, thresholds NOT relaxed):
1. Pipeline runs end to end on the official weights; 12/12 outputs decode, no clipping, -3 dBTP headroom ok, duration gap <=12 ms.
2. **Pitch diagnostic fails on every variant, including vocoder_only** (median 7-18 cents, p95 66-170 vs limit 10/35),
   confirmed by a second tracker (Praat). Cause is HiFi-GAN re-synthesis jitter (+ acoustic model on top: zuixuan p95 106 -> 170),
   not the Latent Mapping. Cannot claim "F0 preserved" within the project's tolerance.
3. **Dynamic dropouts** (active-frame level 20-26 dB below median) appear only when the acoustic model is in the path
   (shige, zuixuan; 7-35 frames, i.e. 1-3% of active frames, mostly unvoiced/breath/phrase-end). vocoder_only has none.
   Latent Mapping makes it larger (a2p spread 10.8/15.8 dB vs 9.6/12.5 dB a2a).
4. Whether a2p sounds more "beautiful" is unknown - no one has listened. Per AGENTS.md, technical failure => technical_blocked,
   not handed to Jade for a listening verdict as a pass.
5. Official-pair check (outputs/official_preview_pairs): mel-L1 to the professional target, a2p vs DTW-aligned amateur:
   0.511 vs 0.511, 0.438 vs 0.509, 0.564 vs 0.617 (3 clips, tiny sample, not a perceptual metric).
