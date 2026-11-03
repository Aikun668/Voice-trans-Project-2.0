# F1 Route 1: source pitch-input investigation and full-chain trial

> Execute inline with superpowers:executing-plans. Jade authorized route 1. No further implementation confirmation is needed for this bounded experiment. No production integration, training or later route is authorized.

**Goal:** Verify suspected pitch-tracker octave errors on original recordings, then test one existing pitch extractor in the unchanged complete NeuralSVB generation path.

**Architecture:** New isolated experiment directory; reuse official model/vocoder, existing input features and engineering checks. Use complete Praat and complete librosa pYIN tracks as distinct controls, not a stitched/hybrid track. Diagnostic signal analysis does not modify audio.

**Tech Stack:** Existing third_party_neuralsvb/venv, librosa 0.9.2 pYIN, Praat/parselmouth 0.4.7, official NeuralSVB 7a80bc5 checkpoints, FFmpeg and SciPy. No dependency installation.

**Spec:** Root AGENTS.md, docs/F1_LISTENING_ACCEPTANCE.md, Jade's approval of route 1 and prohibition on residual/difference synthesis.

## Global constraints

- Main trial: zuixuan_normal. SHIGE pitch evidence only. Retain v002 listening baseline.
- Preserve all original audio, older experiments, current dirty work and production entrypoints; no Git reset, deletion, automatic commit or push.
- Preserve full NeuralSVB a2p mapping and original vocoder; no residual output, waveform blend, EQ, phase reconstruction, note correction, melody smoothing or manual event editing.
- New code only in experiments/F1_neuralsvb_pitch_input_v001; this directory is the isolated experimental workspace. Existing untracked wrappers, weights and runtime are read-only dependencies; an empty Git worktree would omit them, so do not relocate the checkout.
- Pre-register problem, event times, expected outcome and rejection criteria before generation.
- Initial trial plus at most one evidence-based revision. No threshold relaxation. Technical failure remains blocked and is not offered as an accepted listening candidate.

## Review focus

- pYIN's continuity prior must not erase real sung octave intervals.
- Zero pitch in genuine silence must not be interpolated into sustained singing.
- Tracker time grids must match mel frame centers, including tail frames.
- Apparent octave disagreement may be genuine subharmonic/creaky phonation: compare source harmonic evidence, not tracker agreement alone.
- Full-model dropouts may remain after source-pitch errors improve; do not hide them or claim global rasp is fixed.

## Task 1: pitch evidence and tested existing-extractor adapter

Files: experiment PLAN.md/feedback.json; test_pitch_frontend.py; pitch_frontend.py; investigate.py; evidence artifacts.

- [ ] Freeze SHA256 of source, wrappers and previous WAVs; register listening hypothesis.
- [ ] Write failing tests for frame-center alignment, true octave/silence preservation, and invalid rates/support.
- [ ] Run tests RED; implement thin adapters calling existing Praat/pYIN without track fusion; run GREEN.
- [ ] Extract full source tracks on main zuixuan and diagnostic SHIGE. Inspect original-window spectrum/periodicity at suspected events; preserve disagreement and uncertainty in report.

## Task 2: unchanged full-chain trial and gates

Files: run_trial.py; saved tracks/raw results; report.json; REPORT.md.

- [ ] Reuse model/vocoder with identical mel, speaker embedding, pad/crop, model seed and vocoder seed across controls. Only source pitch frontend differs; both use true time coordinates. Compare with frozen v002 separately, acknowledging legacy alignment differences.
- [ ] Generate Praat/pYIN pure vocoder, a2a and a2p internally for attribution. Only a2p is a beautification candidate.
- [ ] Export full-duration WAV at common gain-calibrated loudness, exact 44100/mono/PCM16 source frames; preserve raw native waveforms.
- [ ] Run existing pitch, gain/dropout, discontinuity, clipping, decode, true peak and loudness gates unchanged; add informative local event checks and independent source/output tracker comparisons without replacing failing primary gates.
- [ ] If pYIN full a2p fails, diagnose, document and stop at blocked; do not implement route 2 or produce an unapproved substitute.
- [ ] If full a2p passes, deliver original + v002 reference + one full-chain candidate with gain-matched labels and pending_jade; verify actual HTTP/browser media before opening.

## Execution ledger

- Initial: existing checkout is dirty; preserved, no integration changes. User authorized the bounded trial directly. The experiment folder provides file isolation and shared dependencies are read-only.
- Acceptance: source F0 fidelity, fewer reported jumps/rasp, old beautification remains, still sounds like Jade. Engineering checks do not establish subjective quality.
