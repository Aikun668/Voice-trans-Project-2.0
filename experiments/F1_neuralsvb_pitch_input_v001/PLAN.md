# Route 1 listening hypothesis (registered before generation)

- Main source: test_dataset/my_singing/zuixuan_normal.wav.
- Existing feedback: vocoder and a2p are raspy; a2p jumps near 2,3,9,17–19 s and elsewhere. Preserve v002's appreciated beautification.
- Feature concern: Praat drops from approximately 316 to158 Hz near18.54 s; SHIGE diagnostic drops234 to116 Hz near16.65 s. An independent pYIN tracker disagrees; neither tracker alone is ground truth.
- Hypothesis: incorrect source pitch or voiced/unvoiced detection contributes to some resynthesis artifacts. Verify original periodic/harmonic evidence first; try one whole existing source-pitch frontend, without mixing tracks or correcting melody.
- Expected audible change: fewer unnatural jumps/rough transitions with the original full-model beautification still present. No promise that all rasp/identity shift will be fixed.
- Reject: new melody/octave changes, swallowed consonants, dropouts, loss of old beautification, more rasp or identity drift. Engineering thresholds unchanged; blocked files remain internal.
- Scope: one full-chain pYIN candidate against same-run Praat control; pure vocoder/a2a only internal. SHIGE source-feature evidence only.
- Listening: pending_jade. No agent listening capability; no approval filled on Jade's behalf.
