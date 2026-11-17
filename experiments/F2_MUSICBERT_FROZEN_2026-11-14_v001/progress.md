# MusicBERT frozen side experiment ledger

Plan: PLAN.md. User explicitly authorized inline execution; no fine-tuning.

- Setup: rules and predeclared thresholds saved before any model prediction. Baseline fingerprints saved for docs, f2, main trial and original XinQiang input.
- Ruling: all new files stay in this experiment, existing checkout/assets used read-only. No new worktree: main has extensive unrelated uncommitted/untracked work and must remain unchanged.
- Asset acquisition: GitHub API rate-limited (403); use pinned sparse Git clone from official remote instead. Existing Python environment has no pip; use uv with explicit experiment-local --target, no old environment mutation.
- Ruling: no subagent dispatched. Repository requires Haiku/high, unavailable in current tool allowlist. Strict rerun and separate verification entry point will provide reproducible review; no claim of independent human/agent review.

- Task A complete: official MusicBERT-small 265,448,349 bytes, SHA256 f5bb685afe128bc9e0e445be51b5333b98be6a5456df850bd9c9f2cc80bb7275. Pinned Microsoft commit 2b8739671ba06f819f31f568b8a79da581aaf6f9; HF adapter 64a054c791c2372e146d59cb0b04a5f5eba23fc0. All used parameters match pretrained tensors; zero trainable parameters, no optimizer.
- Task B complete: 10 original behavioral tests observed failing for absent adapter, then passing. Guard regression observed failing, then 11 tests passed after correcting data-path detection (torch/reference is source code, not vocal reference data). First failed inference_v001 kept; authoritative successful inference_v002 and fresh-process inference_replay_v001 saved.
- Task C complete: fixed 12-song / 192-corruption check: 33 restored, 15 restored under shuffled context; normal changes 8/48. Predeclared gates not passed; no threshold modifications or trial-specific tuning. Personal correction directions remain unverified; post-inference review used content boundaries only.
- Task D complete: two PCM24 exports level-matched at -18.5 LUFS, 48 kHz / 413,376 samples. Delivered pitch execution median error 0.756 cents, overlap 99.85%; independent raw-checkpoint numerical replay exact. Baseline 5,181 original files unchanged. Playback, position switching, end reset, seek, feedback branches and mobile layout verified via CLI fallback after in-app browser bootstrap failed. Full evidence in verification.json, browser_test_v001.log and DELIVERY_CHECK.md.
- Final ruling: deliver the user's requested true A/B as an explicitly failed effectiveness screen, never as successful repair. Technical gate passed; scientific and listening gates are not interchangeable. Do not begin fine-tuning.
