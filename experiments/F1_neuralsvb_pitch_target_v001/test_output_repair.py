import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import firwin2, fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colour_match import ColourConfig, colour_match, ltas_db  # noqa: E402
from dip_repair import HOP, DipConfig, repair_dips  # noqa: E402

SR = 22050


def noise(seconds=4.0, seed=0, level=0.1):
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal(int(SR * seconds))


def tilted(y, freqs_hz, gains_db):
    fir = firwin2(2049, np.array(freqs_hz) / (SR / 2), 10 ** (np.array(gains_db) / 20))
    return fftconvolve(y, fir, mode="same")


def band_db(y, lo, hi):
    f, d, _ = ltas_db(y, None, SR, ColourConfig(active_db=60))
    return d[(f >= lo) & (f < hi)].mean()


# ---- colour_match --------------------------------------------------------------------------------------------
def test_colour_match_recovers_a_known_tilt():
    ref = noise()
    shifted = tilted(ref, [0, 400, 1500, 6000, SR / 2], [-2, -2, 2, -5, -5])
    fixed, rep = colour_match(ref, shifted, SR)
    rel = lambda y, lo, hi: band_db(y, lo, hi) - band_db(y, 700, 2500)   # overall level is deliberately not restored
    for lo, hi in [(150, 400), (2000, 3500), (4500, 7000)]:
        assert abs(rel(fixed, lo, hi) - rel(ref, lo, hi)) < 1.0
    assert rep["max_abs_gain_db"] <= ColourConfig().max_gain_db + 1e-6


def test_colour_match_is_identity_when_nothing_changed():
    ref = noise()
    fixed, rep = colour_match(ref, ref.copy(), SR)
    assert np.allclose(fixed, ref, atol=1e-3)
    assert rep["max_abs_gain_db"] < 0.1


def test_colour_match_strength_zero_is_a_noop():
    ref = noise()
    shifted = tilted(ref, [0, 1000, SR / 2], [-4, 4, 4])
    fixed, _ = colour_match(ref, shifted, SR, ColourConfig(strength=0.0))
    assert np.allclose(fixed, shifted, atol=1e-3)


def test_colour_match_length_and_no_mutation():
    ref, other = noise(), noise(seed=1)
    before = other.copy()
    fixed, _ = colour_match(ref, other, SR)
    assert len(fixed) == len(other) and np.array_equal(other, before)


def test_colour_match_never_exceeds_max_gain():
    ref = noise()
    shifted = tilted(ref, [0, 500, 3000, SR / 2], [-20, -20, 0, 0])
    _, rep = colour_match(ref, shifted, SR, ColourConfig(max_gain_db=3.0))
    assert rep["max_abs_gain_db"] <= 3.0 + 1e-6


def test_overall_level_difference_does_not_use_up_the_gain_budget():
    ref = noise()
    quieter = tilted(ref, [0, 400, 1500, 6000, SR / 2], [-7, -7, -7, -12, -12])    # 7 dB quieter, plus a 5 dB high-band loss
    fixed, rep = colour_match(ref, quieter, SR, ColourConfig(max_gain_db=6.0))
    rel = lambda y: band_db(y, 4500, 7000) - band_db(y, 700, 2500)
    assert abs(rel(fixed) - rel(ref)) < 1.0 and rep["max_abs_gain_db"] <= 6.0 + 1e-6


# ---- dip_repair ----------------------------------------------------------------------------------------------
def make_case():
    ref = noise(3.0, seed=2, level=0.05)
    out = ref.copy()
    lo, hi = int(1.0 * SR), int(1.3 * SR)
    out[lo:hi] *= 10 ** (-24 / 20)                 # the model collapsed this stretch by 24 dB
    f0 = np.full(len(ref) // HOP + 1, 220.0)
    f0[int(0.9 * SR / HOP):int(1.4 * SR / HOP)] = 0.0   # ... and it is unvoiced there
    return ref, out, f0, (lo, hi)


def level_db(x):
    return 10 * np.log10(np.mean(np.asarray(x) ** 2) + 1e-12)


def test_dip_is_restored_in_unvoiced_stretch():
    ref, out, f0, (lo, hi) = make_case()
    fixed, rep = repair_dips(out, ref, f0, SR)
    assert level_db(ref[lo:hi]) - level_db(fixed[lo:hi]) < 3.0
    assert rep["frames_touched"] > 0


def test_voiced_frames_untouched_when_restricted():
    ref, out, f0, (lo, hi) = make_case()
    f0_voiced = np.full_like(f0, 220.0)
    fixed, rep = repair_dips(out, ref, f0_voiced, SR, DipConfig(include_voiced=False))
    assert np.allclose(fixed, out) and rep["frames_touched"] == 0


def test_collapse_inside_voiced_note_is_repaired_by_default():
    ref, out, f0, (lo, hi) = make_case()
    f0_voiced = np.full_like(f0, 220.0)
    fixed, rep = repair_dips(out, ref, f0_voiced, SR)
    assert level_db(ref[lo:hi]) - level_db(fixed[lo:hi]) < 3.0 and rep["frames_touched"] > 0


def test_mild_level_differences_are_left_alone():
    ref = noise(3.0, seed=4, level=0.05)
    out = ref * 10 ** (-5 / 20)          # 5 dB lower everywhere: below the 8 dB trigger
    fixed, rep = repair_dips(out, ref, np.full(len(ref) // HOP + 1, 220.0), SR)
    assert np.allclose(fixed, out) and rep["frames_touched"] == 0


def test_unaffected_audio_is_identical_far_from_the_dip():
    ref, out, f0, _ = make_case()
    fixed, _ = repair_dips(out, ref, f0, SR)
    assert np.allclose(fixed[: int(0.5 * SR)], out[: int(0.5 * SR)])
    assert np.allclose(fixed[int(2.0 * SR):], out[int(2.0 * SR):])


def test_no_dip_means_no_change():
    ref = noise(3.0, seed=3)
    f0 = np.zeros(len(ref) // HOP + 1)
    fixed, rep = repair_dips(ref.copy(), ref, f0, SR)
    assert np.allclose(fixed, ref) and rep["frames_touched"] == 0


def test_silent_reference_is_not_amplified():
    ref = np.zeros(SR * 2)
    out = np.zeros(SR * 2)
    fixed, rep = repair_dips(out, ref, np.zeros(SR * 2 // HOP + 1), SR)
    assert np.all(fixed == 0) and rep["frames_touched"] == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        repair_dips(np.zeros(100), np.zeros(101), np.zeros(2), SR)


def test_constant_level_offset_is_not_treated_as_a_dip():
    ref = noise(3.0, seed=5, level=0.05)
    out = ref * 10 ** (-7 / 20)                      # model output uniformly 7 dB quieter, like the real data
    fixed, rep = repair_dips(out, ref, np.full(len(ref) // HOP + 1, 220.0), SR)
    assert rep["frames_touched"] == 0 and np.allclose(fixed, out)
    assert rep["baseline_offset_db"] == pytest.approx(7.0, abs=0.3)


def test_dip_on_top_of_offset_is_restored_at_the_models_level_not_louder():
    ref = noise(3.0, seed=6, level=0.05)
    out = ref * 10 ** (-7 / 20)
    lo, hi = int(1.0 * SR), int(1.3 * SR)
    out[lo:hi] *= 10 ** (-20 / 20)
    fixed, rep = repair_dips(out, ref, np.full(len(ref) // HOP + 1, 220.0), SR)
    neighbour = level_db(out[int(0.5 * SR):int(0.8 * SR)])
    assert abs(level_db(fixed[lo + 500:hi - 500]) - neighbour) < 2.5   # matches the surrounding model level
    assert rep["frames_touched"] > 0


def test_quiet_passage_far_below_the_loud_end_is_not_repaired():
    ref = np.concatenate([noise(2.0, seed=7, level=0.1), noise(1.0, seed=8, level=0.001)])
    out = ref.copy()
    out[2 * SR:] *= 10 ** (-20 / 20)
    fixed, rep = repair_dips(out, ref, np.zeros(len(ref) // HOP + 1), SR)
    assert rep["frames_touched"] == 0


def test_voiced_only_estimate_ignores_unvoiced_frames():
    """Voiced half of the clip is tilted, the 'unvoiced' half is untouched: the curve must follow the voiced half only."""
    from colour_match import voiced_frame_mask  # noqa: F401
    ref = noise(4.0, seed=11)
    shifted = ref.copy()
    half = len(ref) // 2
    shifted[:half] = tilted(ref, [0, 400, 1500, 6000, SR / 2], [-4, -4, 3, -6, -6])[:half]
    f0 = np.zeros(len(ref) // 128 + 1)
    f0[: half // 128] = 200.0
    fixed_v, _ = colour_match(ref, shifted, SR, voiced_f0=f0)
    fixed_all, _ = colour_match(ref, shifted, SR)
    rel = lambda y, lo, hi: band_db(y[:half], lo, hi) - band_db(y[:half], 700, 1100)
    err = lambda y, lo, hi: abs(rel(y, lo, hi) - rel(ref, lo, hi))
    assert err(fixed_v, 2000, 3500) < err(fixed_all, 2000, 3500)
    assert err(fixed_v, 2000, 3500) < 1.5


def test_voiced_mask_marks_only_mostly_voiced_frames():
    from colour_match import voiced_frame_mask
    f0 = np.zeros(400)
    f0[100:300] = 150.0
    m = voiced_frame_mask(f0, 100, ColourConfig())          # frame i spans f0 frames [4i, 4i+16)
    assert not m[0] and not m[-1] and m[40] and not m[20]  # m[20] straddles the unvoiced start
