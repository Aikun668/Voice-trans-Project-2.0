"""Time/format adapters; reuse Praat and SciPy, no melody correction."""
import numpy as np
import parselmouth
from scipy.signal import resample_poly


def aligned_f0(audio, frame_count, sr=22050, hop=128, frame_offset_seconds=0.):
    track = parselmouth.Sound(audio, sr).to_pitch_ac(
        time_step=hop / sr, voicing_threshold=.6, pitch_floor=80, pitch_ceiling=750)
    grid = np.arange(frame_count) * hop / sr + frame_offset_seconds
    times = track.xs()
    index = np.rint((grid - track.x1) / track.dx).astype(int)
    valid = (index >= 0) & (index < len(times))
    f0 = np.zeros(frame_count)
    f0[valid] = track.selected_array['frequency'][index[valid]]
    error = np.abs(grid[valid] - times[index[valid]])
    return f0, {'first_praat_center_seconds': float(track.x1),
                'feature_frame_offset_seconds': frame_offset_seconds,
                'max_nearest_time_error_seconds': float(error.max()) if len(error) else None,
                'outside_praat_support_frames': int((~valid).sum()),
                'pitch_correction_applied': False, 'unvoiced_interpolated': False}


def to_external(native, source_frames):
    result = resample_poly(np.asarray(native, dtype=np.float64), 2, 1)
    if len(result) < source_frames:
        raise ValueError('Model output is shorter than input; refusing silent padding.')
    return result[:source_frames]
