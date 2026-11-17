"""Thin adapters for existing source-pitch extractors; never combine or tune notes."""
import numpy as np
import librosa
import parselmouth


def extract(audio, frame_count, method, sr=22050, hop=128):
    y = np.asarray(audio, dtype=np.float64)
    if sr != 22050 or hop != 128 or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('Expected finite mono 22050 Hz samples on a 128-sample grid')
    if abs(frame_count - (len(y)//hop + 1)) > 1:
        raise ValueError('Requested frame grid is outside waveform support')
    grid = np.arange(frame_count) * hop/sr
    meta = {'method': method, 'sr': sr, 'hop': hop, 'fmin': 80, 'fmax': 750,
            'melody_correction_applied': False, 'unvoiced_interpolated': False,
            'track_fusion_applied': False}
    if method == 'praat':
        track = parselmouth.Sound(y, sr).to_pitch_ac(
            time_step=hop/sr, voicing_threshold=.6, pitch_floor=80, pitch_ceiling=750)
        index = np.rint((grid-track.x1)/track.dx).astype(int)
        valid = (index>=0)&(index<len(track.selected_array['frequency']))
        f0 = np.zeros(frame_count)
        f0[valid] = track.selected_array['frequency'][index[valid]]
        meta.update({'first_tracker_center_seconds': float(track.x1),
                     'max_grid_error_seconds': float(np.max(abs(grid[valid]-track.xs()[index[valid]]))),
                     'outside_support_frames': int(np.sum(~valid))})
    elif method == 'pyin':
        f0, voiced, probability = librosa.pyin(
            y, sr=sr, fmin=80, fmax=750, frame_length=1024, hop_length=hop,
            center=True, fill_na=np.nan)
        if len(f0) < frame_count:
            raise ValueError('pYIN returned less than the requested support')
        f0 = np.where(voiced[:frame_count] & np.isfinite(f0[:frame_count]), f0[:frame_count], 0.)
        meta.update({'frame_length': 1024, 'first_tracker_center_seconds': 0.,
                     'max_grid_error_seconds': 0., 'outside_support_frames': 0,
                     'voiced_probability': probability[:frame_count].tolist()})
    else:
        raise ValueError('Unknown source-pitch method')
    meta['voiced_fraction'] = float(np.mean(f0>0))
    return f0, meta
