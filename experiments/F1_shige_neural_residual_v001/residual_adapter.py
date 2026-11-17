"""Thin neural-feature difference adapter using librosa's existing transforms.

This is an experimental project adaptation, not official NeuralSVB waveform inference.
"""
import numpy as np
import librosa


def render(source, mapped_mel, reconstructed_mel, strength):
    source=np.asarray(source,dtype=np.float64)
    mapped=np.asarray(mapped_mel,dtype=np.float64)
    reconstructed=np.asarray(reconstructed_mel,dtype=np.float64)
    if mapped.shape != reconstructed.shape or mapped.ndim != 2 or mapped.shape[1] != 80:
        raise ValueError('Expected paired [frames,80] model features with identical conditions.')
    if not np.isfinite(mapped).all() or not np.isfinite(reconstructed).all():
        raise ValueError('Nonfinite model features')
    stft=librosa.stft(source,n_fft=1024,hop_length=256,win_length=1024,center=True,pad_mode='constant')
    if stft.shape[1] != mapped.shape[0]:
        raise ValueError(f'Model/source time-grid mismatch: {mapped.shape[0]} vs {stft.shape[1]}')
    basis=librosa.filters.mel(sr=22050,n_fft=512,n_mels=80,fmin=50,fmax=11025).astype(np.float64)
    weights=basis.sum(axis=0)
    supported=weights>0
    delta=mapped-reconstructed
    linear_delta=np.zeros((257,len(mapped)),dtype=np.float64)
    linear_delta[supported]=(basis.T[supported] @ delta.T) / weights[supported,None]
    gain=np.power(10.,float(strength)*linear_delta)
    if not np.isfinite(gain).all():
        raise ValueError('Unbounded model-derived gain')
    modified=stft.copy()
    modified[:257] *= gain
    result=librosa.istft(modified,hop_length=256,win_length=1024,center=True,length=len(source))
    return result, {'strength':float(strength),'model_delta_rms_db':float(20*np.sqrt(np.mean(delta**2))),
                    'applied_log_gain_rms_db':float(20*strength*np.sqrt(np.mean(linear_delta**2))),
                    'applied_gain_db_percentiles':np.percentile(20*strength*linear_delta,[0,5,50,95,100]).tolist(),
                    'original_phase_retained':True,'original_band_above_11025_retained':True,
                    'input_frames':len(source),'output_frames':len(result),
                    'official_vocoder_used_for_output':False,'method':'NeuralSVB mapped-minus-reconstruction log-magnitude gain applied with original phase'}
