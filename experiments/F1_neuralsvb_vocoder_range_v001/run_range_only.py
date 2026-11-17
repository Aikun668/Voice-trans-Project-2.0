"""One range-only vocoder experiment; no acoustic-model or production-code edits."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import wave

import numpy as np
import torch
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'neuralsvb_f1')]
import f1_poc as qa
import neuralsvb_infer as n
import run_poc_batch as batch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    if (OUT / 'report.json').exists() or (OUT / 'diagnostics').exists():
        raise SystemExit('Existing run detected; use a new experiment directory.')
    source = ROOT / 'test_dataset/my_singing/zuixuan_normal.wav'
    source_hash = sha(source)
    old_files = list((ROOT / 'experiments/F1_neuralsvb_v002').rglob('*.wav'))
    old_hashes = {str(p.relative_to(ROOT)): sha(p) for p in old_files}
    original, rate = qa.read_pcm16(source)
    with wave.open(str(source), 'rb') as stream:
        assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) == (44100, 1, 2, 'NONE')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    hp = n._bootstrap(None)
    from vocoders.hifigan import HifiGAN
    from vocoders.pwg import PWG
    from data_gen.tts.data_gen_utils import process_utterance
    voc_config = yaml.safe_load((n.VOC_DIR / 'config.yaml').read_text())
    keys = ['audio_sample_rate', 'audio_num_mel_bins', 'fft_size', 'hop_size', 'win_size']
    assert all(hp[k] == voc_config[k] for k in keys)
    original_hp = dict(hp)
    wav, base_mel = PWG.wav2spec(str(source))
    candidate_wav, candidate_mel_transposed = process_utterance(
        str(source), fft_size=hp['fft_size'], hop_size=hp['hop_size'],
        win_length=hp['win_size'], num_mels=hp['audio_num_mel_bins'],
        fmin=voc_config['fmin'], fmax=voc_config['fmax'], sample_rate=hp['audio_sample_rate'],
        loud_norm=hp['loud_norm'], min_level_db=hp['min_level_db'],
        vocoder='pwg', eps=float(hp.get('wav2spec_eps', 1e-10)),
    )
    matched_mel = candidate_mel_transposed.T
    assert np.array_equal(wav, candidate_wav)
    assert matched_mel.shape == base_mel.shape and not np.array_equal(matched_mel, base_mel)
    assert len(base_mel) <= n.MAX_FRAMES, 'Long input requires a separate chunking design.'
    f0 = n._get_pitch(wav, base_mel, hp)
    assert len(f0) == len(base_mel)
    voc = n._build_vocoder(HifiGAN)
    assert all(hp[k] == original_hp[k] for k in keys + ['fmin', 'fmax'])
    records = {
        'hypothesis': 'Changing only mel range to vocoder-config range reduces reconstruction damage.',
        'source': str(source), 'source_sha256': source_hash,
        'interface': {'sample_rate': 44100, 'channels': 1, 'bits_per_sample': 16, 'encoding': 'pcm_s16le', 'frames': len(original)},
        'internal_rate': 22050, 'seed_before_each_vocoder_call': 0,
        'f0_shared_sha256': array_sha(f0), 'f0_intentionally_modified': False,
        'frame_count': len(base_mel), 'old_acoustic_four_frame_crop_used': False,
        'feature_method': 'official PWG process_utterance, log10 magnitude, center=True and zero edge padding unchanged',
        'baseline_range': [hp['fmin'], hp['fmax']], 'candidate_range': [voc_config['fmin'], voc_config['fmax']],
        'other_feature_parameters': {k: hp[k] for k in keys},
        'vocoder_checkpoint_sha256': sha(next(n.VOC_DIR.glob('model_ckpt_steps_*.ckpt'))),
        'vocoder_config_sha256': sha(n.VOC_DIR / 'config.yaml'),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'librosa', 'praat-parselmouth']},
        'limitations': ['Complete vocoder training feature recipe unconfirmed.', 'Current F0 time alignment retained.',
                        'Upsampling to 44.1 kHz does not restore lost high frequencies.',
                        'New seeded full-length baseline is not bit-identical to v002.',
                        'Agent cannot listen; metrics do not establish removal of rasp or audible jumps.'],
        'listening_status': 'pending_jade', 'outputs': {},
    }
    internal = OUT / 'internal'
    internal.mkdir()
    np.save(internal / 'f0_shared.npy', f0)
    full_rate = {'original': original}
    raw_stats = {}
    for name, mel in [('baseline_50_11025', base_mel), ('matched_0_8000', matched_mel)]:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        with torch.no_grad():
            raw = voc.spec2wav(mel, f0=f0)
        assert array_sha(f0) == records['f0_shared_sha256']
        assert np.isfinite(raw).all() and np.max(np.abs(raw)) > 1e-5
        np.save(internal / f'{name}_mel.npy', mel)
        np.save(internal / f'{name}_raw_22050.npy', raw)
        up = batch.resample(raw.astype(np.float64), 22050, rate)
        assert 0 <= len(up) - len(original) <= 2 * hp['hop_size']
        raw_stats[name] = {'native_frames': len(raw), 'excess_resampled_tail_samples_removed': len(up) - len(original),
                           'raw_peak_abs': float(np.max(np.abs(raw))),
                           'raw_samples_over_fullscale': int(np.sum(np.abs(raw) > 1)), 'mel_sha256': array_sha(mel)}
        full_rate[name] = up[:len(original)]
        print('GENERATED', name, 'frames', len(original), flush=True)
    measured = {k: qa.measure_loudness(v, rate) for k, v in full_rate.items()}
    target = batch.common_target(list(measured.values()))
    records['common_target_lufs'] = target
    folder = OUT / 'diagnostics'
    folder.mkdir()
    paths = {}
    for name, samples in full_rate.items():
        path = folder / f'zuixuan_normal__{name}.wav'
        qa.export_wav(samples, rate, target - measured[name]['integrated_lufs'], path)
        paths[name] = path
    ref, _ = qa.read_pcm16(paths['original'])
    loud_ref = qa.measure_loudness(ref, rate)
    for name, path in paths.items():
        actual, actual_rate = qa.read_pcm16(path)
        with wave.open(str(path), 'rb') as stream:
            contract = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes(), stream.getcomptype()) == (44100, 1, 2, len(original), 'NONE')
            pcm = stream.readframes(stream.getnframes())
        decoded = qa.run_ffmpeg(['-v', 'error', '-nostdin', '-xerror', '-i', str(path), '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1']).stdout
        assert decoded == pcm, 'Full-file decode differs from WAV payload'
        actual_loud = qa.measure_loudness(actual, actual_rate)
        entry = {'file': str(path.relative_to(OUT)), 'sha256': sha(path), 'contract_pass': contract,
                 'full_decode_matches_pcm': True, 'samples': qa.sample_stats(actual, actual_rate), 'loudness': actual_loud}
        if name != 'original':
            # Existing helper's informational Praat path expects native-rate output.
            # Analyze a resampled copy of the ACTUAL final PCM, not the pre-export raw.
            native_analysis = batch.resample(actual, actual_rate, 22050)
            checks = batch.check_output(SimpleNamespace(hp=hp), ref, rate, loud_ref, native_analysis, 22050)
            checks['analysis_rate_note'] = 'Existing helper analyzes actual final PCM resampled to 22050; interface is checked separately on final 44100 PCM.'
            if not contract:
                checks['errors'].append('Final file violates 44100/mono/PCM16/source-frame-count contract')
            if actual_loud['true_peak_dbtp'] > batch.TRUE_PEAK_LIMIT_DBTP:
                checks['errors'].append('Final 44100 PCM true peak exceeds existing limit')
            if abs(actual_loud['integrated_lufs'] - loud_ref['integrated_lufs']) > .2:
                checks['errors'].append('Final 44100 PCM loudness differs by >0.2 LU')
            checks['technical_pass'] = not checks['errors']
            entry.update({'checks': checks, 'generation': raw_stats[name]})
            print('CHECKED', name, json.dumps({'technical_pass': checks['technical_pass'], 'errors': checks['errors'], 'pitch': checks['pitch_vs_original']}, ensure_ascii=False), flush=True)
        records['outputs'][name] = entry
    assert sha(source) == source_hash
    assert all(sha(ROOT / p) == digest for p, digest in old_hashes.items())
    records['original_and_previous_experiment_hashes_unchanged'] = True
    records['preserved_old_wav_count'] = len(old_hashes)
    records['technical_pass'] = records['outputs']['matched_0_8000']['checks']['technical_pass']
    records['status'] = 'technical_pass_pending_jade' if records['technical_pass'] else 'technical_blocked'
    (OUT / 'report.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FINISHED', records['status'], 'report', OUT / 'report.json', flush=True)


if __name__ == '__main__':
    main()
