"""Isolated staged contract investigation and full model checks."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys
import wave
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / 'neuralsvb_f1')]
import feature_contract as contract
import neuralsvb_infer as n
import run_poc_batch as batch
import f1_poc as qa


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(message):
    print(message, flush=True)
    with (HERE / 'execution.log').open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')


def checked_exports(source, raws, folder, hp):
    folder.mkdir(parents=True, exist_ok=False)
    before, sr = qa.read_pcm16(source)
    assert sr == 44100
    items = {'original': before}
    for name, raw in raws.items():
        assert np.isfinite(raw).all()
        np.save(folder / (name + '_native.npy'), raw)
        items[name] = contract.to_external(raw, len(before))
    loud = {k: qa.measure_loudness(v, sr) for k, v in items.items()}
    target = batch.common_target(list(loud.values()))
    actuals = {}
    paths = {}
    for name, value in items.items():
        path = folder / (source.stem + '__' + name + '.wav')
        qa.export_wav(value, sr, target - loud[name]['integrated_lufs'], path)
        paths[name] = path
        actuals[name], out_rate = qa.read_pcm16(path)
        assert out_rate == sr
    ref = actuals['original']
    loud_ref = qa.measure_loudness(ref, sr)
    report = {'source': str(source), 'source_sha256': sha(source), 'target_lufs': target,
              'listening_status': 'pending_jade', 'outputs': {}}
    for name, path in paths.items():
        value = actuals[name]
        with wave.open(str(path), 'rb') as stream:
            spec = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes(), stream.getcomptype())
            pcm = stream.readframes(stream.getnframes())
        assert spec == (44100, 1, 2, len(before), 'NONE')
        decoded = qa.run_ffmpeg(['-v', 'error', '-nostdin', '-xerror', '-i', str(path), '-f', 's16le', 'pipe:1']).stdout
        assert decoded == pcm
        final_loud = qa.measure_loudness(value, sr)
        item = {'file': path.name, 'sha256': sha(path), 'samples': qa.sample_stats(value, sr),
                'loudness': final_loud, 'interface_pass': True, 'full_decode_pass': True}
        if name != 'original':
            native = batch.resample(value, 44100, 22050)
            c = batch.check_output(SimpleNamespace(hp=hp), ref, sr, loud_ref, native, 22050)
            if final_loud['true_peak_dbtp'] > batch.TRUE_PEAK_LIMIT_DBTP:
                c['errors'].append('Final external true peak exceeds -3 dBTP')
            if abs(final_loud['integrated_lufs'] - loud_ref['integrated_lufs']) > .2:
                c['errors'].append('Final external loudness differs by >0.2 LU')
            c['technical_pass'] = not c['errors']
            item['checks'] = c
            log(json.dumps({'clip': source.stem, 'variant': name, 'pass': c['technical_pass'],
                            'errors': c['errors'], 'pitch': c['pitch_vs_original'],
                            'praat': c['pitch_vs_original_praat_info_only']}, ensure_ascii=False))
        report['outputs'][name] = item
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def torch_features(wav, hp):
    from modules.hifigan.mel_utils import mel_spectrogram
    original_stft = torch.stft
    # Compatibility only: official helper expects a real/imag pair, PyTorch 2 requires explicit flag.
    def compatible_stft(*args, **kwargs):
        kwargs.setdefault('return_complex', False)
        return original_stft(*args, **kwargs)
    torch.stft = compatible_stft
    try:
        with torch.no_grad():
            result = mel_spectrogram(torch.tensor(wav, dtype=torch.float32)[None], hp)
        return result[0].numpy().T
    finally:
        torch.stft = original_stft


def bridge_mel(mel, hp, voc_hp):
    import librosa
    from librosa.feature.inverse import mel_to_stft
    # Approximate magnitude-domain bridge. Never interpolate log-bin indices as if frequencies matched.
    magnitude = mel_to_stft(np.power(10., mel.T), sr=22050, n_fft=512, power=1.,
                            fmin=hp['fmin'], fmax=hp['fmax'])
    basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80,
                                fmin=voc_hp['fmin'], fmax=voc_hp['fmax'])
    converted = np.log(np.maximum(basis @ magnitude, 1e-5)).T
    # New frame centers are half a hop later; align time coordinates without changing duration.
    grid = np.arange(len(converted))
    return np.stack([np.interp(grid + .5, grid, converted[:, k]) for k in range(80)], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['probe', 'full'], required=True)
    ap.add_argument('--profile', choices=['pwg', 'torch'], default='pwg')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--clips', nargs='+', default=['zuixuan_normal'])
    args = ap.parse_args()
    output = HERE / args.run_name
    if output.exists():
        raise SystemExit('Run directory already exists; refusing overwrite.')
    paths = [ROOT / 'test_dataset/my_singing' / (name + '.wav') for name in args.clips]
    source_hashes = {str(p): sha(p) for p in paths}
    preserved = {str(p): sha(p) for folder in (ROOT / 'experiments').glob('F1_neuralsvb*')
                 if folder != HERE for p in folder.rglob('*.wav')}
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    engine = n.NeuralSVB()
    hp = engine.hp
    report = {'stage': args.stage, 'profile': args.profile, 'clips': [], 'listening_status': 'pending_jade',
              'no_melody_or_tempo_correction': True, 'f0_adapter': 'Praat frame-center mapping; no note correction',
              'interface': '44100/mono/PCM16, exact source sample count',
              'thresholds': {'pitch_median_cents': 10, 'pitch_p95_cents': 35, 'gain_spread_db': 8, 'min_gain_db': -12, 'jump_ratio': 2.5}}
    for source in paths:
        log('START ' + source.stem + ' stage=' + args.stage + ' profile=' + args.profile)
        wav, mel, old_f0, spk = engine.features(str(source))
        f0, time_info = contract.aligned_f0(wav, len(mel))
        log(json.dumps({'clip': source.stem, 'time_alignment': time_info}))
        folder = output / source.stem
        if args.profile == 'torch':
            voc_mel = torch_features(wav, engine.vocoder.config)
            f0_voc, _ = contract.aligned_f0(wav, len(voc_mel), frame_offset_seconds=64 / 22050)
        else:
            voc_mel, f0_voc = mel, f0
        torch.manual_seed(0)
        raws = {'vocoder': engine.vocoder.spec2wav(voc_mel, f0=f0_voc)}
        log('GENERATED ' + source.stem + ' vocoder')
        if args.stage == 'full':
            from utils.pitch_utils import f0_to_coarse
            count = len(mel)
            pad = (-count) % 4
            padded_mel = np.pad(mel, ((0, pad), (0, 0)), mode='edge')
            padded_f0 = np.pad(f0, (0, pad), mode='edge')
            pitch = torch.tensor(f0_to_coarse(padded_f0), dtype=torch.long)[None]
            mel_t = torch.tensor(padded_mel, dtype=torch.float32)[None]
            spk_t = torch.tensor(spk, dtype=torch.float32)[None]
            align = torch.arange(len(padded_mel), dtype=torch.long)[None]
            torch.manual_seed(0)
            with torch.no_grad():
                out = engine.model(amateur_mel=mel_t, prof_mel=mel_t,
                    amateur_pitch=pitch, prof_pitch=pitch, amateur_spk_id=spk_t, prof_spk_id=spk_t,
                    a2p_alignment=align, p2a_alignment=None, infer=False,
                    concurrent_ways=['a2a', 'p2p', 'a2p'], disable_map=False)
            for way in ['a2a', 'a2p']:
                target_mel = out[way]['mel_out'][0].detach().numpy()[:count]
                if args.profile == 'torch':
                    target_mel = bridge_mel(target_mel, hp, engine.vocoder.config)
                torch.manual_seed(0)
                raws[way] = engine.vocoder.spec2wav(target_mel, f0=f0_voc[:count])
                log('GENERATED ' + source.stem + ' ' + way)
            # One bounded, explicit inference revision: posterior mean replaces the sampled latent.
            with torch.no_grad():
                cond = engine.model.prepare_condition(mel_t, pitch, spk_t)
                joined = torch.cat([cond['h_pitch'], cond['h_content'], cond['h_style']], -1)
                g = engine.model.encoded_embed_proj(joined).transpose(1, 2)
                mean = out['a2a']['m_q']
                mapped = engine.model.z_mapping_function(mean, cond['h_style'].transpose(1, 2))
                mask = cond['tgt_nonpadding'].transpose(1, 2)
                for name, latent in [('a2a_mean', mean), ('a2p_mean', mapped)]:
                    target_mel = engine.model.vae_model.decoder(latent, mask, g=g)[0].transpose(0, 1).numpy()[:count]
                    if args.profile == 'torch':
                        target_mel = bridge_mel(target_mel, hp, engine.vocoder.config)
                    torch.manual_seed(0)
                    raws[name] = engine.vocoder.spec2wav(target_mel, f0=f0_voc[:count])
                    log('GENERATED ' + source.stem + ' ' + name)
        checked = checked_exports(source, raws, folder, hp)
        checked['f0_time_alignment'] = time_info
        checked['profile'] = args.profile
        np.save(folder / 'f0_aligned.npy', f0)
        np.save(folder / 'f0_legacy.npy', old_f0)
        (folder / 'report.json').write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding='utf-8')
        report['clips'].append(checked)
    assert all(sha(p) == digest for p, digest in source_hashes.items())
    assert all(sha(p) == digest for p, digest in preserved.items())
    report['source_hashes_unchanged'] = True
    report['previous_wav_files_unchanged'] = len(preserved)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    log('FINISHED ' + str(output))


if __name__ == '__main__':
    main()
