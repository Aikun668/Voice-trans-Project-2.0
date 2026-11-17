"""Read existing features/audio and live HTTP bytes; never generate candidate audio."""
from pathlib import Path
import sys, json, hashlib, urllib.request
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import f1_poc as qa

base = ROOT / 'experiments/F1_shige_neural_residual_v001'
manifest = json.loads((base / 'delivery_manifest.json').read_text(encoding='utf-8'))
result = {'http_files': [], 'delivered_measurements': {}, 'scope': 'read-only diagnostics; no audio generation'}
ref, sr = sf.read(base / manifest[0]['file'])
for item in manifest:
    path = base / item['file']
    data = urllib.request.urlopen('http://127.0.0.1:52876/' + item['file'], timeout=15).read()
    result['http_files'].append({
        'id': item['id'], 'expected_sha256': item['sha256'],
        'disk_matches': hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
        'http_matches': hashlib.sha256(data).hexdigest() == item['sha256']})
    x, rate = sf.read(path)
    if item['id'] != 'original':
        result['delivered_measurements'][item['id']] = {
            'correlation': float(np.corrcoef(ref, x)[0, 1]),
            'relative_residual_rms_db': float(10 * np.log10(np.mean((x-ref)**2) / np.mean(ref**2))),
            'identical_samples_fraction': float(np.mean(x == ref)), 'bands': qa.band_levels(x, rate)}
print('HTTP verified', flush=True)
f = np.load(base / 'diagnostics/neural_features.npz')
o = f['source_mel'].astype(float)
a = f['reconstruction_mel'].astype(float)
p = f['mapped_mel'].astype(float)
w = 10**(2*o)
w /= w.sum()
rms = lambda v: float(20 * np.sqrt(np.sum(w*v*v)))
unweighted = lambda v: float(20 * np.sqrt(np.mean(v*v)))
result['model_feature_decomposition'] = {
    'frames': len(o), 'full_a2p_minus_source_weighted_rms_db': rms(p-o),
    'reconstruction_minus_source_weighted_rms_db': rms(a-o), 'mapping_only_weighted_rms_db': rms(p-a),
    'full_a2p_minus_source_unweighted_rms_db': unweighted(p-o),
    'reconstruction_minus_source_unweighted_rms_db': unweighted(a-o),
    'mapping_only_unweighted_rms_db': unweighted(p-a),
    'feature_mel_reconstruction_exact': bool(np.array_equal(a, f['disabled_mapping_mel'])),
    'source_weighting': '10**(2*source_log10_mel); descriptive only, not a listening metric'}
b = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80, fmin=50, fmax=11025).astype(float)
s = b.sum(axis=0)
supported = s > 0
d = np.zeros((257, len(o)))
d[supported] = (b.T[supported] @ (p-a).T) / s[supported, None]
X = librosa.stft(ref, n_fft=1024, hop_length=256, win_length=1024, pad_mode='constant')
W = abs(X[:257])**2
W /= W.sum()
mean = np.sum(W*d)
perfreq = np.sum(W*d, axis=1) / np.maximum(W.sum(axis=1), 1e-30)
result['projected_mapping'] = {
    'weighted_global_mean_db': float(20*mean),
    'weighted_rms_db': float(20*np.sqrt(np.sum(W*d*d))),
    'weighted_rms_after_removing_constant_gain_db': float(20*np.sqrt(np.sum(W*(d-mean)**2))),
    'static_per_frequency_rms_db': float(20*np.sqrt(np.sum(W*(perfreq[:, None]-mean)**2))),
    'time_varying_residual_rms_db': float(20*np.sqrt(np.sum(W*(d-perfreq[:, None])**2))),
    'delivery_strengths': [.5, .75],
    'definition': 'Full-strength projected log gain weighted by source STFT power; not perceptual magnitude'}
source = ROOT / 'test_dataset/my_singing/shige_normal.wav'
legacy = json.loads((ROOT / 'experiments/F1_neuralsvb_v002/report.json').read_text(encoding='utf-8'))
clip = next(c for c in legacy['clips'] if Path(c['input']).stem == 'shige_normal')
result['legacy_hashes_ok'] = {
    k: hashlib.sha256((ROOT / 'experiments/F1_neuralsvb_v002' / v['file']).read_bytes()).hexdigest() == v['sha256']
    for k, v in clip['variants'].items()}
result['source_hash_ok'] = hashlib.sha256(source.read_bytes()).hexdigest() == clip['input_sha256']
aligned = ROOT / 'experiments/F1_neuralsvb_contract_v001/full_normal/shige_normal'
paths = {
    'original': source,
    'legacy_a2p': ROOT / 'experiments/F1_neuralsvb_v002' / clip['variants']['a2p']['file'],
    'legacy_vocoder': ROOT / 'experiments/F1_neuralsvb_v002' / clip['variants']['vocoder_only']['file'],
    'aligned_a2p': aligned / 'shige_normal__a2p.wav',
    'aligned_a2a': aligned / 'shige_normal__a2a.wav',
    'aligned_vocoder': aligned / 'shige_normal__vocoder.wav',
    'residual_standard': base / 'delivery/shige_normal__beautify_standard.wav'}
features = {}
for name, path in paths.items():
    x, rate = sf.read(path)
    if rate == 44100:
        x = resample_poly(x, 1, 2)
    X = librosa.stft(x, n_fft=512, hop_length=128, win_length=512, pad_mode='constant')
    features[name] = np.log10(np.maximum(b @ abs(X), 1e-10)).T
m = min(map(len, features.values()))
original = features['original'][:m]
weight = 10**(2*original)
weight /= weight.sum()
result['actual_audio_spectral_distances_db'] = {}
for name, mel in features.items():
    diff = mel[:m] - original
    offset = np.sum(weight*diff)
    result['actual_audio_spectral_distances_db'][name] = {
        'weighted_shape_rms_after_constant_gain_removed': float(20*np.sqrt(np.sum(weight*(diff-offset)**2))),
        'all_bins_unweighted_shape_rms_after_constant_gain_removed': float(20*np.sqrt(np.mean((diff-offset)**2)))}
target = HERE / 'measurements.json'
if target.exists():
    raise FileExistsError('Refusing to overwrite existing audit measurements')
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
