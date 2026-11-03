"""CLI and reproducible matched A/B/C exports. Existing output dirs are refused."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from f1_poc import measure_loudness, export_wav, read_pcm16, compare_audio, sample_stats
from f1_adaptive_v0.analyzer import read_wav, analyze
from f1_adaptive_v0.decision_engine import load_rules, decide, fixed_baseline
from f1_adaptive_v0.dsp_processor import process


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, out, rules_path=None, disabled=(), stage='full'):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if stage not in ('analyze', 'decide', 'full'):
        raise ValueError('Unknown stage')
    x, sr, source_info = read_wav(source)
    source_hash = sha(source)
    features = analyze(x, sr)
    features['loudness'] = measure_loudness(x, sr)
    rules = load_rules(rules_path)
    decision = decide(features, rules)
    for key in disabled:
        if key not in ('eq', 'compressor', 'deesser'):
            raise ValueError(f'Unknown disabled effect: {key}')
        decision[key]['enabled'] = False
        decision[key]['override'] = 'Explicitly disabled by CLI'
    decision['mode'] = 'process' if any(decision[k]['enabled'] for k in ('eq', 'compressor', 'deesser')) else 'bypass'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'features.json', features)
    save(out / 'rules_snapshot.json', rules)
    if stage == 'analyze':
        return features
    save(out / 'decision.json', decision)
    if stage == 'decide':
        return decision
    report = {'schema_version': 'f1.run.v0', 'technical_pass': False,
              'listening_acceptance': 'pending_jade', 'status': 'running',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'source': {**source_info, 'sha256': source_hash}, 'disabled_effects': list(disabled),
              'versions': {k: version(k) for k in ('numpy', 'scipy', 'librosa', 'pedalboard', 'soundfile', 'PyYAML')},
              'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
              'outputs': [],
              'pitch_diagnostic_limit': 'Legacy autocorrelation 60–500 Hz; not a full-range pitch guarantee.',
              'hypothesis': 'Adaptive conservative compression may help objectively high-dynamic inputs; may bypass normal inputs.',
              'listening_target': 'If compression triggers: loud syllables less intrusive, quiet syllables easier to hear; expression retained.',
              'veto': ['dullness', 'pumping', 'lost consonants/tails', 'flattened expression', 'changed identity', 'new artifacts']}
    save(out / 'qa_report.json', report)
    try:
        fixed = fixed_baseline()
        for key in disabled:
            fixed[key]['enabled'] = False
        save(out / 'fixed_decision.json', fixed)
        variants = [('original_matched.wav', x),
                    ('fixed_baseline.wav', process(x, sr, fixed)),
                    ('adaptive.wav', process(x, sr, decision))]
        meters = [measure_loudness(y, sr) for _, y in variants]
        # One common target; constant gain only. No hidden limiter/makeup processing.
        target = min(-18., *(m['integrated_lufs'] - 3.2 - m['true_peak_dbtp'] for m in meters))
        report['target_lufs'] = target
        errors = []
        if features['full_scale_samples']:
            errors.append('Input has full-scale samples: clipping risk requires review.')
        for (name, y), meter in zip(variants, meters):
            gain = target - meter['integrated_lufs']
            export_wav(y, sr, gain, out / name)
            actual, rate = read_pcm16(out / name)
            loudness = measure_loudness(actual, rate)
            check = compare_audio(x, actual, sr)
            if rate != sr:
                check['errors'].append('Sample rate changed')
            if loudness['true_peak_dbtp'] > -2.9:
                check['errors'].append('True peak exceeds -2.9 dBTP')
            if abs(loudness['integrated_lufs'] - target) > .3:
                check['errors'].append('Loudness match exceeds 0.3 LU')
            check['passed'] = not check['errors']
            errors.extend(f'{name}: {e}' for e in check['errors'])
            raw_delta = float(np.sqrt(np.mean((y - x) ** 2)))
            report['outputs'].append({'file': name, 'sha256': sha(out / name), 'gain_db': gain,
                                      'raw_dsp_delta_rms': raw_delta,
                                      'raw_rms_reduction_db': float(20 * np.log10(np.linalg.norm(x) / max(np.linalg.norm(y), 1e-12))),
                                      'samples': sample_stats(actual, rate), 'loudness': loudness, 'integrity': check})
        loud = [p['loudness']['integrated_lufs'] for p in report['outputs']]
        if max(loud) - min(loud) > .3:
            errors.append('A/B/C loudness spread exceeds 0.3 LU')
        if sha(source) != source_hash:
            errors.append('Source hash changed during run')
        report.update(technical_pass=not errors, status='technical_pass' if not errors else 'technical_blocked', errors=errors)
        (out / 'LISTENING.md').write_text(
            '# A/B/C experiment — listening pending\n\n'
            f"Technical status: {report['status']}. Adaptive: {decision['mode']}.\n\n"
            'A original_matched.wav: original with constant gain only.\n'
            'B fixed_baseline.wav: predeclared gentle compressor comparator.\n'
            'C adaptive.wav: feature-driven decision; may be identical to A.\n\n'
            'Keep player volume fixed. Compare full aligned WAVs. Prefer A/B/C/no difference; '
            'report dullness, pumping, lost words/tails, flattened expression or altered identity with timestamps.\n'
            'No direct agent listening. Technical pass does not establish audible benefit. '
            'If technically blocked, files are diagnostic artifacts and must not count as accepted candidates.\n', encoding='utf-8')
        save(out / 'qa_report.json', report)
        return report
    except Exception as exc:
        report.update(status='failed', error=str(exc), technical_pass=False)
        save(out / 'qa_report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New directory; must not exist')
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--disable', nargs='*', choices=('eq', 'compressor', 'deesser'), default=[])
    parser.add_argument('--stage', choices=('analyze', 'decide', 'full'), default='full')
    args = parser.parse_args()
    result = run(args.input, args.output, args.rules, args.disable, args.stage)
    print(json.dumps({'output': str(args.output.resolve()), 'status': result.get('status', args.stage),
                      'technical_pass': result.get('technical_pass')}, ensure_ascii=False))
    return 2 if result.get('technical_pass') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
