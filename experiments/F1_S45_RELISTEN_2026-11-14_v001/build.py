"""Repackage archived S45 audio with fixed gain only; all old files stay read-only."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ARCHIVE=PROJECT/'experiments/F1-EXP-020'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve()
    if HERE not in out.parents:raise ValueError('Output must be inside this experiment')
    if out.exists():raise FileExistsError(out)
    qa_path=PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/audio_qa.py'
    spec=importlib.util.spec_from_file_location('s45_audio_checks',qa_path)
    qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
    archive=json.loads((ARCHIVE/'qa/qa_report.json').read_text(encoding='utf-8'))
    mapping=json.loads((ARCHIVE/'qa/blind_key.json').read_text(encoding='utf-8'))['mapping_letter_to_version']
    if mapping.get('B')!='S45':raise ValueError('Archived version mapping changed')
    sources={'original':ARCHIVE/'listening/audio/original.wav','s45':ARCHIVE/'listening/audio/B.wav'}
    expected={'original':next(r for r in archive['exports'] if r['tag']=='original')['sha256'],
              's45':next(r for r in archive['versions'] if r['id']=='S45')['matched']['sha256']}
    for key,path in sources.items():
        if sha(path)!=expected[key]:raise ValueError('Archived file hash mismatch: '+str(path))
    protected=[PROJECT/'AGENTS.md',PROJECT/'docs/ROADMAP.md',*sources.values(),qa_path]
    protected.extend(p for p in ARCHIVE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md'))
    for root in (PROJECT/'f2',PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001'):
        protected.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.pt'))
    musicbert=PROJECT/'experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001'
    protected.extend(musicbert.glob('*.py'))
    protected.append(musicbert/'assets/checkpoint_last_musicbert_small.pt')
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in protected}
    out.mkdir();(out/'audio').mkdir()
    dump(out/'protected_files.json',hashes)
    arrays={};rates={};before={}
    for key,path in sources.items():
        x,sr=sf.read(io.BytesIO(path.read_bytes()),dtype='float64')
        if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Finite mono audio required')
        arrays[key]=x;rates[key]=sr;before[key]=qa.levels(x,sr)
    rate=rates['original'];source=arrays['original']
    if rates['s45']!=rate or arrays['s45'].shape!=source.shape:raise ValueError('Archived audio is not aligned')
    common=min(v['lufs'] for v in before.values())
    files={};checks={};decoded={}
    for key,x in arrays.items():
        gain_db=common-before[key]['lufs'];gain=10**(gain_db/20)
        path=out/'audio'/f'{key}.wav';sf.write(path,x*gain,rate,subtype='PCM_24')
        y,sr=sf.read(io.BytesIO(path.read_bytes()),dtype='float64');decoded[key]=y
        measured=qa.levels(y,sr)
        _,orig_pitch=qa.pitch(source,rate);_,pitch=qa.pitch(y,sr)
        shared=np.isfinite(orig_pitch)&np.isfinite(pitch);union=np.isfinite(orig_pitch)|np.isfinite(pitch)
        delta=float(np.median((pitch-orig_pitch)[shared]));overlap=float(shared.sum()/max(union.sum(),1))
        if len(y)!=len(source) or sr!=rate or not np.isfinite(y).all() or np.max(np.abs(y))>=1:raise ValueError('Export shape/finite/clipping failed')
        if not np.allclose(y,x*gain,atol=1.2e-7,rtol=0):raise ValueError('Export changed beyond fixed gain and PCM rounding')
        if abs(measured['lufs']-common)>.1 or measured['true_peak_dbtp']>-1:raise ValueError('Loudness or true-peak gate failed')
        if abs(delta)>=5 or overlap<.95:raise ValueError('Pitch/voicing engineering gate failed')
        checks[key]={'archive_file':str(sources[key]),'archive_sha256':expected[key],'sha256':sha(path),
                     'samples':len(y),'sample_rate':sr,'fixed_gain_db':gain_db,'clipped_samples':int((np.abs(y)>=1).sum()),
                     'finite':True,**measured,'pitch_center_delta_cents':delta,'voiced_overlap':overlap,
                     'p95_abs_pitch_delta_cents':float(np.percentile(np.abs((pitch-orig_pitch)[shared]),95)),
                     'only_fixed_gain_and_pcm_rounding':True}
        files[key]=f'audio/{key}.wav'
    if abs(checks['original']['lufs']-checks['s45']['lufs'])>.1:raise ValueError('Versions are not level matched')
    changed=[name for name,value in hashes.items() if sha(PROJECT/name)!=value]
    if changed:raise ValueError('Protected project files changed: '+str(changed))
    meta={'source_experiment':'F1-EXP-020','version':'S45','archive_mapping':'B = S45',
          'files':files,'duration':len(source)/rate,'common_lufs':common,
          'archived_report_target_lufs':archive['target_lufs'],'archived_audio_remeasured':before,
          'audio_checks':checks,'technical_pass':True,'listening':'pending_jade',
          's45_parameters_changed':False,'s45_algorithm_rerun':False,'model_loaded':False,
          'whole_key_shift':False,'new_limiter_or_effects':False,
          'protected_file_count':len(hashes),'protected_files_changed':changed}
    dump(out/'delivery.json',meta)
    page=(HERE/'page.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(page.replace('__METADATA__',json.dumps(meta,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
