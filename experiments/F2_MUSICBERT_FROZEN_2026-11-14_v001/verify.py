"""Strict final readback: fresh-process replay, audio, source, main-line and controls."""
import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from adapter import HERE,PROJECT,sha,dump,_reuse


def main():
    baseline=json.loads((HERE/'baseline.json').read_text(encoding='utf-8'))
    changed=[name for name,expected in baseline['files'].items() if not (PROJECT/name).is_file() or sha(PROJECT/name)!=expected]
    if changed:raise ValueError('Existing project files changed: '+str(changed))
    first=HERE/'inference_v002';replay=HERE/'inference_replay_v001';delivery=HERE/'delivery_v001'
    record=json.loads((first/'record.json').read_text(encoding='utf-8'))
    if sha(HERE/'rules.md')!=record['rules_sha256']:raise ValueError('Predeclared rules changed')
    replay_checks={}
    with np.load(first/'inference.npz',allow_pickle=False) as a,np.load(replay/'inference.npz',allow_pickle=False) as b:
        for key in a.files:
            replay_checks[key]=bool(np.array_equal(a[key],b[key]))
        if not all(replay_checks.values()):raise ValueError('Fresh process differs')
        source=a['source'].copy();rendered=a['rendered'].copy();rate=int(a['sample_rate'])
        grid=a['time'].copy();command=a['command'].copy();mask=a['mask'].copy()
    if not np.array_equal(source[mask==0],rendered[mask==0]):raise ValueError('Unselected raw samples changed')
    qa=_reuse(PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/audio_qa.py','final_frozen_qa')
    meta=json.loads((delivery/'delivery.json').read_text(encoding='utf-8'))
    checks={};decoded={}
    for key,relative in meta['files'].items():
        path=delivery/relative;x,sr=sf.read(io.BytesIO(path.read_bytes()),dtype='float64')
        if x.shape!=source.shape or sr!=rate or not np.isfinite(x).all() or np.max(np.abs(x))>=1:raise ValueError('WAV readback failed')
        if sha(path)!=meta['audio_checks'][key]['sha256']:raise ValueError('Export hash changed')
        levels=qa.levels(x,sr)
        if abs(levels['lufs']-meta['common_lufs'])>.1 or levels['true_peak_dbtp']>-1:raise ValueError('Export level gate failed')
        _,before=qa.pitch(source,rate);_,after=qa.pitch(x,rate)
        shared=np.isfinite(before)&np.isfinite(after);union=np.isfinite(before)|np.isfinite(after)
        decoded[key]=x
        checks[key]={'sha256':sha(path),'samples':len(x),'rate':sr,**levels,
                     'voiced_overlap_vs_source':float(shared.sum()/max(union.sum(),1)),
                     'pitch_delta_median_cents':float(np.median((after-before)[shared]))}
        if checks[key]['voiced_overlap_vs_source']<.95:raise ValueError('Voiced overlap failed')
    delivered_pitch=qa.compare(decoded['original'],decoded['model'],rate,grid,command)
    if not delivered_pitch['pitch_technical_pass']:raise ValueError('Delivered pitch command failed')
    normal=json.loads((HERE/'evaluation_v001/normal_controls.json').read_text(encoding='utf-8'))
    corrupt=json.loads((HERE/'evaluation_v001/corrupt_controls.json').read_text(encoding='utf-8'))
    normal_changed=sum(bool(r['modified_over_50']) for r in normal)
    recovered=sum(bool(r['recovered_exact']) for r in corrupt)
    shuffled=sum(bool(r['shuffled_recovered_exact']) for r in corrupt)
    equivalence=json.loads((HERE/'model_verification_v001/model_equivalence.json').read_text(encoding='utf-8'))
    if not equivalence['hf_equivalence_pass']:raise ValueError('Numerical model verification failed')
    result={'existing_project_files_verified':len(baseline['files']),'existing_project_files_changed':changed,
            'mainline_unchanged':not changed,'predeclared_rules_unchanged':True,
            'fresh_process_replay_exact':all(replay_checks.values()),'replay_arrays':replay_checks,
            'raw_unselected_samples_exact':True,'audio_readback':checks,'delivered_pitch':delivered_pitch,
            'controls_recounted':{'normal_total':len(normal),'normal_changed_over_50':normal_changed,
               'corrupt_total':len(corrupt),'restored_exact':recovered,'shuffled_restored_exact':shuffled},
            'model_numerical_equivalence':equivalence,'training_steps':0,'technical_pass':True,
            'effectiveness_gate_pass':False,'fine_tuning_recommended_now':False,'listening':'pending_jade'}
    dump(HERE/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
