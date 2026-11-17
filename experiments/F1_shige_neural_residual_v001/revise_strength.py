"""One targeted revision after the 0.94s local transient gate failed at strength=1."""
from pathlib import Path
import sys,json,hashlib,wave
import numpy as np
import parselmouth
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
import f1_poc as qa
from residual_adapter import render


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pitch_difference(a,b):
    tracks=[]
    for value in [a,b]:
        native=resample_poly(value,1,2)
        track=parselmouth.Sound(native,22050).to_pitch_ac(time_step=128/22050,
            pitch_floor=80,pitch_ceiling=750,voicing_threshold=.6).selected_array['frequency']
        tracks.append(track)
    ok=(tracks[0]>0)&(tracks[1]>0)
    cents=abs(1200*np.log2(tracks[1][ok]/tracks[0][ok]))
    return {'comparable_frames':int(ok.sum()),'median_abs_cents':float(np.median(cents)),
            'p95_abs_cents':float(np.percentile(cents,95))}


def local_check(a,b):
    size=882;count=len(a)//size
    af=a[:count*size].reshape(count,size);bf=b[:count*size].reshape(count,size)
    ar=np.sqrt(np.mean(af**2,axis=1));br=np.sqrt(np.mean(bf**2,axis=1))
    active=ar>max(ar.max()*.08,1e-4)
    ratio=np.max(abs(np.diff(bf,axis=1)),axis=1)/np.maximum(np.max(abs(np.diff(af,axis=1)),axis=1)*br/np.maximum(ar,1e-12),1e-12)
    gain=20*np.log10(np.maximum(br,1e-12)/np.maximum(ar,1e-12));gain-=np.median(gain[active])
    return {'max_local_jump_ratio_active':float(ratio[active].max()),
            'local_jump_flag_times_s':(np.where(active&(ratio>2.5))[0]*.02).tolist(),
            'min_relative_gain_active_db':float(gain[active].min())}


def main():
    previous=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
    source=Path(previous['source']); original,sr=qa.read_pcm16(source)
    assert sha(source)==previous['source_sha256'] and sr==44100
    features=np.load(HERE/'diagnostics/neural_features.npz')
    folder=HERE/'revision_strength075';folder.mkdir()
    raw={'original':original};metadata={}
    for name,strength in [('beautify_gentle',.5),('beautify_standard',.75)]:
        raw[name],metadata[name]=render(original,features['mapped_mel'],features['reconstruction_mel'],strength)
    measured={k:qa.measure_loudness(v,sr) for k,v in raw.items()}
    target=min(-18.,*[m['integrated_lufs']-3.1-m['true_peak_dbtp'] for m in measured.values()])
    files={}
    for name,value in raw.items():
        p=folder/f'shige_normal__{name}.wav';qa.export_wav(value,sr,target-measured[name]['integrated_lufs'],p);files[name]=p
    ref,_=qa.read_pcm16(files['original']); ref_loud=qa.measure_loudness(ref,sr)
    report={'source':str(source),'source_sha256':sha(source),'target_lufs':target,
            'revision_reason':'Strength1.0 had local transient ratio2.693 at0.94s; only stronger candidate strength changed to0.75.',
            'changed_control_only':'model_difference_strength:1.0->0.75',
            'counterfactual_identity_max_error':previous['counterfactual_identity_max_error'],
            'method':previous['method'],'official_default_waveform_pipeline':False,
            'listening_status':'pending_jade','outputs':{}}
    for name,p in files.items():
        actual,rate=qa.read_pcm16(p)
        with wave.open(str(p),'rb') as stream:
            assert (stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getnframes())==(44100,1,2,len(original))
            pcm=stream.readframes(stream.getnframes())
        assert qa.run_ffmpeg(['-v','error','-nostdin','-xerror','-i',str(p),'-f','s16le','pipe:1']).stdout==pcm
        loud=qa.measure_loudness(actual,rate)
        item={'file':str(p.relative_to(HERE)),'sha256':sha(p),'samples':qa.sample_stats(actual,rate),
              'loudness':loud,'contract_pass':True,'full_decode_pass':True}
        if name!='original':
            checks=qa.compare_audio(ref,actual,rate);local=local_check(ref,actual);pitch=pitch_difference(ref,actual)
            if local['local_jump_flag_times_s']: checks['errors'].append('New local sample transient ratio exceeds2.5')
            if local['min_relative_gain_active_db'] < -12: checks['errors'].append('Local active dropout below-12dB')
            if pitch['median_abs_cents']>10 or pitch['p95_abs_cents']>35: checks['errors'].append('Independent Praat pitch gate failed')
            if loud['true_peak_dbtp']>-3: checks['errors'].append('True peak exceeds-3dBTP')
            if abs(loud['integrated_lufs']-ref_loud['integrated_lufs'])>.2: checks['errors'].append('Loudness differs by>0.2LU')
            checks['passed']=not checks['errors']
            item.update({'checks':checks,'local_checks':local,'independent_pitch':pitch,'adaptation':metadata[name]})
            print(name,json.dumps({'checks':checks,'local':local,'praat':pitch},ensure_ascii=False),flush=True)
        report['outputs'][name]=item
    report['technical_pass']=all(x.get('checks',{'passed':True})['passed'] for x in report['outputs'].values())
    assert sha(source)==previous['source_sha256']
    (HERE/'revision_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REVISION_FINISHED technical_pass=',report['technical_pass'],flush=True)


if __name__=='__main__':
    main()
