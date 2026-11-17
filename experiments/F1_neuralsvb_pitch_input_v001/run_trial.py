"""Complete official NeuralSVB generation; only source-pitch frontend differs."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,os,wave
import numpy as np
import torch
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'neuralsvb_f1')]
import neuralsvb_infer as n
import run_poc_batch as batch
import f1_poc as qa


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_diagnostics(before,after,sr=44100):
    size=round(sr*.02);count=min(len(before),len(after))//size
    b=before[:count*size].reshape(count,size)
    a=after[:count*size].reshape(count,size)
    br=np.sqrt(np.mean(b*b,axis=1));ar=np.sqrt(np.mean(a*a,axis=1))
    active=br>max(br.max()*.08,1e-4)
    gains=20*np.log10(np.maximum(ar,1e-12)/np.maximum(br,1e-12))
    gains-=np.median(gains[active])
    ratio=np.max(abs(np.diff(a,axis=1)),axis=1)/np.maximum(np.max(abs(np.diff(b,axis=1)),axis=1)*ar/np.maximum(br,1e-12),1e-12)
    drops=np.where(active&(gains < -12))[0]
    jumps=np.where(active&(ratio>2.5))[0]
    return {'dropout_frames':[{'time_s':round(float(i*.02),3),'gain_db':round(float(gains[i]),2)} for i in drops],
            'max_local_jump_ratio_active':float(np.max(ratio[active])),
            'local_jump_times_s':(jumps*.02).tolist(),
            'caution':'Local sample-difference metric is not a complete perceptual jump detector.'}


def main():
    output=HERE/'trial';output.mkdir(exist_ok=False)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    preserved=json.loads((HERE/'preservation_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==digest for p,digest in preserved.items())
    source=ROOT/'test_dataset/my_singing/zuixuan_normal.wav'
    original,rate=qa.read_pcm16(source)
    engine=n.NeuralSVB()
    wav,mel,_,spk=engine.features(str(source))
    evidence=np.load(HERE/'evidence/zuixuan_normal.npz')
    assert np.array_equal(mel,evidence['mel']) and np.array_equal(wav,evidence['wav'])
    count=len(mel);pad=(-count)%4
    from utils.pitch_utils import f0_to_coarse
    mel_t=torch.tensor(np.pad(mel,((0,pad),(0,0)),mode='edge'),dtype=torch.float32)[None]
    speaker=torch.tensor(spk,dtype=torch.float32)[None]
    align=torch.arange(count+pad,dtype=torch.long)[None]
    raw={'original':original};input_records={}
    for frontend in ['praat','pyin']:
        f0=evidence[frontend].copy()
        torch.manual_seed(0)
        raw[frontend+'_vocoder']=engine.vocoder.spec2wav(mel,f0=f0)
        pitch=torch.tensor(f0_to_coarse(np.pad(f0,(0,pad),mode='edge')),dtype=torch.long)[None]
        torch.manual_seed(0)
        with torch.no_grad():
            result=engine.model(amateur_mel=mel_t,prof_mel=mel_t,amateur_pitch=pitch,prof_pitch=pitch,
                amateur_spk_id=speaker,prof_spk_id=speaker,a2p_alignment=align,p2a_alignment=None,
                infer=False,concurrent_ways=['a2a','p2p','a2p'],disable_map=False)
        for route in ['a2a','a2p']:
            predicted=result[route]['mel_out'][0].cpu().numpy()[:count]
            np.save(output/f'{frontend}_{route}_mel.npy',predicted)
            torch.manual_seed(0)
            raw[frontend+'_'+route]=engine.vocoder.spec2wav(predicted,f0=f0)
        input_records[frontend]={'f0_sha256':hashlib.sha256(f0.tobytes()).hexdigest(),
            'voiced_fraction':float(np.mean(f0>0)),'pitch_input_modified_after_extraction':False}
        print('GENERATED',frontend,'vocoder + a2a + full a2p',flush=True)
    for name,value in raw.items():
        if name!='original':
            np.save(output/f'{name}_native.npy',value)
            converted=resample_poly(value.astype(np.float64),2,1)
            if len(converted)<len(original):
                raise ValueError('Output is short; refusing padding')
            raw[name]=converted[:len(original)]
    loud={k:qa.measure_loudness(v,rate) for k,v in raw.items()}
    target=batch.common_target(list(loud.values()))
    paths={}
    for name,value in raw.items():
        path=output/f'zuixuan_normal__{name}.wav'
        qa.export_wav(value,rate,target-loud[name]['integrated_lufs'],path)
        paths[name]=path
    ref,_=qa.read_pcm16(paths['original']);loud_ref=qa.measure_loudness(ref,rate)
    report={'source':str(source),'source_sha256':sha(source),'model_seed':0,'vocoder_seed':0,
        'method':'Full official model and vocoder; existing pitch frontends compared without fusion',
        'changed_variable':'whole source-pitch frontend Praat AC vs librosa pYIN; both time-aligned',
        'mel_profile':{k:engine.hp[k] for k in ['audio_sample_rate','fft_size','hop_size','win_size','fmin','fmax']},
        'checkpoint_sha256':sha(n.VAE_DIR/'model_ckpt_steps_200000.ckpt'),
        'vocoder_checkpoint_sha256':sha(n.VOC_DIR/'model_ckpt_steps_1170000.ckpt'),
        'common_target_lufs':target,'inputs':input_records,'outputs':{},'listening_status':'pending_jade',
        'no_residual_no_blending_no_note_correction':True}
    for name,path in paths.items():
        actual,sr=qa.read_pcm16(path)
        with wave.open(str(path),'rb') as stream:
            assert (stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getnframes())==(44100,1,2,len(original))
            pcm=stream.readframes(stream.getnframes())
        assert qa.run_ffmpeg(['-v','error','-nostdin','-xerror','-i',str(path),'-f','s16le','pipe:1']).stdout==pcm
        measured=qa.measure_loudness(actual,sr)
        entry={'file':str(path.relative_to(HERE)),'sha256':sha(path),'samples':qa.sample_stats(actual,sr),
               'loudness':measured,'exact_external_contract':True,'full_ffmpeg_decode':True}
        if name!='original':
            checks=batch.check_output(SimpleNamespace(hp=engine.hp),ref,rate,loud_ref,
                                      resample_poly(actual,1,2),22050)
            local=local_diagnostics(ref,actual)
            if measured['true_peak_dbtp']>-3:
                checks['errors'].append('Final true peak above -3dBTP')
            if abs(measured['integrated_lufs']-loud_ref['integrated_lufs'])>.2:
                checks['errors'].append('Final loudness mismatch above .2LU')
            checks['technical_pass']=not checks['errors']
            entry.update(checks=checks,local_diagnostics=local)
            print('CHECKED',name,json.dumps({'pass':checks['technical_pass'],'errors':checks['errors'],
                'gain':checks['active_gain'],'pitch':checks['pitch_vs_original'],
                'local_drop_frames':len(local['dropout_frames'])},ensure_ascii=False),flush=True)
        report['outputs'][name]=entry
    report['previous_files_verified']=len(preserved)
    report['previous_files_unchanged']=all(sha(ROOT/p)==digest for p,digest in preserved.items())
    assert report['previous_files_unchanged']
    report['full_candidate_technical_pass']=report['outputs']['pyin_a2p']['checks']['technical_pass']
    report['delivery_status']='eligible_pending_listening' if report['full_candidate_technical_pass'] else 'technical_blocked_internal_only'
    (HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRIAL COMPLETE',report['delivery_status'],flush=True)


if __name__=='__main__':
    main()
