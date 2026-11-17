"""Full model-driven SHIGE beautification candidates, no vocoder-only delivery."""
from pathlib import Path
import hashlib,json,os,sys,wave
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'neuralsvb_f1'),str(ROOT/'experiments/F1_neuralsvb_contract_v001')]
import f1_poc as qa
import neuralsvb_infer as n
from feature_contract import aligned_f0
from residual_adapter import render


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE/'diagnostics').exists():
        raise SystemExit('Existing run; refusing overwrite.')
    source=ROOT/'test_dataset/my_singing/shige_normal.wav'
    source_hash=sha(source)
    old={str(p):sha(p) for folder in (ROOT/'experiments').glob('F1_neuralsvb*') for p in folder.rglob('*.wav')}
    original,sr=qa.read_pcm16(source)
    assert sr==44100
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    hp=n._bootstrap(None)
    from vocoders.pwg import PWG
    from modules.voice_conversion.svb_vae import MleSVBVAE
    from utils.pitch_utils import f0_to_coarse
    from resemblyzer import VoiceEncoder
    ckpt=torch.load(n.VAE_DIR/'model_ckpt_steps_200000.ckpt',map_location='cpu')
    sd=ckpt['state_dict']['model']
    model=MleSVBVAE(sd['vc_asr.token_embed.weight'].shape[0])
    model.load_state_dict(sd,strict=True);model.eval()
    wav,mel=PWG.wav2spec(str(source))
    f0,time_info=aligned_f0(wav,len(mel));count=len(mel);pad=(-count)%4
    mt=torch.tensor(np.pad(mel,((0,pad),(0,0)),mode='edge'),dtype=torch.float32)[None]
    pt=torch.tensor(f0_to_coarse(np.pad(f0,(0,pad),mode='edge')),dtype=torch.long)[None]
    spk=VoiceEncoder(device='cpu').embed_utterance(wav)
    st=torch.tensor(spk,dtype=torch.float32)[None];align=torch.arange(len(mt[0]),dtype=torch.long)[None]
    torch.manual_seed(0)
    with torch.no_grad():
        out=model(amateur_mel=mt,prof_mel=mt,amateur_pitch=pt,prof_pitch=pt,
                  amateur_spk_id=st,prof_spk_id=st,a2p_alignment=align,p2a_alignment=None,
                  infer=False,concurrent_ways=['a2a','p2p','a2p'],disable_map=False)
        cond=model.prepare_condition(mt,pt,st)
        g=model.encoded_embed_proj(torch.cat([cond['h_pitch'],cond['h_content'],cond['h_style']],-1)).transpose(1,2)
        disabled=model.vae_model.decoder(out['a2a']['z_q'],cond['tgt_nonpadding'].transpose(1,2),g=g)[0].transpose(0,1).numpy()[:count]
    base=out['a2a']['mel_out'][0].detach().numpy()[:count]
    mapped=out['a2p']['mel_out'][0].detach().numpy()[:count]
    assert np.max(abs(disabled-base))<1e-5, 'Counterfactual decoder conditions do not match.'
    counterfactual,off_meta=render(original,disabled,base,1.)
    identity_error=float(np.max(abs(counterfactual-original)))
    assert identity_error<1e-6
    folder=HERE/'diagnostics';folder.mkdir()
    np.savez(folder/'neural_features.npz',source_mel=mel,reconstruction_mel=base,mapped_mel=mapped,
             disabled_mapping_mel=disabled,f0=f0,latent=out['a2a']['z_q'].detach().numpy())
    raw={'original':original};meta={}
    for name,strength in [('beautify_gentle',.5),('beautify_full',1.)]:
        raw[name],meta[name]=render(original,mapped,base,strength)
        assert meta[name]['model_delta_rms_db']>.01, 'Mapping has no measurable effect.'
        print('RENDERED',name,json.dumps(meta[name]),flush=True)
    loud={k:qa.measure_loudness(v,sr) for k,v in raw.items()}
    target=min(-18.,*[m['integrated_lufs']-3.1-m['true_peak_dbtp'] for m in loud.values()])
    paths={}
    for name,value in raw.items():
        p=folder/f'shige_normal__{name}.wav'
        qa.export_wav(value,sr,target-loud[name]['integrated_lufs'],p);paths[name]=p
    ref,_=qa.read_pcm16(paths['original']);ref_loud=qa.measure_loudness(ref,sr)
    report={'method':'NeuralSVB beautification difference with original-phase synthesis',
            'official_default_waveform_pipeline':False,'source':str(source),'source_sha256':source_hash,
            'target_lufs':target,'f0_alignment':time_info,'counterfactual_identity_max_error':identity_error,
            'checkpoint_sha256':sha(n.VAE_DIR/'model_ckpt_steps_200000.ckpt'),
            'listening_status':'pending_jade','user_feedback_previous_delivery':'SHIGE still has audible jumps; vocoder reconstruction is not a beautification delivery.',
            'outputs':{}}
    for name,p in paths.items():
        actual,rate=qa.read_pcm16(p)
        with wave.open(str(p),'rb') as stream:
            assert (stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getnframes(),stream.getcomptype())==(44100,1,2,len(original),'NONE')
            pcm=stream.readframes(stream.getnframes())
        decoded=qa.run_ffmpeg(['-v','error','-nostdin','-xerror','-i',str(p),'-f','s16le','pipe:1']).stdout
        assert decoded==pcm
        measured=qa.measure_loudness(actual,rate)
        entry={'file':str(p.relative_to(HERE)),'sha256':sha(p),'samples':qa.sample_stats(actual,rate),
               'loudness':measured,'contract_pass':True,'full_decode_pass':True}
        if name!='original':
            c=qa.compare_audio(ref,actual,rate)
            if measured['true_peak_dbtp']>-3: c['errors'].append('True peak exceeds -3dBTP')
            if abs(measured['integrated_lufs']-ref_loud['integrated_lufs'])>.2: c['errors'].append('Loudness differs by >0.2LU')
            c['passed']=not c['errors']
            entry.update({'checks':c,'adaptation':meta[name]})
            print('CHECKED',name,json.dumps(c,ensure_ascii=False),flush=True)
        report['outputs'][name]=entry
    assert sha(source)==source_hash and all(sha(p)==h for p,h in old.items())
    report['source_and_previous_wavs_unchanged']=True
    report['previous_wavs_verified']=len(old)
    (HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINISHED',HERE/'report.json',flush=True)


if __name__=='__main__':
    main()
