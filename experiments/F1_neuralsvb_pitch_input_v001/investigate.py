"""Source feature and harmonic evidence only. Does not synthesize audio."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import librosa
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'neuralsvb_f1')]
import neuralsvb_infer as n
from pitch_frontend import extract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harmonic_window(y,sr,center,low,high):
    segment=y[max(0,int((center-.05)*sr)):int((center+.05)*sr)]
    freq,power=periodogram(segment,fs=sr,window='hann',nfft=65536)
    energy=lambda hz:float(power[abs(freq-hz)<10].sum())
    ac=np.correlate(segment,segment,mode='full')[len(segment)-1:]
    ac/=max(ac[0],1e-20)
    return {'center_s':center,'low_candidate_hz':low,'high_candidate_hz':high,
            'low_to_high_peak_energy_db':float(10*np.log10(max(energy(low),1e-30)/max(energy(high),1e-30))),
            'autocorrelation_at_low_period':float(ac[round(sr/low)]),
            'autocorrelation_at_high_period':float(ac[round(sr/high)]),
            'interpretation_limit':'Descriptive original-signal evidence; missing fundamentals/subharmonics can remain ambiguous.'}


def main():
    evidence_dir=HERE/'evidence'
    evidence_dir.mkdir(exist_ok=False)
    preserve={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments').rglob('*.wav')}
    preserve.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'neuralsvb_f1').glob('*.py')})
    preserve.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'test_dataset/my_singing').glob('*.wav')})
    (HERE/'preservation_manifest.json').write_text(json.dumps(preserve,indent=2),encoding='utf-8')
    n._bootstrap(None)
    from vocoders.pwg import PWG
    all_evidence={}
    for stem,event in [('zuixuan_normal',18.535),('shige_normal',16.655)]:
        wav,mel=PWG.wav2spec(str(ROOT/'test_dataset/my_singing'/f'{stem}.wav'))
        tracks={};metadata={}
        for method in ['praat','pyin']:
            tracks[method],metadata[method]=extract(wav,len(mel),method)
        np.savez(evidence_dir/f'{stem}.npz',wav=wav,mel=mel,**tracks)
        times=np.arange(len(mel))*128/22050
        region=abs(times-event)<.04
        windows=[]
        for center in [event-.08,event+.04]:
            nearby=(abs(times-center)<.025)&(tracks['pyin']>0)
            high=float(np.median(tracks['pyin'][nearby])) if nearby.any() else (316 if stem.startswith('zuixuan') else 234)
            windows.append(harmonic_window(wav,22050,center,high/2,high))
        comparison={'event_s':event,'times':times[region].tolist(),
                    'praat_hz':tracks['praat'][region].tolist(),'pyin_hz':tracks['pyin'][region].tolist(),
                    'harmonic_windows':windows,'metadata':metadata}
        all_evidence[stem]=comparison
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        lo=max(0,event-.45);hi=min(len(wav)/22050,event+.45)
        a=int(lo*22050);b=int(hi*22050)
        power=np.abs(librosa.stft(wav[a:b],n_fft=2048,hop_length=64))
        axes[0].imshow(20*np.log10(np.maximum(power,1e-7)),origin='lower',aspect='auto',
                       extent=[lo,lo+(power.shape[1]-1)*64/22050,0,11025],vmin=-55,vmax=15)
        axes[0].set_ylim(60,1000);axes[0].set_ylabel('Hz');axes[0].set_title(stem+' original waveform spectrum + independent source-pitch tracks')
        for method in tracks:
            values=tracks[method].copy();values[values==0]=np.nan
            axes[1].plot(times,values,label=method,linewidth=1.3)
        axes[1].set_xlim(lo,hi);axes[1].set_ylim(60,600);axes[1].set_xlabel('Original time (s)');axes[1].set_ylabel('Hz');axes[1].legend()
        fig.tight_layout();fig.savefig(evidence_dir/f'{stem}_event.png',dpi=140);plt.close(fig)
        print(stem,json.dumps({'harmonic_windows':windows,'voiced_fraction':{k:metadata[k]['voiced_fraction'] for k in metadata}}),flush=True)
    (evidence_dir/'source_evidence.json').write_text(json.dumps(all_evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Source evidence complete; no generated audio.',flush=True)


if __name__=='__main__':
    main()
