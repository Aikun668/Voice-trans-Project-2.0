"""Export exactly two level-matched audio versions after strict engineering QA."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from adapter import HERE,PROJECT,sha,dump,new_output,_reuse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out);(out/'audio').mkdir()
    record=json.loads((args.inference/'record.json').read_text(encoding='utf-8'))
    with np.load(args.inference/'inference.npz',allow_pickle=False) as z:
        source=z['source'].copy();rendered=z['rendered'].copy();rate=int(z['sample_rate'])
        grid=z['time'].copy();command=z['command'].copy();mask=z['mask'].copy()
    if sha(record['input'])!=record['input_sha256']:raise ValueError('Source changed')
    qa=_reuse(PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/audio_qa.py','frozen_audio_qa')
    arrays={'original':source,'model':rendered}
    raw={name:qa.levels(x,rate) for name,x in arrays.items()}
    common=min([-18.5]+[v['lufs']-1.2-v['true_peak_dbtp'] for v in raw.values()])
    pitch=qa.compare(source,rendered,rate,grid,command)
    if not pitch['pitch_technical_pass']:raise ValueError('Rendered pitch failed engineering gate: '+str(pitch))
    files={};checks={}
    for name,x in arrays.items():
        gain=10**((common-raw[name]['lufs'])/20)
        y=x*gain;path=out/'audio'/f'{name}.wav'
        if not np.isfinite(y).all() or np.max(np.abs(y))>=1:raise ValueError('Nonfinite or clipping export')
        sf.write(path,y,rate,subtype='PCM_24')
        actual,sr=sf.read(io.BytesIO(path.read_bytes()),dtype='float64')
        levels=qa.levels(actual,sr)
        checks[name]={'sha256':sha(path),'samples':len(actual),'sample_rate':sr,
                      'peak_absolute':float(np.max(np.abs(actual))),'clipping_samples':int((np.abs(actual)>=1).sum()),
                      'fixed_gain_db':float(20*np.log10(gain)),**levels}
        if len(actual)!=len(source) or sr!=rate or not np.isfinite(actual).all():raise ValueError('Audio shape/readback failed')
        if abs(levels['lufs']-common)>.1 or levels['true_peak_dbtp']>-1:raise ValueError('Measured loudness/peak failed')
        files[name]=f'audio/{name}.wav'
    levels_difference=abs(checks['original']['lufs']-checks['model']['lufs'])
    if levels_difference>.1:raise ValueError('Versions differ in loudness')
    meta={'duration':len(source)/rate,'files':files,'common_lufs':common,
          'checkpoint_sha256':record['checkpoint_sha256'],'input_sha256':record['input_sha256'],
          'audio_checks':checks,'pitch_checks':pitch,'technical_pass':True,'fine_tuning_steps':0,
          'outside_mask_raw_samples_exact':bool(np.array_equal(source[mask==0],rendered[mask==0])),
          'rendering_has_no_limiter':True,'effectiveness_gate_pass':False,'listening':'pending_jade'}
    if not meta['outside_mask_raw_samples_exact']:raise ValueError('Unselected samples were modified')
    dump(out/'delivery.json',meta)
    page=(HERE/'listening_template.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(page.replace('__METADATA__',json.dumps(meta,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
