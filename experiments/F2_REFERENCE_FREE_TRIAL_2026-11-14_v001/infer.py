"""Fresh-process inference accepts one source recording and one trained checkpoint only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from trial import HERE,PROJECT,HOP,sha,dump,new_output,own_features,model_features,load_model,predict_array,render_command,synthesize,PitchModel
from audit_data import pitch_tracks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();source=args.input.resolve();checkpoint=args.checkpoint.resolve();out=new_output(args.out)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    forbidden=[Path(r'C:\Users\杨玉璋\Downloads\PopBuTFy').resolve(),
               PROJECT/'experiments/F2_POPBUTFY_CLEAN_2026-11-13_v001']
    accessed=[]
    def file_guard(event,data):
        if event!='open' or not isinstance(data[0],(str,bytes,os.PathLike)):return
        mode=data[1]
        if isinstance(mode,str) and any(c in mode for c in 'wax+'):return
        try:p=Path(data[0]).resolve()
        except (ValueError,OSError,TypeError):return
        text=p.as_posix().lower()
        if p!=source and p!=checkpoint:
            if any(root==p or root in p.parents for root in forbidden) or 'ref_vocals' in text or '/reference/' in text:
                raise PermissionError('Reference/training source unavailable during inference: '+str(p))
            if HERE in p.parents and p.suffix.lower() in ('.npz','.csv','.json','.pt'):
                raise PermissionError('Training/label artifact unavailable during inference: '+str(p))
        if p==source or p==checkpoint:accessed.append(str(p))
    sys.addaudithook(file_guard)
    model=load_model(checkpoint)
    x,rate=sf.read(source,dtype='float64')
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Need finite mono source')
    g=math.gcd(rate,22050);analysis=resample_poly(x,22050//g,rate//g)
    time,f0,cc,agreement=pitch_tracks(analysis,22050)
    features=model_features(model,f0);prediction=predict_array(model,features)
    initial_prediction=predict_array(PitchModel().eval(),own_features(f0))
    assert np.array_equal(initial_prediction,np.zeros(len(time)))
    command=render_command(f0,prediction);command[~agreement]=0.
    zero_audio,zero_mask=synthesize(x,rate,np.zeros(len(time)))
    assert np.array_equal(zero_audio,x) and not zero_mask.any()
    rendered,mask=synthesize(x,rate,command)
    if np.sum(np.abs(command)>=20)<5:raise ValueError('Model produced too little correction for a meaningful audio trial')
    assert rendered.shape==x.shape and np.isfinite(rendered).all()
    np.savez_compressed(out/'inference.npz',rendered=rendered,source=x,rate=rate,time=time,source_f0=f0,
                        prediction=prediction,command=command,mask=mask,source_cc=cc)
    record={'input':str(source),'input_sha256':sha(source),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'sample_rate':rate,'samples':len(x),'duration':len(x)/rate,'input_features':features.shape[1],'reference_input':False,
            'reference_and_training_data_access_blocked':True,'source_and_checkpoint_reads':sorted(set(accessed)),
            'zero_model_prediction_exact':True,'zero_command_audio_exact':True,
            'frame_count':len(time),'changed_frames_over_20_cents':int((np.abs(command)>=20).sum()),
            'changed_samples':int((mask>0).sum()),'prediction_abs_percentiles':np.percentile(np.abs(prediction),[50,90,99]).tolist(),
            'command_range_cents':[float(command.min()),float(command.max())],
            'outside_generated_mask_exact':bool(np.array_equal(x[mask==0],rendered[mask==0])),
            'source_key_shift_switch':False,'listening':'pending_jade'}
    dump(out/'record.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
