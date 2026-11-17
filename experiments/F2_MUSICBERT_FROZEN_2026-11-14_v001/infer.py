"""Single-input frozen inference; references and existing labels are unavailable."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from adapter import (HERE,PROJECT,ASSETS,sha,dump,new_output,pitch_tracks,segment_notes,
                     FrozenMusicBert,choose_target,make_command,synthesize)


def is_forbidden_read(p,source,out):
    p=Path(p).resolve();text=p.as_posix().lower()
    code=p.suffix.lower() in ('.py','.pyc','.pyd','.dll')
    forbidden=(not code) and (('popbutfy' in text and ASSETS not in p.parents)
              or 'confirmed_pairs' in text or '/ref_vocals/' in text
              or (PROJECT/'test_dataset/reference') in p.parents)
    if p.suffix.lower() in ('.wav','.mp3','.flac','.m4a','.pt','.ckpt','.npz','.npy'):
        forbidden|=p!=source and ASSETS not in p.parents and out not in p.parents
    return forbidden


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    source=args.input.resolve();out=new_output(args.out)
    read_inputs=[];blocked=[]
    def guard(event,data):
        if event!='open' or not isinstance(data[0],(str,bytes,os.PathLike)):return
        mode=data[1]
        if isinstance(mode,str) and any(c in mode for c in 'wax+'):return
        try:p=Path(data[0]).resolve()
        except (TypeError,ValueError,OSError):return
        disallowed=is_forbidden_read(p,source,out)
        if p==source or ASSETS in p.parents:read_inputs.append(str(p))
        if disallowed:
            blocked.append(str(p));raise PermissionError('Reference or external audio/label is blocked during inference: '+str(p))
    sys.addaudithook(guard)
    # Demonstrate that the guard is active before running the model.
    try:(PROJECT/'test_dataset/reference/forbidden-probe.wav').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('Reference guard inactive')
    blob=source.read_bytes();x,rate=sf.read(io.BytesIO(blob),dtype='float64')
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Finite mono WAV required')
    model=FrozenMusicBert()
    time,f0,cc,agreement=pitch_tracks(x,rate)
    notes=segment_notes(time,f0,agreement)
    probabilities=model.probabilities(notes)
    again=model.probabilities(notes)
    if not np.array_equal(probabilities,again):raise ValueError('Frozen inference is nondeterministic')
    rows=[choose_target(note,p) for note,p in zip(notes,probabilities)]
    command=make_command(time,rows);command[f0<=0]=0.
    rendered,mask=synthesize(x,rate,command)
    if rendered.shape!=x.shape or not np.isfinite(rendered).all():raise ValueError('Invalid audio rendering')
    after=model.parameter_digest()
    if after!=model.before:raise ValueError('Model weights changed')
    if not np.array_equal(rendered[mask==0],x[mask==0]):raise ValueError('Unselected samples changed')
    np.savez_compressed(out/'inference.npz',source=x,rendered=rendered,sample_rate=rate,
                        time=time,f0=f0,cc=cc,agreement=agreement,probabilities=probabilities,command=command,mask=mask)
    dump(out/'notes.json',rows)
    dump(out/'model_load.json',{**model.meta,'parameter_digest_after':after,'frozen_after':after==model.before})
    record={'input':str(source),'input_sha256':sha(source),'sample_rate':rate,'samples':len(x),
            'duration':len(x)/rate,'checkpoint_sha256':model.meta['checkpoint_sha256'],
            'rules_sha256':sha(HERE/'rules.md'),'reference_input':False,'guard_probe_blocked':bool(blocked),
            'model_inputs_read':sorted(set(read_inputs)),'notes_total':len(notes),
            'eligible_notes':sum(n['eligible'] for n in notes),'accepted_notes':sum(r['accepted'] for r in rows),
            'changed_frames_over_20_cents':int((np.abs(command)>=20).sum()),
            'command_range_cents':[float(command.min()),float(command.max())],
            'outside_mask_exact':True,'deterministic_repeat_exact':True,'training_steps':0,
            'global_key_shift':False,'listening':'pending_jade'}
    dump(out/'record.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='model_inputs_read'},ensure_ascii=False,indent=2),flush=True)
    print('all_note_suggestions',json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
