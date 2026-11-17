"""Fit a small standard bidirectional GRU using own pitch context, never reference inputs."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from trial import PitchModel,new_output,dump,sha,load_model,predict_array


def chunks(samples):
    result=[]
    for r in samples:
        n=len(r['y']);starts=list(range(0,max(n-256,0)+1,192))
        if starts[-1]+256<n:starts.append(max(n-256,0))
        for start in sorted(set(starts)):
            stop=min(start+256,n)
            if r['mask'][start:stop].sum()>=10:
                result.append({k:r[k][start:stop] for k in ('x','y','mask')}|{'weight':r['weight']})
    return result


def batch(rows):
    lengths=torch.tensor([len(r['y']) for r in rows],dtype=torch.long);n=int(lengths.max())
    x=torch.zeros(len(rows),n,19);y=torch.zeros(len(rows),n);m=torch.zeros(len(rows),n)
    for i,r in enumerate(rows):
        k=len(r['y']);x[i,:k]=torch.from_numpy(r['x']);y[i,:k]=torch.from_numpy(r['y'])
        m[i,:k]=torch.from_numpy(r['mask'].astype(np.float32))*r['weight']
    return x,y,m,lengths


@torch.inference_mode()
def evaluate(model,samples):
    model.eval();truth=[];pred=[];identity=[];rule=[]
    for r in samples:
        p=predict_array(model,r['x']);m=r['mask']
        if r['kind']=='real_pair':
            truth.extend(r['y'][m]);pred.extend(p[m]);rule.extend((-100*r['x'][:,2])[m])
        else:identity.extend(p[m])
    y,p,b=np.array(truth),np.array(pred),np.array(rule);large=np.abs(y)>=60;normal=np.abs(y)<=25
    return {'frames':len(y),'nonzero_target_frames':int(large.sum()),'zero_mae':float(np.mean(np.abs(y))),
            'rule_mae':float(np.mean(np.abs(y-b))),'model_mae':float(np.mean(np.abs(y-p))),
            'error_subset_zero_mae':float(np.mean(np.abs(y[large]))),'error_subset_model_mae':float(np.mean(np.abs(y[large]-p[large]))),
            'normal_over_50_cents_rate':float(np.mean(np.abs(p[normal])>50)) if normal.any() else None,
            'professional_identity_over_50_rate':float(np.mean(np.abs(identity)>50)),
            'prediction_abs_median':float(np.median(np.abs(p)))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=40);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261114);rng=np.random.default_rng(20261114)
    samples=torch.load(args.data,map_location='cpu')
    sets={s:[r for r in samples if r['split']==s] for s in ('train','validation','test')}
    groups={s:{r['split_group'] for r in rr} for s,rr in sets.items()}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    out=new_output(args.out);model=PitchModel();initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.0001)
    criterion=nn.SmoothL1Loss(beta=.25,reduction='none');training=chunks(sets['train'])
    start=time.monotonic();best=float('inf');best_epoch=0;history=[];best_state=None
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0.;order=rng.permutation(len(training))
        for begin in range(0,len(order),32):
            rows=[training[i] for i in order[begin:begin+32]];x,y,mask,lengths=batch(rows)
            optimizer.zero_grad(set_to_none=True);prediction=model(x,lengths)
            weights=mask*(1+2*(y.abs()>=60).float())
            loss=(criterion(prediction/100,y/100)*weights).sum()/weights.sum().clamp_min(1)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss)
        metrics=evaluate(model,sets['validation'])
        row={'epoch':epoch,'loss':loss_sum/max(1,(len(training)+31)//32),'validation':metrics,'seconds':round(time.monotonic()-start,1)}
        history.append(row);dump(out/'history.json',history);print(json.dumps(row),flush=True)
        # Selection uses only validation songs. Chinese recordings are never queried here.
        if metrics['model_mae']<best:
            best=metrics['model_mae'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        elif epoch-best_epoch>=7 and epoch>=12:break
    model.load_state_dict(best_state)
    changed=any(not torch.equal(initial[k],best_state[k]) for k in initial)
    assert changed,'Training failed to change weights'
    payload={'schema':'own-pitch-context-v1','state':best_state,'best_epoch':best_epoch,
             'input_features':19,'reference_input':False,'dataset_sha256':sha(args.data),
             'architecture':'PyTorch GRU(19,48,bidirectional) + LayerNorm/MLP; continuous signed cents'}
    torch.save(payload,out/'model.pt');restored=load_model(out/'model.pt')
    x=sets['validation'][0]['x']
    assert np.array_equal(predict_array(restored,x),predict_array(model,x))
    report={'best_epoch':best_epoch,'trained_epochs':len(history),'parameters':sum(p.numel() for p in model.parameters()),
            'weights_changed':changed,'checkpoint_reload_exact':True,'checkpoint_sha256':sha(out/'model.pt'),
            'dataset_sha256':sha(args.data),'sequence_counts':{k:len(v) for k,v in sets.items()},
            'song_counts':{k:len(v) for k,v in groups.items()},
            'validation':evaluate(restored,sets['validation']),'test':evaluate(restored,sets['test']),
            'reference_input':False,'chinese_training_examples':0,'torch':torch.__version__,
            'seconds':round(time.monotonic()-start,1),'listening':'pending_jade'}
    report['validation_improved_over_zero']=report['validation']['model_mae']<report['validation']['zero_mae']
    dump(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
