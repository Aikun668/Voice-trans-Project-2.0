"""Known note perturbations supplement weak real-pair labels; report real transfer separately."""
import argparse
from collections import defaultdict,Counter
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch
from trial import context_features,make_practice,new_output,dump,sha,load_model,predict_array


def metrics(model,arrays):
    x,y,w,kind=arrays;p=predict_array(model,x);result={}
    for name,k in [('real',0),('practice',1)]:
        use=kind==k;target=y[use];prediction=p[use];large=np.abs(target)>=60;normal=np.abs(target)<=25
        result[name]={'frames':int(use.sum()),'zero_mae':float(np.mean(np.abs(target))),
              'model_mae':float(np.mean(np.abs(target-prediction))),
              'error_frames':int(large.sum()),'error_zero_mae':float(np.mean(np.abs(target[large]))),
              'error_model_mae':float(np.mean(np.abs(target[large]-prediction[large]))),
              'normal_over_50_rate':float(np.mean(np.abs(prediction[normal])>50)),
              'prediction_abs_median':float(np.median(np.abs(prediction)))}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out);start=time.monotonic()
    samples=torch.load(args.data/'dataset.pt',map_location='cpu')
    manifest=json.loads((args.data/'label_manifest.json').read_text(encoding='utf-8'))
    packs={r['pair_id']:r['pack'] for r in manifest};arrays=defaultdict(lambda:[[],[],[],[]]);counts=defaultdict(Counter)
    def append(split,x,y,mask,weight,kind):
        a=arrays[split];a[0].append(x[mask]);a[1].append(y[mask]);a[2].append(weight[mask])
        a[3].append(np.full(int(mask.sum()),kind,dtype=np.int8))
    for r in samples:
        if r['kind']!='real_pair':continue
        with np.load(io.BytesIO(Path(packs[r['id']]).read_bytes()),allow_pickle=False) as z:
            source=z['source_f0'];professional=z['professional_f0'];valid=z['professional_agreement']
            append(r['split'],context_features(source),r['y'],r['mask'],np.full(len(source),.15),0)
            for variant in range(2):
                seed=int(hashlib.sha256((r['id']+str(variant)).encode()).hexdigest()[:8],16)
                changed,y=make_practice(professional,seed)
                if not np.any(y):
                    if variant==1:continue
                    counts[r['split']].update(normal_sequences=1)
                else:counts[r['split']].update(perturbed_sequences=1,known_error_frames=int((valid&(np.abs(y)>=60)).sum()))
                weight=.4+2.6*(np.abs(y)>=60)
                append(r['split'],context_features(changed),y.astype(np.float32),valid,weight,1)
    sets={s:tuple(np.concatenate(p,axis=0) for p in parts) for s,parts in arrays.items()}
    dump(out/'practice_counts.json',{k:dict(v) for k,v in counts.items()})
    print(json.dumps({'stage':'features','seconds':round(time.monotonic()-start,1),'counts':{k:dict(v) for k,v in counts.items()},
                     'shapes':{k:v[0].shape for k,v in sets.items()}}),flush=True)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=40,max_leaf_nodes=31,min_samples_leaf=150,
               l2_regularization=10,learning_rate=.08,early_stopping=False,warm_start=True,random_state=20261114)
    model._f2_feature_kind='own-context-56';best=float('inf');history=[];best_trees=0;best_probe=None
    with threadpool_limits(limits=2):
        for trees in (40,80,160):
            model.set_params(max_iter=trees);model.fit(sets['train'][0],sets['train'][1],sample_weight=sets['train'][2])
            val=metrics(model,sets['validation']);row={'trees':trees,'validation':val,'seconds':round(time.monotonic()-start,1)}
            history.append(row);dump(out/'history.json',history);print(json.dumps(row),flush=True)
            if val['practice']['model_mae']<best:
                best=val['practice']['model_mae'];best_trees=trees
                best_probe=predict_array(model,sets['validation'][0][:100])
                torch.save({'schema':'own-pitch-context-hgb-v1','model':model,'input_features':56,'reference_input':False,
                            'dataset_sha256':sha(args.data/'dataset.pt'),'sklearn':sklearn.__version__,
                            'training':'known perturbations plus weak real labels'},out/'model.pt')
    restored=load_model(out/'model.pt')
    assert np.array_equal(predict_array(restored,sets['validation'][0][:100]),best_probe)
    report={'architecture':'HistGradientBoostingRegressor, 56 own-pitch contextual features','best_trees':best_trees,
            'checkpoint_sha256':sha(out/'model.pt'),'reference_input':False,'input_features':56,
            'selection':'only synthetic validation-song known-inverse MAE; real validation reported separately, no Chinese tuning',
            'counts':{k:dict(v) for k,v in counts.items()},'validation':metrics(restored,sets['validation']),
            'test':metrics(restored,sets['test']),'checkpoint_reload_exact':True,'sklearn':sklearn.__version__,
            'seconds':round(time.monotonic()-start,1),'listening':'pending_jade',
            'limitation':'Artificial note shifts are not natural human errors. Their success does not prove real pitch correction.'}
    report['real_validation_improved_over_zero']=report['validation']['real']['model_mae']<report['validation']['real']['zero_mae']
    dump(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
