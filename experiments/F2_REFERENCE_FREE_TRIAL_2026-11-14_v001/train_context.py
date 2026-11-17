"""CPU baseline: gradient boosting over own pitch and neighboring context, with song holdouts."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import time
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch
from trial import context_features,new_output,dump,sha,load_model,predict_array


def metrics(model,data):
    x,y,w,kind=data;p=predict_array(model,x);real=kind==0
    y,p=y[real],p[real];large=np.abs(y)>=60;normal=np.abs(y)<=25
    identity=predict_array(model,x[kind==1])
    return {'frames':len(y),'nonzero_target_frames':int(large.sum()),'zero_mae':float(np.mean(np.abs(y))),
            'model_mae':float(np.mean(np.abs(y-p))),
            'error_subset_zero_mae':float(np.mean(np.abs(y[large]))),
            'error_subset_model_mae':float(np.mean(np.abs(y[large]-p[large]))),
            'normal_over_50_cents_rate':float(np.mean(np.abs(p[normal])>50)),
            'professional_identity_over_50_rate':float(np.mean(np.abs(identity)>50)),
            'prediction_abs_median':float(np.median(np.abs(p)))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out);start=time.monotonic()
    samples=torch.load(args.data/'dataset.pt',map_location='cpu')
    manifest=json.loads((args.data/'label_manifest.json').read_text(encoding='utf-8'))
    packs={r['pair_id']:r['pack'] for r in manifest}
    arrays=defaultdict(lambda:[[],[],[],[]]);groups=defaultdict(set)
    for i,r in enumerate(samples):
        key=r['id'].removesuffix('-identity')
        with np.load(io.BytesIO(Path(packs[key]).read_bytes()),allow_pickle=False) as z:
            f=z['source_f0'] if r['kind']=='real_pair' else z['target_f0']
            x=context_features(f)
        v=r['mask'];y=r['y'][v];weight=r['weight']*(1+2*(np.abs(y)>=60))
        a=arrays[r['split']];a[0].append(x[v]);a[1].append(y);a[2].append(weight)
        a[3].append(np.full(len(y),0 if r['kind']=='real_pair' else 1,dtype=np.int8))
        groups[r['split']].add(r['split_group'])
    sets={s:tuple(np.concatenate(part,axis=0) for part in a) for s,a in arrays.items()}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    print(json.dumps({'stage':'features','seconds':round(time.monotonic()-start,1),'shapes':{k:v[0].shape for k,v in sets.items()}}),flush=True)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=40,max_leaf_nodes=15,
             min_samples_leaf=300,l2_regularization=10,learning_rate=.08,early_stopping=False,warm_start=True,random_state=20261114)
    model._f2_feature_kind='own-context-56'
    history=[];best=float('inf');best_trees=0
    with threadpool_limits(limits=2):
        for trees in (40,80,120):
            model.set_params(max_iter=trees)
            model.fit(sets['train'][0],sets['train'][1],sample_weight=sets['train'][2])
            validation=metrics(model,sets['validation'])
            row={'trees':trees,'validation':validation,'seconds':round(time.monotonic()-start,1)}
            history.append(row);dump(out/'history.json',history);print(json.dumps(row),flush=True)
            if validation['model_mae']<best:
                best=validation['model_mae'];best_trees=trees
                torch.save({'schema':'own-pitch-context-hgb-v1','model':model,'input_features':56,
                            'reference_input':False,'dataset_sha256':sha(args.data/'dataset.pt'),
                            'sklearn':sklearn.__version__},out/'model.pt')
    restored=load_model(out/'model.pt');x=sets['validation'][0][:100]
    if best_trees==120:assert np.array_equal(predict_array(restored,x),predict_array(model,x))
    report={'architecture':'HistGradientBoostingRegressor on 56 own-pitch context features',
            'best_trees':best_trees,'checkpoint_sha256':sha(out/'model.pt'),'dataset_sha256':sha(args.data/'dataset.pt'),
            'input_features':56,'reference_input':False,'synthetic_training_examples':0,
            'song_counts':{k:len(v) for k,v in groups.items()},'validation':metrics(restored,sets['validation']),
            'test':metrics(restored,sets['test']),'sklearn':sklearn.__version__,'seconds':round(time.monotonic()-start,1),
            'checkpoint_reload_prediction_matches_selected_validation':True,'listening':'pending_jade'}
    report['validation_improved_over_zero']=report['validation']['model_mae']<report['validation']['zero_mae']
    dump(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
