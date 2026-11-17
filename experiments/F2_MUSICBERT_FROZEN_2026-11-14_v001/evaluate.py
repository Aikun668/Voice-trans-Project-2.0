"""Predeclared control evaluation. This script never updates model parameters."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import numpy as np
from adapter import (HERE,PROJECT,sha,dump,new_output,segment_notes,FrozenMusicBert,choose_target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out)
    manifest=PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/data_v001/label_manifest.json'
    entries=json.loads(manifest.read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for entry in entries:
        if entry['split']=='test':groups[entry['split_group']].append(entry)
    rng=np.random.default_rng(20261114)
    group_names=sorted(groups);rng.shuffle(group_names)
    selected=[];sequences=[];excluded=[]
    for group in group_names:
        used=0
        for entry in sorted(groups[group],key=lambda r:r['pair_id']):
            with np.load(entry['pack'],allow_pickle=False) as z:
                notes=segment_notes(z['professional_time'],z['professional_f0'],z['professional_agreement'])
            eligible=[i for i,n in enumerate(notes) if n['eligible']]
            if len(notes)<5 or not eligible:
                excluded.append({'pair_id':entry['pair_id'],'reason':'fewer_than_five_notes_or_no_stable_note'})
                continue
            indices=eligible[:max(0,4-used)]
            if indices:
                sequences.append({'song_group':group,'pair_id':entry['pair_id'],'pack':entry['pack'],
                                  'pack_sha256':sha(entry['pack']),'notes':notes,'indices':indices})
                used+=len(indices)
            if used>=4:break
        if used:selected.append(group)
        if len(selected)>=12:break
    cases=sum(len(s['indices'])*4 for s in sequences)
    if len(selected)<10 or cases<100:raise ValueError(f'Insufficient controls: {len(selected)} songs, {cases} cases')
    dump(out/'control_manifest.json',{'manifest_sha256':sha(manifest),'seed':20261114,'sequences':sequences,
         'excluded':excluded,'label_scope':'Known inverse of injected note shift; extracted source notes are not human musical truth.'})
    model=FrozenMusicBert();normal=[];corrupt=[]
    for seq in sequences:
        notes=seq['notes'];indices=seq['indices']
        base_p=model.probabilities(notes,indices)
        for idx,p in zip(indices,base_p):
            note=notes[idx];label=note['pitch']
            normal_row=choose_target(note,p)
            normal.append({'song_group':seq['song_group'],'pair_id':seq['pair_id'],'index':idx,
                           'source_note':label,'suggestion':normal_row,
                           'modified_over_50':normal_row['accepted'] and abs(normal_row['shift_cents'])>50})
            shuffled=copy.deepcopy(notes)
            other=[j for j in range(len(notes)) if j!=idx]
            order=np.array(other);rng.shuffle(order)
            for dest,source in zip(other,order):
                shuffled[dest]['pitch']=notes[int(source)]['pitch']
                shuffled[dest]['center']=notes[int(source)]['center']
            shuffled_p=model.probabilities(shuffled,[idx])[0]
            for shift in (-2,-1,1,2):
                perturbed={**note,'pitch':label+shift,'center':note['center']+shift}
                row=choose_target(perturbed,p);shuffled_row=choose_target(perturbed,shuffled_p)
                result=row['target'] if row['accepted'] else perturbed['pitch']
                shuffled_result=shuffled_row['target'] if shuffled_row['accepted'] else perturbed['pitch']
                corrupt.append({'song_group':seq['song_group'],'pair_id':seq['pair_id'],'index':idx,
                    'injected_semitones':shift,'known_original_note':label,'input_note':perturbed['pitch'],
                    'result_note':result,'shuffled_result_note':shuffled_result,'recovered_exact':result==label,
                    'shuffled_recovered_exact':shuffled_result==label,'rule_recovered_exact':False,
                    'accepted':row['accepted'],'wrong_accepted':row['accepted'] and result!=label,
                    'raw_local_best_recovered_exact':row['target']==label,'suggestion':row})
        print('checked',seq['song_group'],seq['pair_id'],'cases_so_far',len(corrupt),flush=True)
    recovered=float(np.mean([r['recovered_exact'] for r in corrupt]))
    shuffled_rate=float(np.mean([r['shuffled_recovered_exact'] for r in corrupt]))
    harm=float(np.mean([r['modified_over_50'] for r in normal]))
    summary={'seed':20261114,'songs':len(selected),'professional_contexts':len(sequences),
             'normal_notes':len(normal),'controlled_corruptions':len(corrupt),
             'known_inverse_recovery_rate':recovered,'nearest_semitone_rule_recovery_rate':0.,
             'shuffled_context_recovery_rate':shuffled_rate,'context_lift_percentage_points':100*(recovered-shuffled_rate),
             'normal_control_modification_over_50_fraction':harm,
             'accepted_corruption_fraction':float(np.mean([r['accepted'] for r in corrupt])),
             'wrong_accepted_fraction':float(np.mean([r['wrong_accepted'] for r in corrupt])),
             'raw_local_best_recovery_rate':float(np.mean([r['raw_local_best_recovered_exact'] for r in corrupt])),
             'context_signal_gate':bool(recovered>=.5 and recovered-shuffled_rate>=.10),
             'normal_preservation_gate':bool(harm<=.05),'training_steps':0,
             'parameter_digest_after':model.parameter_digest(),'parameter_digest_before':model.before,
             'label_limit':'Only recovery of known injected shifts; not accuracy on natural human errors.',
             'pretraining_overlap_unknown':True,'thresholds_changed_after_results':False}
    assert summary['parameter_digest_after']==summary['parameter_digest_before']
    dump(out/'normal_controls.json',normal);dump(out/'corrupt_controls.json',corrupt);dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
