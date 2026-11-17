"""Post-inference review only; cannot feed references back into model decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from adapter import HERE,sha,dump,new_output

CONFIRMED=Path(r'C:\Users\杨玉璋\.codex\worktrees\f2-collaboration-handoff\Voice-trans-Project-2.0\experiments\F2_CONFIRMED_RETRAIN_2026-11-09_v001\confirmed_pairs.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inference',type=Path,required=True)
    ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=new_output(args.out)
    record=json.loads((args.inference/'record.json').read_text(encoding='utf-8'))
    notes=json.loads((args.inference/'notes.json').read_text(encoding='utf-8'))
    controls=json.loads((args.evaluation/'summary.json').read_text(encoding='utf-8'))
    # This is the first stage allowed to read human-confirmed content boundaries.
    pairs=json.loads(CONFIRMED.read_text(encoding='utf-8'))
    matching=[p for p in pairs if p['self']['sha256']==record['input_sha256']]
    with np.load(args.inference/'inference.npz',allow_pickle=False) as z:
        grid=z['time'];command=z['command'];active=np.abs(command)>=50
    known=np.zeros(len(grid),dtype=bool);coverage=[]
    for pair in matching:
        start,end=pair['self']['seconds'];m=(grid>=start)&(grid<=end);known|=m
        coverage.append({'phrase':pair['phrase'],'confirmed_content_seconds':[start,end],
                         'changed_frames_over_50':int((m&active).sum()),
                         'changed_seconds_over_50':float((m&active).sum()*.02),
                         'direction_verified':False,
                         'target_note_status':'Content cut confirmed, no human-confirmed per-note target in this table.'})
    changes=[]
    for row in notes:
        if not row['accepted']:continue
        changes.append({k:row[k] for k in ('index','start','end','pitch','target','shift_cents','odds','local_target_probability')})
        changes[-1]['overlapping_confirmed_phrases']=[p['phrase'] for p in matching if min(row['end'],p['self']['seconds'][1])>max(row['start'],p['self']['seconds'][0])]
    summary={'input_sha256':record['input_sha256'],'confirmed_table_sha256':sha(CONFIRMED),
             'post_inference_only':True,'raw_notes':len(notes),'safety_eligible_notes':record['eligible_notes'],
             'accepted_notes':len(changes),'all_accepted_changes':changes,'confirmed_problem_coverage':coverage,
             'modified_seconds_over_50_outside_confirmed_cuts':float((active&~known).sum()*.02),
             'outside_cuts_are_not_automatically_normal':True,
             'personal_direction_gate_demonstrated':False,
             'personal_direction_gate_reason':'Cut confirmation does not establish the correct note or correction direction.',
             'controlled_context_gate':controls['context_signal_gate'],
             'controlled_normal_preservation_gate':controls['normal_preservation_gate'],
             'immediate_finetuning_recommended':False,
             'recommendation':'Do not start fine-tuning now: the predeclared recovery and normal-preservation gates failed. Zero-shot failure does not prove fine-tuning cannot work.',
             'listening':'pending_jade','selection_thresholds_changed':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
