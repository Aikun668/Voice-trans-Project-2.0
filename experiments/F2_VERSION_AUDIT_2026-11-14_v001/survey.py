"""Read project-owned assets across existing checkouts; never mutate surveyed assets."""
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
ROOTS=[HERE.parents[1],Path(r'C:\Users\杨玉璋\.codex\worktrees\f2-collaboration-handoff\Voice-trans-Project-2.0'),Path(r'C:\Users\杨玉璋\.codex\worktrees\f2-initial-plan\Voice-trans-Project-2.0')]
LABELS=['main','handoff','initial']
EXTS={'.md','.py','.html','.js','.json','.csv','.cmd','.ps1','.yaml','.yml','.txt'}
PRUNE={'__pycache__','node_modules','vendor','.git','.pytest_cache','.ruff_cache','numba_cache','.numba_cache','audio','wav','raw','maps','sequences','features','checkpoints','weights','weights_dl','site-packages'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csv_write(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def walk(root):
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in PRUNE and not d.startswith(('.venv','venv'))]
        for name in files:
            p=Path(base)/name
            if p.suffix.lower() in EXTS:yield p

def category(name):
    if name=='F2_REFERENCE_FREE_TRIAL_2026-11-14_v001':return 'retain_latest_failed_trial','最新无参考试训，效果失败；保留证据，不设为正式模型'
    if name=='F2_POPBUTFY_CLEAN_2026-11-13_v001':return 'retain_live_input','最新试训 audit_data.py 仍读取其候选和映射'
    if name in {'F2_CONFIRMED_RETRAIN_2026-11-09_v001','F2_MANUAL_MARKS_REVIEW_2026-11-09_v002'}:return 'retain_human_approval','17 对确认标注、原话及编辑页；不可用旧候选替代'
    if name in {'F1_demo_chain_2026-11-02_v001','F1_demo_chain_2026-11-02_v003','F2_MODEL_LISTEN_2026-11-09_v002','F2_XINQIANG_RESYNTH_DIAG_2026-11-08_v001','F2_TASK3_PAIR_AUDIT_2026-11-05_v001'}:return 'retain_dependency_history','历史阶段，但已查到代码复用或上下文依赖；不可整目录搬移'
    if name=='F2_MODEL_SEQUENCE_2026-11-09_v001':return 'retain_reference_baseline_history','旧有参考模型完整序列对照；不能冒充最新无参考模型'
    if name=='F2_REFERENCE_SEQUENCE_2026-11-09_v001':return 'retain_dependency_history','render_v002 仍被 model_sequence.py 读取；直接参考修音是历史对照'
    if name.startswith(('F1','WOMENDE','DEMO','HANDOFF_ALIGN','PINGFAN')):return 'logical_archive_paused_stage','固定 DSP / 旧美化阶段；成果反馈冻结保留，需先核对引用再物理搬移'
    if name.startswith('F2'):return 'logical_archive_old_reference_stage','旧有参考、候选、切口或单字诊断阶段；不作当前执行入口，保留来源与反馈'
    return 'review_scope','其他项目资料，仅盘点不推断可删除'

def main():
    files=[];dirs=[];texts={};references=[];skipped=Counter()
    names={p.name for r in ROOTS for p in (r/'experiments').iterdir() if p.is_dir() and not p.name.startswith('.')}
    pattern=re.compile(r'(?:F[12]|WOMENDE|DEMO|HANDOFF|PINGFAN)[A-Za-z0-9_-]*_v\d{3}')
    for label,root in zip(LABELS,ROOTS):
        for p in (root/'experiments').iterdir():
            if p.is_dir() and not p.name.startswith('.') and p!=HERE:
                kind,reason=category(p.name);dirs.append({'checkout':label,'path':str(p),'name':p.name,'status':kind,'reason':reason})
        targets=[p for p in root.iterdir() if p.is_file() and p.suffix in EXTS]
        for owned in ['docs','experiments','f2','f1_adaptive_v0','neuralsvb_f1','archive']:
            d=root/owned
            if d.exists():targets.extend(walk(d))
        for p in sorted(set(targets)):
            if HERE in p.parents:continue
            rel=p.relative_to(root).as_posix();size=p.stat().st_size
            row={'checkout':label,'path':str(p),'relative_path':rel,'bytes':size,'read_full_text':False,'sha256':''}
            if size<=260000:
                try:
                    s=p.read_text(encoding='utf-8-sig');row.update(read_full_text=True,sha256=sha(p));texts[(label,rel)]=s
                except (UnicodeError,OSError) as e:row['read_error']=str(e)
            else:skipped[label]+=1
            files.append(row)
            if not row['read_full_text']:continue
            found={m[0] for m in re.finditer(pattern,s) if m[0] in names}
            for name in sorted(found):
                if rel.startswith('experiments/'+name+'/'):continue
                references.append({'checkout':label,'from':str(p),'target_experiment':name,
                                   'kind':'code_reference_or_literal' if p.suffix in {'.py','.js','.cmd','.ps1'} else 'document_or_metadata_mention'})
    csv_write(HERE/'files.csv',files);csv_write(HERE/'experiment_status.csv',dirs);csv_write(HERE/'references.csv',references)
    same=defaultdict(list)
    for row in files:
        if row['relative_path'].startswith(('docs/','f2/')) or row['relative_path'] in ['README.md','AGENTS.md']:
            same[row['relative_path']].append(row)
    conflicts=[{'relative_path':k,
                'normalized_text_differs':len({texts.get((r['checkout'],r['relative_path'])) for r in rs})>1,
                'line_ending_or_encoding_only':len({texts.get((r['checkout'],r['relative_path'])) for r in rs})==1,
                'copies':[{'checkout':r['checkout'],'path':r['path'],'sha256':r['sha256']} for r in rs]}
               for k,rs in same.items() if len(rs)>1 and len({r['sha256'] for r in rs})>1]
    dump(HERE/'divergent_copies.json',conflicts)
    # Establish the exact buggy boundary derivation and preserve all manual evidence.
    confirmed=ROOTS[1]/'experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001/confirmed_pairs.json'
    pairs=json.loads(confirmed.read_text(encoding='utf-8'))
    delivery=ROOTS[0]/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/delivery_v001/delivery.json'
    meta=json.loads(delivery.read_text(encoding='utf-8'))
    selected=[p for p in pairs if p['self']['sha256']==meta['source_sha256']]
    offsets=[r-s for p in selected for r,s in zip(p['reference']['seconds'],p['self']['seconds'])]
    start=statistics.median(offsets)
    source_audit={'confirmed_table':str(confirmed),'confirmed_sha256':sha(confirmed),'pair_count':len(pairs),
                  'selected':[{'id':p['id'],'phrase':p['phrase'],'self':p['self']['seconds'],'reference':p['reference']['seconds']} for p in selected],
                  'offset_candidates':offsets,'median_offset':start,'self_duration':meta['duration'],
                  'computed_equal_duration_reference':[start,start+meta['duration']],
                  'actual_export':meta['reference_display']['seconds'],
                  'approved_full_recording_reference_span_found':False,
                  'inference_reference_input':meta['reference_input']}
    source_audit['derivation_matches_export_within_one_sample']=all(abs(a-b)<1/44100 for a,b in zip(source_audit['computed_equal_duration_reference'],source_audit['actual_export']))
    assert source_audit['derivation_matches_export_within_one_sample']
    dump(HERE/'reference_origin.json',source_audit)
    # Same basename is insufficient: inspect actual source hashes across all checkouts.
    audio=[]
    for root,label in zip(ROOTS,LABELS):
        for sub in ['wav','ref_vocals']:
            folder=root/'test_dataset/my_singing/rerecord_2026-11-04_f2'/sub
            if folder.exists():
                for p in folder.glob('*.wav'):audio.append({'checkout':label,'path':str(p),'name':p.name,'sha256':sha(p)})
    csv_write(HERE/'personal_audio_sources.csv',audio)
    summary={'scope_roots':[str(r) for r in ROOTS],'experiment_instances':len(dirs),'unique_experiment_names':len({r['name'] for r in dirs}),
             'project_text_or_metadata_files':len(files),'full_text_files_read':sum(r['read_full_text'] for r in files),
             'large_metadata_files_not_read_in_full':dict(skipped),'status_counts':dict(Counter(r['status'] for r in dirs)),
             'divergent_docs_or_f2_paths':len(conflicts),'normalized_text_divergent_paths':sum(c['normalized_text_differs'] for c in conflicts),'reference_mentions':len(references),
             'limitations':['Static code literals and text references are not a complete runtime dependency graph.',
                            'No listening, historic training, environment cleanup or mass archival performed.',
                            'Old does not mean safe to move; candidate classifications are logical archival only.']}
    dump(HERE/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
