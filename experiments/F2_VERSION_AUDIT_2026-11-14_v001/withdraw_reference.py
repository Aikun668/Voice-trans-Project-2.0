"""Withdraw only the confirmed-bad generated reference; retain source evidence."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
TRIAL=PROJECT/'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001'
DELIVERY=TRIAL/'delivery_v001'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    record=HERE/'withdrawal.json'
    if record.exists():raise FileExistsError(record)
    meta=json.loads((DELIVERY/'delivery.json').read_text(encoding='utf-8'))
    html_path=DELIVERY/'index.html';page=html_path.read_text(encoding='utf-8')
    match=re.search(r'const data=(.*?);const player=',page)
    if not match:raise ValueError('Unexpected page schema; abort rather than patch another page')
    view=json.loads(match[1])
    assert view['checkpoint_sha256']==meta['checkpoint_sha256']=='d63eea5b0dea900aa4fb100fbb0159c0e4cba3177821d24ac6e20ceebdca955d'
    wrong=(DELIVERY/view['files']['reference']).resolve()
    assert wrong.parent==DELIVERY.resolve()/'audio'
    assert sha(wrong)==view['audio_checks']['reference']['sha256']=='ee6a1e62936b38effa5f673901f37703862167cbfce39b6c46e519b9d68d6d81'
    snapshots=HERE/'before';snapshots.mkdir(exist_ok=False)
    paths=[html_path,DELIVERY/'delivery.json',TRIAL/'deliver.py',TRIAL/'REPORT.md',TRIAL/'artifact_hashes.json',TRIAL/'verification.json']
    before={str(p):sha(p) for p in paths}
    assert before[str(TRIAL/'deliver.py')]=='598317cf5638eb649ef8fda352fde9b2c836a3fb6ee745b4c0b1867cf3560d1c'
    for p in paths:(snapshots/(p.parent.name+'__'+p.name)).write_bytes(p.read_bytes())
    protected={str(DELIVERY/'audio'/n):sha(DELIVERY/'audio'/n) for n in ['original.wav','model.wav']}
    source=Path(meta['reference_display']['source']);protected[str(source)]=sha(source)
    confirmed=Path(r'C:\Users\杨玉璋\.codex\worktrees\f2-collaboration-handoff\Voice-trans-Project-2.0\experiments\F2_CONFIRMED_RETRAIN_2026-11-09_v001\confirmed_pairs.json')
    protected[str(confirmed)]=sha(confirmed)
    for key in ['files','audio_checks']:view[key].pop('reference',None)
    view.pop('reference_display',None)
    view['reference_withdrawn']={'date':'2026-11-14','reason':'用户确认唱词范围不一致；等时长推算不是已确认的完整切口。'}
    page=page[:match.start(1)]+json.dumps(view,ensure_ascii=False)+page[match.end(1):]
    page,count=re.subn(r'<button data-version="reference"[^>]*>.*?</button>','',page)
    assert count==1
    page=page.replace('三个版本已用固定增益调整到接近相同响度。模型保留你的音区；原唱只作旋律参照。',
                      '两版已调整到接近相同响度。原唱片段内容不一致，已撤下。')
    assert 'data-version="reference"' not in page and 'audio/reference.wav' not in page
    quarantined=(DELIVERY/'_superseded_2026-11-14'/'reference_content_mismatch.wav').resolve()
    assert DELIVERY.resolve() in quarantined.parents and not quarantined.exists()
    quarantined.parent.mkdir(exist_ok=False)
    wrong.rename(quarantined)
    html_path.write_text(page,encoding='utf-8')
    # Retire the unsafe builder; do not regenerate, stretch audio, or change model inference.
    builder=TRIAL/'deliver.py';code=builder.read_text(encoding='utf-8')
    first_end=code.find('\n')
    guard='\n# Withdrawn 2026-11-14: median-offset plus equal-duration cropping is not a confirmed content span.\nif __name__ == "__main__":\n    raise SystemExit("原唱等时长导出入口已停用：请使用已确认的双方切口，不得推算整段。见 docs/版本与引用核查_2026-11-14.md")\n'
    builder.write_text(code[:first_end+1]+guard+code[first_end+1:],encoding='utf-8')
    report=TRIAL/'REPORT.md';old=report.read_text(encoding='utf-8')
    notice='\n> **2026-11-14 引用更正**：用户确认③原唱唱词范围不一致，已从在线页面撤下；原错误生成音频隔离保留，原唱源文件及模型版未改。原来的三版检查与本页以下记录均为撤下前快照，不能当作当前三版可用。等时长导出脚本已停用。原因及版本清单见 [核查记录](../../docs/版本与引用核查_2026-11-14.md)。\n'
    report.write_text(old.split('\n',1)[0]+'\n'+notice+old.split('\n',1)[1],encoding='utf-8')
    for p,h in protected.items():assert sha(Path(p))==h
    result={'date':'2026-11-14','url':'http://127.0.0.1:62512/index.html',
            'reason':'Approved local phrase ranges were extrapolated by a median offset into an unapproved full-duration reference crop.',
            'before':before,'after':{str(p):sha(p) for p in paths},'protected_hashes_unchanged':protected,
            'quarantined_audio':str(quarantined),'quarantined_sha256':sha(quarantined),
            'original_reference_url_removed':not wrong.exists(),'remaining_versions':['original','model'],
            'historical_manifest_and_verification':'Preserved as pre-withdrawal snapshots; superseded for changed page/builder/report.'}
    record.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'withdrawn':str(wrong),'quarantined':str(quarantined),'protected_files':len(protected)},ensure_ascii=False))

if __name__=='__main__':main()
