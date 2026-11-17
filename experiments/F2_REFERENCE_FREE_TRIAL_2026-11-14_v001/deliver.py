"""Prepare three honest listening versions after blind model inference has finished."""

# Withdrawn 2026-11-14: median-offset plus equal-duration cropping is not a confirmed content span.
if __name__ == "__main__":
    raise SystemExit("原唱等时长导出入口已停用：请使用已确认的双方切口，不得推算整段。见 docs/版本与引用核查_2026-11-14.md")
import argparse
import html
import io
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from trial import HERE,sha,new_output,dump
from audio_qa import levels,compare

CONFIRMED=Path(r'C:\Users\杨玉璋\.codex\worktrees\f2-collaboration-handoff\Voice-trans-Project-2.0\experiments\F2_CONFIRMED_RETRAIN_2026-11-09_v001\confirmed_pairs.json')


def make_page(out,metadata):
    js=json.dumps(metadata,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,">
<title>心墙 · 首次无参考试训试听</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f1;color:#21322e;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:850px;margin:45px auto;padding:28px;background:white;border-radius:16px}h1{font-size:27px;line-height:1.4;margin:0 0 12px}p{margin:8px 0}.note{color:#52665c;font-size:14px}.versions{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0 14px}button,a.download{font:inherit;border:1px solid #cbd6cf;background:#fff;border-radius:8px;padding:10px 16px;color:inherit;cursor:pointer}button[aria-pressed=true]{background:#244f40;color:white;border-color:#244f40}audio{width:100%;margin:12px 0}.tools{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 20px}.download{text-decoration:none}fieldset{border:1px solid #dae1db;border-radius:8px;padding:14px;margin:16px 0}legend{font-weight:600}label{display:inline-block;margin:4px 18px 4px 0}textarea{width:100%;min-height:85px;padding:10px;font:inherit;border:1px solid #cbd6cf;border-radius:8px}details{font-size:13px;margin-top:25px}#status{min-height:28px;color:#52665c}.bottom{border-top:1px solid #eee;margin-top:24px;padding-top:12px;font-size:13px;color:#68756e}@media(max-width:600px){main{margin:12px;padding:20px}h1{font-size:23px}}
</style><main><h1>《心墙》：先听这一条</h1>
<p style="padding:12px 15px;background:#fff3d9;border-radius:8px"><strong>未通过自动效果验证。</strong> 已知正常音存在误改。本页保留模型的真实结果，用于诊断试听。</p>
<p>模型只读了你的完整清唱，原唱没有参与这次处理。</p>
<p class="note">戴耳机、固定音量。三个版本已用固定增益调整到接近相同响度。模型保留你的音区；原唱只作旋律参照。</p>
<div class="versions"><button data-version="original" aria-pressed="true">① 你的原音</button><button data-version="model" aria-pressed="false">② 模型试验版</button><button data-version="reference" aria-pressed="false">③ 原唱对应片段</button></div>
<p id="description"></p><audio id="player" controls preload="metadata"></audio><div class="tools"><button id="start">从头听</button><button id="changed">从改动处前面听</button><a class="download" id="download" download>下载当前音频</a></div><p id="status" role="status"></p>
<fieldset><legend>模型版唱得准不准？</legend><label><input type="radio" name="accuracy" value="更准">更准</label><label><input type="radio" name="accuracy" value="差不多">差不多</label><label><input type="radio" name="accuracy" value="更不准">更不准</label><label><input type="radio" name="accuracy" value="拿不准">拿不准</label></fieldset>
<fieldset><legend>声音有没有新问题？</legend><label><input type="radio" name="naturalness" value="没听到新问题">没听到新问题</label><label><input type="radio" name="naturalness" value="有电音或跳变">有电音或跳变</label><label><input type="radio" name="naturalness" value="不像本人">不像本人</label><label><input type="radio" name="naturalness" value="拿不准">拿不准</label></fieldset>
<fieldset><legend>你会选哪个？</legend><label><input type="radio" name="preference" value="原音">原音</label><label><input type="radio" name="preference" value="模型版">模型版</label><label><input type="radio" name="preference" value="都不行">都不行</label><label><input type="radio" name="preference" value="拿不准">拿不准</label></fieldset>
<label for="notes">哪一句、哪个字，或哪一秒有问题</label><textarea id="notes"></textarea><div class="tools"><button id="export">复制反馈</button></div><textarea id="feedback-text" aria-label="可复制的反馈" readonly hidden></textarea>
<details><summary>查看这轮的检查记录</summary><pre id="qa" style="white-space:pre-wrap"></pre></details>
<p class="bottom">Agent 不能听音。程序检查通过不代表音准和自然度已验收。请以你的试听为准。</p></main><script>
const data=METADATA;const player=document.getElementById('player');let current='original';
const descriptions={original:'本人完整原音，只调整响度。',model:'训练后保存并重新载入模型，由清唱自身的前后文预测修正。',reference:'与这条清唱对应的原唱段落，未做节奏拉伸。'};
function save(){const f={};for(const name of ['accuracy','naturalness','preference'])f[name]=document.querySelector('input[name="'+name+'"]:checked')?.value||'';f.notes=document.getElementById('notes').value;localStorage.setItem('f2-rf-trial-v001',JSON.stringify(f));}
function choose(version){const previous=current;const position=player.currentTime||0;const playing=!player.paused;player.pause();current=version;player.src=data.files[version];document.querySelectorAll('[data-version]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.version===version)));document.getElementById('description').textContent=descriptions[version];document.getElementById('download').href=data.files[version];player.addEventListener('loadedmetadata',()=>{player.currentTime=(version==='reference'||previous==='reference')?0:(position>=data.duration-.15?data.first_change_preview:Math.min(position,Math.max(0,player.duration-.05)));if(playing)player.play().catch(()=>{});},{once:true});document.getElementById('status').textContent='';}
document.querySelectorAll('[data-version]').forEach(b=>b.onclick=()=>choose(b.dataset.version));document.getElementById('start').onclick=()=>{player.currentTime=0;player.play().catch(()=>{});};document.getElementById('changed').onclick=()=>{if(current==='reference'){document.getElementById('status').textContent='先切回原音或模型版，再定位改动处。';return;}player.currentTime=data.first_change_preview;player.play().catch(()=>{});};
document.querySelectorAll('input,textarea').forEach(e=>e.addEventListener('change',save));document.getElementById('notes').addEventListener('input',save);
try{const old=JSON.parse(localStorage.getItem('f2-rf-trial-v001')||'{}');for(const name of ['accuracy','naturalness','preference']){const input=Array.from(document.querySelectorAll('input[name="'+name+'"]')).find(e=>e.value===old[name]);if(input)input.checked=true;}document.getElementById('notes').value=old.notes||'';}catch(e){}
document.getElementById('export').onclick=async()=>{save();const value=name=>document.querySelector('input[name="'+name+'"]:checked')?.value||'未答';const text=['【心墙 无参考首次试训反馈】','模型权重：'+data.checkpoint_sha256,'音准：'+value('accuracy'),'新问题：'+value('naturalness'),'选择：'+value('preference'),'补充：'+(document.getElementById('notes').value||'未答')].join(String.fromCharCode(10));const field=document.getElementById('feedback-text');field.value=text;try{if(!navigator.clipboard?.writeText)throw new Error('Clipboard unavailable');await navigator.clipboard.writeText(text);field.hidden=true;document.getElementById('status').textContent='已复制，直接粘贴到聊天里即可。';}catch(error){field.hidden=false;field.focus();field.select();document.getElementById('status').textContent='自动复制未获允许，文本已选中，按 Ctrl+C 即可。';}};
document.getElementById('qa').textContent='本人录音未用于训练。\\n模型输入中不含原唱信息。\\n权重 SHA256：'+data.checkpoint_sha256+'\\n导出响度：'+data.common_lufs+' LUFS\\n工程检查：'+(data.technical_pass?'通过':'未通过')+'\\n听感：Jade 已反馈，无有用改善';choose('original');
</script></html>'''.replace('METADATA',js)
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    record=json.loads((args.inference/'record.json').read_text(encoding='utf-8'))
    with np.load(io.BytesIO((args.inference/'inference.npz').read_bytes()),allow_pickle=False) as z:
        source=z['source'].copy();rendered=z['rendered'].copy();rate=int(z['rate']);grid=z['time'].copy()
        command=z['command'].copy();mask=z['mask'].copy()
    assert sha(record['input'])==record['input_sha256']
    confirmed=json.loads(CONFIRMED.read_text(encoding='utf-8'))
    selected=[p for p in confirmed if p['self']['sha256']==record['input_sha256']]
    if not selected:raise ValueError('No confirmed reference correspondence for this recording')
    ref=Path(selected[0]['reference']['path'])
    if sha(ref)!=selected[0]['reference']['sha256']:raise ValueError('Reference source hash differs from confirmed table')
    offsets=[]
    for p in selected:
        offsets.extend([r-s for r,s in zip(p['reference']['seconds'],p['self']['seconds'])])
    start=max(0.,float(np.median(offsets)));duration=len(source)/rate
    ref_audio,ref_rate=sf.read(ref,dtype='float64');begin=round(start*ref_rate);end=begin+round(duration*ref_rate)
    reference=ref_audio[begin:end]
    if reference.ndim==2:reference=reference.mean(axis=1)
    originals={'original':(source,rate),'model':(rendered,rate),'reference':(reference,ref_rate)}
    raw_levels={name:levels(x,sr) for name,(x,sr) in originals.items()}
    common=min([-18.5]+[v['lufs']-1.2-v['true_peak_dbtp'] for v in raw_levels.values()])
    common=math.floor(common*10)/10
    out=new_output(args.out);(out/'audio').mkdir();files={};qa={}
    normalized={}
    for name,(x,sr) in originals.items():
        gain_db=common-raw_levels[name]['lufs'];gain=10**(gain_db/20)
        y=x*gain
        path=out/'audio'/(name+'.wav');sf.write(path,y,sr,subtype='PCM_16')
        check,check_rate=sf.read(path,dtype='float64');data,_=sf.read(path,dtype='int16')
        measured=levels(check,check_rate)
        stats={'samples':len(check),'sample_rate':check_rate,'finite':bool(np.isfinite(check).all()),
               'clipped_samples':int(np.sum(np.abs(data.astype(np.int32))>=32767)),
               'gain_db':gain_db,'sha256':sha(path),**measured}
        stats['technical_pass']=bool(len(check)==len(x) and check_rate==sr and stats['finite'] and stats['clipped_samples']==0
                                     and measured['true_peak_dbtp']<=-1 and abs(measured['lufs']-common)<=.2)
        qa[name]=stats;normalized[name]=check;files[name]='audio/'+name+'.wav'
    pitch_qa=compare(source,normalized['model'],rate,grid,command)
    changed=np.flatnonzero(np.abs(command)>=20)
    metadata={'title':'心墙01-轻度跑调','files':files,'duration':duration,'source_input':record['input'],
              'source_sha256':record['input_sha256'],'checkpoint_sha256':record['checkpoint_sha256'],
              'first_change_preview':max(0.,float(grid[changed[0]])-.5),'common_lufs':common,
              'audio_checks':qa,'pitch_checks':pitch_qa,'reference_input':False,
              'reference_display':{'source':str(ref),'source_sha256':sha(ref),'seconds':[begin/ref_rate,end/ref_rate],
                                  'method':'start estimated from already confirmed phrase correspondences; no time stretch; display only'},
              'technical_pass':all(x['technical_pass'] for x in qa.values()) and pitch_qa['pitch_technical_pass'],
              'listening':'pending_jade','reference_is_not_model_input':True,'effectiveness_pass':False,'diagnostic_only':True}
    dump(out/'delivery.json',metadata)
    if not metadata['technical_pass']:
        print(json.dumps(metadata,ensure_ascii=False));raise SystemExit(2)
    make_page(out,metadata)
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
