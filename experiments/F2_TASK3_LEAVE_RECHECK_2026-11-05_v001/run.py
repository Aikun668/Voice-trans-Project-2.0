from pathlib import Path
import os, json, hashlib, math, time
from importlib.metadata import version
os.environ.setdefault('USE_TF','0');os.environ.setdefault('HF_HUB_OFFLINE','1')
root=Path(__file__).resolve().parents[2]; here=Path(__file__).resolve().parent; out=here/'results'
if out.exists():raise SystemExit('输出目录已存在，拒绝覆盖')
out.mkdir()
import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner
answers=json.loads((root/'experiments/F2_MIN_EFFORT_PACK_2026-11-05_v001/results/jade_answers_2026-11-06.json').read_text(encoding='utf-8-sig'))
a=next(x for x in answers['lyrics'] if x['group']=='离开我的依赖01')
text=a['train_text']; assert text[10]=='边' and a['text'][10]=='旁' and len(text)==17
wav=root/'test_dataset/my_singing/rerecord_2026-11-04_f2/wav/离开我的依赖01-轻度跑调.wav'; model_path=Path(r'D:\My_Grand_Projects2026\Videos-Master-Project\Qwen3-ForcedAligner-0.6B')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
before=sha(wav);audio,sr=sf.read(str(wav),dtype='float32',always_2d=True);audio=audio.mean(axis=1)
torch.set_num_threads(4);device='cuda:0' if torch.cuda.is_available() else 'cpu';dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
print('加载本地 Qwen，按本人轻度版实唱“身边”重新对齐，设备：',device,flush=True)
model=Qwen3ForcedAligner.from_pretrained(str(model_path),dtype=dtype,device_map=device,local_files_only=True)
t=time.monotonic();aligned=list(model.align(audio=(audio,sr),text=text,language='Chinese')[0]);print('对齐耗时：',round(time.monotonic()-t,1),'秒',flush=True)
units=[{'index':i,'char':str(u.text),'start_s':float(u.start_time),'end_s':float(u.end_time)} for i,u in enumerate(aligned,1)]
checks={'text_matches':' '.join(x['char'] for x in units).replace(' ','')==text,'finite':all(math.isfinite(x['start_s']) and math.isfinite(x['end_s']) for x in units),'inside':all(0<=x['start_s']<=x['end_s']<=len(audio)/sr+.05 for x in units),'monotonic':all(units[i]['start_s']>=units[i-1]['start_s'] and units[i]['end_s']>=units[i-1]['end_s'] for i in range(1,len(units))),'zero_duration':sum(x['start_s']==x['end_s'] for x in units),'source_hash_unchanged':sha(wav)==before}
old=json.loads((root/'experiments/F2_FOUR_GROUPS_ALIGN_2026-11-05_v001/results/alignment.json').read_text(encoding='utf8'));old=next(x for x in old['items'] if x['group']=='离开我的依赖01' and x['label']=='轻度跑调')
assert len(units)==len(old['tokens'])==17
comparisons=[{'index':u['index'],'char':u['char'],'new_start_s':u['start_s'],'new_end_s':u['end_s'],'old_start_s':o['start_s'],'old_end_s':o['end_s'],'start_shift_s':round(u['start_s']-o['start_s'],4),'end_shift_s':round(u['end_s']-o['end_s'],4)} for u,o in zip(units,old['tokens'])]
result={'source':str(wav.relative_to(root)),'source_sha256':before,'source_time_base':'完整文件秒','model':str(model_path),'qwen_asr_version':version('qwen-asr'),'device':device,'input_text_basis':'Jade 确认本人轻度实唱身边；此前模型用了原唱候选身旁','checks':checks,'units':units,'old_comparison':comparisons,'status':'模型结果，仍需按范围人工验证；不自动用于训练','audio_output_generated':False}
(out/'alignment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('工程检查:',json.dumps(checks,ensure_ascii=False),flush=True)
for x in comparisons:
 if x['index'] in (10,11,12,13):print(x,flush=True)
