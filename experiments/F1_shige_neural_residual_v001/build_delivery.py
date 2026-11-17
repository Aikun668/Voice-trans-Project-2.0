from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'revision_report.json').read_text(encoding='utf-8'))
    assert report['technical_pass']
    destination=ROOT/'delivery';destination.mkdir()
    labels={'original':'原音','beautify_gentle':'神经美化 A · 较温和','beautify_standard':'神经美化 B · 较明显'}
    manifest=[]
    for name,item in report['outputs'].items():
        if name!='original': assert item['checks']['passed']
        src=ROOT/item['file']; dst=destination/src.name
        shutil.copyfile(src,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==item['sha256']
        manifest.append({'id':name,'label':labels[name],'file':'delivery/'+dst.name,
                         'sha256':item['sha256'],'frames':item['samples']['frames'],
                         'duration':item['samples']['duration_seconds']})
    (ROOT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    buttons=''.join(f'<button data-version="{m["id"]}" aria-pressed="{str(m["id"]=="original").lower()}">{m["label"]}</button>' for m in manifest)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>SHIGE · 完整神经美化候选</title><style>body{margin:60px auto;max-width:880px;padding:0 22px;background:#f3f6fa;color:#172337;font:17px/1.7 system-ui,sans-serif}h1{font-size:30px;margin-bottom:8px}.intro{color:#435267}.card{background:#fff;border:1px solid #d7dfe9;border-radius:16px;padding:28px;margin:24px 0}.versions{display:flex;gap:10px;flex-wrap:wrap}button{font:inherit;border:1px solid #b9c8db;background:#fff;color:#172337;border-radius:8px;padding:10px 16px;cursor:pointer}button[aria-pressed=true]{background:#2357a4;color:white;border-color:#2357a4}audio{width:100%;margin:18px 0}.hint{font-size:15px;color:#536177}.cues{display:flex;gap:10px;flex-wrap:wrap}.cues button{font-size:14px;padding:5px 10px}a{color:#2357a4}.status{padding-top:12px}h2{font-size:20px}</style><h1>SHIGE · 完整神经美化候选</h1><p class="intro">比较原音与两条完整美化。切换版本会接续同一时间，方便听同一句。</p><div class="card"><div class="versions">''' + buttons + '''</div><p id="status" class="status">当前：原音</p><audio id="player" controls preload="metadata" src="''' + manifest[0]['file'] + '''"></audio><div class="cues"><button data-time="0">从头听</button><button data-time="0.75">字头附近 · 0.75 秒</button><button data-time="16">尾段 · 16 秒</button></div></div><div class="card"><h2>这次需要判断</h2><p>哪条比原音更悦耳、更自然？是否仍有跳变、沙哑或变嗲？两条都不如原音也可以直接说。</p><p class="hint">A、B 使用同一神经美化映射，只有强度不同；保留原录音相位。完整 WAV 为 44.1 kHz、单声道、16-bit PCM，响度已匹配。</p><p class="hint">工程筛选通过；Agent 无直接听音能力，尚未判断美化是否成功。</p><a href="REPORT.md">本轮任务报告</a> · <a href="TASK_LOG.md">执行日志</a></div><script>const variants=''' + json.dumps(manifest,ensure_ascii=False) + ''';const player=document.getElementById('player');let busy=false;async function selectVersion(id){if(busy)return;busy=true;const item=variants.find(x=>x.id===id);const time=player.currentTime||0;const resume=!player.paused;player.pause();try{await new Promise((resolve,reject)=>{player.addEventListener('loadedmetadata',resolve,{once:true});player.addEventListener('error',reject,{once:true});player.src=item.file;player.load()});player.currentTime=Math.min(time,Math.max(0,player.duration-.05));document.querySelectorAll('[data-version]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.version===id)));document.getElementById('status').textContent='当前：'+item.label;if(resume)await player.play()}catch(e){document.getElementById('status').textContent='音频加载失败，请刷新重试'}finally{busy=false}}document.querySelectorAll('[data-version]').forEach(b=>b.addEventListener('click',()=>selectVersion(b.dataset.version)));document.querySelectorAll('[data-time]').forEach(b=>b.addEventListener('click',()=>{player.currentTime=Number(b.dataset.time)}));window.selectVersion=selectVersion;</script></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    rows=[]
    for name,item in report['outputs'].items():
        if name=='original': continue
        c=item['checks'];p=item['independent_pitch'];local=item['local_checks']
        rows.append(f"| {labels[name]} | {c['pitch_median_abs_cents']:.2f}/{c['pitch_p95_abs_cents']:.2f} | {p['median_abs_cents']:.2f}/{p['p95_abs_cents']:.2f} | {c['active_gain_spread_db']:.2f}/{local['min_relative_gain_active_db']:.2f} | {local['max_local_jump_ratio_active']:.2f} | 通过，pending_jade |")
    text='''# SHIGE 完整神经美化任务报告

日期：2026-10-26。主交付为原音、两条完整神经美化候选；不以声码器重建作为成果。声音是否更悦耳、是否仍跳变/沙哑/变嗲，等待Jade；Agent无直接听音能力。

## 本轮改变

原路径的a2a已压掉部分清音/字头，a2p叠加了这些重建损伤；声码器版本Jade仍听到跳变，旧全局采样差指标漏检。

本轮沿用NeuralSVB官方声学模型/美化映射权重，取完全相同条件和潜变量下a2p与a2a的对数幅度差，只把模型学习到的差异作用于原录音频谱。用librosa的STFT/ISTFT保留原始相位，不由估计F0重新合成声源；模型未覆盖的11.025kHz以上原录音信息保留。输入音高仅供模型分析，不修旋律、不移调、不拉伸。

这是项目的“神经频谱差分适配”，不是官方默认NeuralSVB波形直出，也不是新训练的端到端模型。新增代码负责差分、对应频率网格投影及调用现成变换，真实音色变化仍来自神经美化映射。是否产生正向美化收益尚未验证。

## 真正使用模型的证据

关闭映射后，特征差归零，原相位合成还原原音；记录的波形最大误差小于1e-6。开启映射后，模型频谱差RMS约7.03dB；A/B分别应用0.5/0.75倍差分。未把原声码器的变化、固定EQ或只改变响度充当美化。

同一神经模型可同时出多维音色差，不代表每一维都改善；声学模型的偏好未被自动当作Jade偏好。

## 一次有依据的修订

初测0.5与1.0均过旧全文件门槛。新增20ms局部瞬变检查沿用原2.5比值线，1.0在约0.94s达到2.693，因此拒绝。只把强档从1.0回收到0.75，局部比值降至2.149，通过；0.5正向工程基线保留。没有剪补该片段或放宽阈值。旧失败文件仍留在diagnostics/，不进入试听页。

新增局部检查可捕捉这次强档事件，但不是通用听觉跳变检测器，不能声称穷尽所有可听问题。保留原相位也不保证音色不会显得嗲；仍需要Jade判断。

## 最终工程结果

所有文件44100Hz/mono/PCM16，757286样本，约17.17s；实际导出PCM重新读取并完整FFmpeg解码。响度为共同目标-18LUFS（或根据峰值共同下调），真峰值≤-3dBTP，无削波。原f1_poc.compare_audio完整门槛均使用，包括原相位路线可适用的波形相关性和频带变化；本轮还要求第二Praat在10/35音分门槛内。

| 候选 | 项目音高中位/P95音分 | Praat中位/P95音分 | 动态跨度/局部最小增益dB | 最大局部瞬变比 | 状态 |
|---|---|---|---|---|---|
'''+'\n'.join(rows)+'''

两项有意义的适配器测试先失败后通过：关闭映射即使重建分支受损也应还原输入；已知差分产生预期增益、原高频保留且样本数不变。局部瞬变用实际1.0档失败→0.75档通过作为本轮修订证据。

## 文件与复现

- delivery/：原音与A/B完整WAV。delivery_manifest.json记录哈希/样本数。
- run_experiment.py：官方模型推理、差分合成、首轮检查。
- revise_strength.py：只调强度的一次修订及加强的工程检查。
- residual_adapter.py：必要适配层；使用librosa的变换与模型的mel基底。
- diagnostics/neural_features.npz：原mel、a2a、a2p、关映射mel、输入F0和潜变量。
- report.json、extended_checks.json、revision_report.json：原始检查证据，失败也保留。

来源：NeuralSVB https://github.com/MoonInTheRiver/NeuralSVB ，官方代码7a80bc51806103f09c7448b0daf93f49615ce52f，GPL-3.0，权重1030_vae_mle/step200000；librosa0.9.2现成STFT/ISTFT/mel（ISC）。未训练/微调、未安装依赖、未修改官方文件/旧实验/原素材。检查点及源哈希见report.json。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'TASK_LOG.md').write_text((ROOT/'PLAN.md').read_text(encoding='utf-8')+'\n\n## 最终筛选\n\n0.5与0.75通过完整与局部工程门槛，只有这两条完整美化进入交付；原1.0失败档保留内部。听感pending_jade。交付规格/解码/哈希已检查，浏览器播放验收待完成。\n',encoding='utf-8')
    print(json.dumps({'delivery_files':len(manifest),'candidates':[x['id'] for x in manifest if x['id']!='original']},ensure_ascii=False))


if __name__=='__main__':main()
