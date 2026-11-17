"""Filter engineering-pass audio, preserve failures, write review page/report."""
from pathlib import Path
import hashlib
import html
import json
import shutil

ROOT = Path(__file__).resolve().parent
LABELS = {'molihua_normal': '茉莉花 · 正常', 'shige_normal': 'SHIGE · 正常',
          'zuixuan_normal': 'zuixuan · 正常', 'shige_medium': 'SHIGE · 中度跑调',
          'shige_severe': 'SHIGE · 严重跑调'}


def main():
    reports = [json.loads((ROOT / run / 'report.json').read_text(encoding='utf-8'))
               for run in ['full_normal', 'full_severity']]
    destination = ROOT / 'delivery'
    if destination.exists():
        raise SystemExit('Delivery already exists; refusing overwrite.')
    destination.mkdir()
    manifest = []
    rows = []
    cards = []
    processed, passed = 0, 0
    for run, report in zip(['full_normal', 'full_severity'], reports):
        for clip in report['clips']:
            stem = Path(clip['source']).stem
            outputs = clip['outputs']
            selected = ['original']
            if outputs['vocoder']['checks']['technical_pass']:
                selected.append('vocoder')
            chosen = None
            if outputs['a2p']['checks']['technical_pass']:
                chosen = 'a2p'
            elif outputs['a2p_mean']['checks']['technical_pass']:
                chosen = 'a2p_mean'
            if chosen:
                selected.append(chosen)
            entries = []
            card = ['<section><h2>' + LABELS[stem] + '</h2>']
            if stem in ['zuixuan_normal', 'shige_severe']:
                card.append('<p class="blocked">声码器通过现有工程拦截，但另一音高诊断仍有边界提醒。请重点留意换音、沙哑与跳变；尚未确认听感改善。</p>')
            if not chosen:
                card.append('<p class="blocked">完整美化路径未通过工程检查，本组暂不提供美化试听候选。</p>')
            for name in selected:
                src = ROOT / run / stem / outputs[name]['file']
                dst = destination / src.name
                shutil.copyfile(src, dst)
                assert hashlib.sha256(dst.read_bytes()).hexdigest() == outputs[name]['sha256']
                title = {'original': '原音（仅匹配响度）', 'vocoder': '声码器重建对照',
                         'a2p': '完整美化 · 官方采样', 'a2p_mean': '完整美化 · 后验均值适配'}[name]
                card.append('<div class="track"><label>' + title + '</label><audio controls preload="metadata" src="delivery/' + dst.name + '"></audio></div>')
                entries.append({'variant': name, 'file': 'delivery/' + dst.name, 'sha256': outputs[name]['sha256'],
                                'technical_pass': name != 'original', 'role': 'reference' if name == 'original' else 'engineering_pass_pending_jade',
                                'frames': outputs[name]['samples']['frames'],
                                'duration_seconds': outputs[name]['samples']['duration_seconds']})
            if stem == 'zuixuan_normal':
                card.append('<p>重点比较约 2、3、9、17–19 秒，也请留意全程其他位置；这些是此前 a2p 的反馈时段。</p>')
            card.append('</section>')
            cards.append(''.join(card))
            manifest.append({'clip': stem, 'label': LABELS[stem], 'listening_status': 'pending_jade',
                             'selected_beautifier': chosen, 'tracks': entries})
            for variant, data in outputs.items():
                if variant == 'original':
                    continue
                processed += 1
                check = data['checks']
                passed += int(check['technical_pass'])
                pitch = check['pitch_vs_original']
                praat = check['pitch_vs_original_praat_info_only']
                rows.append(f"| {LABELS[stem]} | {variant} | {'PASS' if check['technical_pass'] else 'BLOCKED'} | {pitch.get('median_abs_cents', float('nan')):.2f}/{pitch.get('p95_abs_cents', float('nan')):.2f} | {praat.get('median_abs_cents', float('nan')):.2f}/{praat.get('p95_abs_cents', float('nan')):.2f} | {check['active_gain']['spread_db']:.2f}/{check['active_gain']['min_db']:.2f} | {', '.join(check['errors']) or '无'} |")
    (ROOT / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    text = f'''# 第二、三步任务报告

完成范围：修正Praat音高帧的时间定位，接回原声学模型与声码器，复跑三条正常素材与SHIGE中/严重对照；生成一次后验均值推理修订并筛选。Agent不能直接听音，全部听感为pending_jade。

## 可验证的结论

- 先修时间坐标，zuixuan纯声码器音高偏差由17.23/106.76降到4.44/33.17音分，现有工程检查通过。未用修音高、平滑旋律或移调。
- 保留原官方PWG的50–11025Hz/log10特征路径：声学模型输入与输出沿用同一约定，直接接原声码器。此前范围匹配实验未改善，不能只依配置文件认定声码器必须改成0–8000Hz；实际训练特征仍有公开信息缺口。
- 共核验{processed}条处理输出，{passed}条通过、{processed-passed}条拦截。每条最多筛出声码器对照和一个完整美化候选；若官方采样已通过则优先保留它，不凭指标猜测均值版本更悦耳。
- 全部最终WAV为44100Hz/单声道/PCM16，各自样本数与原录音完全一致；完整FFmpeg解码一致；每条素材内共同响度和真峰值余量按原规则计算。
- 原素材和之前实验WAV哈希保持。诊断失败产物保留在full_normal/full_severity，delivery只放通过者及匹配响度原音。
- 中度/严重使用SHIGE，因为zuixuan没有medium；未用mild冒充medium。这些是保留原始跑调的音色处理实验，不是音准修复。

## 工程结果（项目阈值不变）

项目音高偏差门槛为中位≤10、P95≤35音分；动态跨度≤8dB、局部相对增益≥-12dB、相邻采样跳变比≤2.5。Praat原为独立信息项，边界/失败值照实列出，不隐去。不同提取器的有效帧集合有区别，不把两个汇总数的差当严格逐帧改善量。

| 素材 | 变体 | 状态 | 项目音高中位/P95 | Praat中位/P95 | 动态跨度/最小增益dB | 拦截原因 |
|---|---|---|---|---|---|---|
''' + '\n'.join(rows) + '''

## 自查与限制

时间适配器有三项合成信号/边界测试：保留无声间隔、随时间移动的音高落在真实帧中心、精确保留时长且拒绝补零掩盖缺失。先有失败后实现，随后增加移动音高回归用例。

后验均值修订仅把同一声学模型中的随机潜变量改为其均值，作为有限的确定性推理对照；没有训练/微调，不等同官方默认流程。所有未通过的变体仍按拦截处理，没有通过混原音、增益补洞或调宽门槛隐藏问题。

相邻采样跳变比通过不能证明听觉跳变消失；音高估计也可能受谐波/气声影响。44.1kHz交付由22.05kHz模型输出重采样，不恢复高频带宽。模型单WAV/自身F0/恒等对齐属于项目适配，官方原始输入推理仍WIP；中文内容特征泛化和说话人特征一致性仍需听感验证。

## Jade验收

打开index.html。先以固定播放音量比较原音和声码器对照，判断全程沙哑、局部跳变是否减轻；再听本组若有的完整美化候选，判断是否更自然悦耳、是否变嗲、是否仍像本人。可直接反馈更喜欢哪个，以及不满意的时间点。没有候选的组不要求勉强选择。

正常/中度/严重是不同演唱录音，不能把它们的绝对指标或主观质量差异全部归因于处理；每条只与自己原音比较。正常单素材暂通过之后再看其他录音能否复现收益。

正常SHIGE的额外局部衰减集中在约0.8–1.0s、16.8–17.0s，最低约-26.4dB；zuixuan包含约2.24s、7.8–8.2s、9.48–9.62s、19.34–19.52s等事件。这些事件多落在源F0无基频帧中心，可能涉及清音/呼吸/句子边缘，但无基频不等于没有歌词或重要声音。不能以此取消工程拦截，也不将这些时刻自动当成Jade全部主观跳变的精确边界。详表见两组dropout_localization.json。

参数、版本及中间结果见run_pipeline.py、feature_contract.py、provenance.json、各组report.json和execution.log。工作日志见TASK_LOG.md。既有production入口未改动；本轮为隔离实验，尚未把结果整合成正式产品实现。
'''
    (ROOT / 'REPORT.md').write_text(text, encoding='utf-8')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1 · 接口修正后试听</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#172337;max-width:960px;margin:40px auto;padding:0 20px}h1{font-size:28px}h2{font-size:21px;margin:0 0 14px}section{background:white;border:1px solid #dae1e9;border-radius:14px;padding:24px;margin:22px 0}.track{display:grid;grid-template-columns:240px 1fr;gap:16px;align-items:center;margin:14px 0}audio{width:100%}.note{background:#e8f0fb;padding:16px;border-radius:10px}.blocked{color:#864b16}@media(max-width:650px){.track{grid-template-columns:1fr;gap:4px}}a{color:#2055ad}</style><h1>F1 · 接口修正后试听</h1><p class="note">工程筛选完成，听感等待 Jade。Agent 无直接听音能力。先比较原音与声码器重建，再判断完整美化是否带来收益。中度/严重版保留原有跑调。</p><p><a href="REPORT.md">任务报告</a> · <a href="TASK_LOG.md">任务日志</a></p>''' + ''.join(cards) + '''<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})}));</script></html>'''
    (ROOT / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({'processed': processed, 'technical_pass': passed, 'blocked': processed-passed,
                      'selected': [{m['clip']: m['selected_beautifier']} for m in manifest]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
