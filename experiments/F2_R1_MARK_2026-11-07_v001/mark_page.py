"""P2 逐音拍板页：把程序"拿不准没修"和"修得不到位"的音列出来，交 Jade 逐音标注。

输入：F2_R1_TRAIN_2026-11-07_v001 的对齐结果（重算一遍保证一致）。
输出：index.html（每音两个片段：你的录音、原唱对应位置；导出文字）。
只读源文件；输出仅写入本目录。

运行: .venv-f1-adaptive/Scripts/python.exe experiments/F2_R1_MARK_2026-11-07_v001/mark_page.py
"""
from pathlib import Path
import base64
import io
import json
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
if (DEST / 'index.html').exists() or (DEST / 'clips').exists():
    raise SystemExit(f'输出已存在，拒绝覆盖: {DEST}')
(DEST / 'clips').mkdir()

sys.path.insert(0, str(ROOT / 'experiments/F2_R1_TRAIN_2026-11-07_v001'))
import r1_pipeline as R            # noqa: E402
from run import load_anchors       # noqa: E402

SEGS = R.TEST_SEGS                 # 莫问归期两段（当前测试歌）
CTX_SELF = 1.2                     # 你的录音：音前后各留 1.2 秒上下文
CTX_REF = 0.8


def to_mp3_b64(p, t0, t1):
    sr = sf.info(str(p)).samplerate
    y, _sr = sf.read(str(p), start=max(int(t0 * sr), 0),
                     stop=int(np.ceil(min(t1, sf.info(str(p)).duration) * sr)), dtype='float64',
                     always_2d=True)
    y = y.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype='PCM_16', format='WAV')
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                           '-ac', '1', '-codec:a', 'libmp3lame', '-b:a', '64k', '-f', 'mp3', 'pipe:1'],
                          input=buf.getvalue(), capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise SystemExit(f'mp3 转码失败: {proc.stderr.decode()[:300]}')
    return base64.b64encode(proc.stdout).decode('ascii'), round(len(y) / sr, 1)


def main():
    anchors = load_anchors()
    # 上一轮模型实际修的量（apply_notes.csv），按 段+起点 对上
    prev = {}
    csv_p = ROOT / 'experiments/F2_R1_TRAIN_2026-11-07_v001/results/apply_notes.csv'
    if csv_p.exists():
        import csv as _csv
        for r in _csv.DictReader(open(csv_p, encoding='utf-8')):
            prev[(r['segment'], round(float(r['t0']), 2))] = float(r['model_shift'])
    cards, manifest = [], []
    for seg in SEGS:
        song = R.seg_song_take(seg)[0]
        a = R.analyze(R.seg_path(seg), R.ref_path(song), anchor=anchors.get(seg))
        seg_p, ref_p = R.seg_path(seg), R.ref_path(song)
        dur_self_total = sf.info(str(seg_p)).duration
        for n in a['notes']:
            n['model_shift'] = prev.get((seg, round(n['t0'], 2)), 0.0)
            _label, status = R.rule_of(n)          # rule_of 返回 (修正量, 状态)
            n['status'] = status
            underfix = status == 'fix' and abs(n['model_shift']) < 0.6 * abs(n['dev'])
            suspect = status != 'fix' and abs(n['dev']) >= 60
            if not (suspect or underfix):
                continue
            kind = 'underfix' if underfix else 'suspect'
            iid = f"m{len(cards)}"
            s0 = max(n['t0'] - CTX_SELF, 0.0)
            s1 = min(n['t1'] + CTX_SELF, dur_self_total)
            r0 = max(n['map_t0'] - CTX_REF, 0.0)
            r1 = n['map_t1'] + CTX_REF
            self_b64, self_d = to_mp3_b64(seg_p, s0, s1)
            ref_b64, ref_d = to_mp3_b64(ref_p, r0, r1)
            if kind == 'suspect':
                why = {':ref_unstable': ''}.get(n['status']) or {
                    'ref_unstable': '原唱这个位置在滑音，程序不敢拿它当目标',
                    'too_large': '偏差超过 150 音分，程序怕修错不敢动',
                    'too_short': '音太短，程序按惯例不修'}.get(n['status'], n['status'])
                q = f'''<p><b>这个音怎么处理？</b>（偏了 {n['dev']:+.0f} 音分，时长 {n['dur']} 秒）</p>
      <label><input type="radio" name="{iid}" value="要修">要修，向原唱的音靠</label>
      <label><input type="radio" name="{iid}" value="故意">我故意这样唱，别动</label>
      <label><input type="radio" name="{iid}" value="拿不准">拿不准</label>'''
                tip = f'程序跳过原因：{why}'
            else:
                q = f'''<p><b>程序只修了一半</b>（偏 {n['dev']:+.0f}，上一轮实际修了 {n['model_shift']:+.0f}）</p>
      <label><input type="radio" name="{iid}" value="修满">修到位</label>
      <label><input type="radio" name="{iid}" value="保持">现在这样就行</label>
      <label><input type="radio" name="{iid}" value="回退">别修这个</label>'''
                tip = '模型修正量偏小，未触发回退门槛'
            cards.append(f'''
  <div class="card">
    <h3>{seg} · {n['t0']:.2f} 秒 · 音名 {n['note']}</h3>
    <p class="meta">{tip}｜原唱这段参考的音：{n['ref_note']}</p>
    <div class="row">
      <div class="col"><p>① 你的录音（{self_d} 秒，该音在中间）</p><audio controls preload="none" src="data:audio/mpeg;base64,{self_b64}"></audio></div>
      <div class="col"><p>② 原唱对应位置（{ref_d} 秒）</p><audio controls preload="none" src="data:audio/mpeg;base64,{ref_b64}"></audio></div>
    </div>
    {q}
    <input type="text" class="note" id="{iid}_note" placeholder="补充（可不填）">
  </div>''')
            manifest.append({'id': iid, 'kind': kind, 'segment': seg, 't0': n['t0'], 't1': n['t1'],
                             'note': n['note'], 'ref_note': n['ref_note'], 'dev': n['dev'],
                             'model_shift': n.get('model_shift', 0), 'dur': n['dur'],
                             'ref_spread': round(n['ref_spread'], 1),
                             'self_clip_s': [round(s0, 2), round(s1, 2)],
                             'ref_clip_s': [round(r0, 2), round(r1, 2)]})
            print(f"[mark] {seg} {n['t0']:.2f}s {n['note']} dev {n['dev']:+.0f} {kind}")

    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    if not cards:
        (DEST / 'index.html').write_text('<html><body><p>没有需要拍板的音。</p></body></html>', encoding='utf-8')
        print('[mark] 没有需要拍板的音')
        return
    ids_js = json.dumps(manifest, ensure_ascii=False)
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F2 第一轮 P2 · 逐音拍板（{len(manifest)} 个音）</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 860px; margin: 0 auto; padding: 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; }} h3 {{ margin: 0 0 4px; font-size: 15px; }}
  .head {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 14px; line-height: 1.7; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  .meta {{ color: #666; font-size: 13px; margin: 2px 0 8px; }}
  .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .col {{ flex: 1 1 260px; }}
  .col p {{ margin: 4px 0; font-size: 13px; }}
  audio {{ width: 100%; height: 36px; }}
  label {{ margin-right: 14px; font-size: 14px; }}
  .note {{ width: 100%; margin-top: 6px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; box-sizing: border-box; }}
  button {{ padding: 10px 22px; font-size: 15px; cursor: pointer; }}
  textarea {{ width: 100%; height: 160px; margin-top: 10px; font-size: 13px; }}
</style>
</head>
<body>
<h1>F2 第一轮 P2 · 逐音拍板</h1>
<div class="head">
  <b>这是什么</b>：上一版修音你听不出差别，主要原因是"该修的音被程序按'拿不准不修'拦住了"。这页把拦下来的音逐个放给你，<b>你点一下，它就有资格被修</b>。<br>
  <b>怎么听</b>：①你的录音里那个音在片段中间；②原唱对应位置是程序认为"该唱成的音"。判断这个音是该修还是你故意的。<br>
  <b>声明</b>：Agent 不能听音，全部以你为准。预计 2~3 分钟。
</div>
{''.join(cards)}
<div style="text-align:center; margin: 20px 0;"><button onclick="collect()">生成答案</button></div>
<textarea id="out" readonly placeholder="点"生成答案"后，全选复制发回即可"></textarea>
<script>
const ITEMS = {ids_js};
function collect() {{
  const lines = ['【F2 第一轮 P2 逐音拍板 2026-11-07】'];
  for (const it of ITEMS) {{
    const v = document.querySelector('input[name="' + it.id + '"]:checked');
    const note = document.getElementById(it.id + '_note').value.trim();
    lines.push(it.segment + ' ' + it.t0.toFixed(2) + 's ' + it.note + '（偏' + it.dev + '）：'
      + (v ? v.value : '未答') + (note ? '｜' + note : ''));
  }}
  document.getElementById('out').value = lines.join('\\n');
  document.getElementById('out').select();
}}
</script>
</body>
</html>'''
    (DEST / 'index.html').write_text(html, encoding='utf-8')
    print(f'[mark] {len(manifest)} 个音待拍板；页面 {DEST / "index.html"}')


if __name__ == '__main__':
    main()
