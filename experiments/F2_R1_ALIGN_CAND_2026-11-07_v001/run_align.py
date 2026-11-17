"""P1 原唱候选定位（F2 第一轮有参考修音 · 训练计划书 P1）。

对 9 段未核对的非严重档录音，用 Praat F0（0.1s 步长）+ 子序列 DTW（100 音分调差网格）
在三首原唱分离人声中定位候选区间，并输出 Jade 核对页（音频内嵌 Base64）。

只读源文件；所有输出仅写入本实验目录。运行前/后校验源文件 SHA-256 一致。
方法与 experiments/F2_DATA_INVENTORY_2026-11-05_v001 相同（该法已在《心墙》对照版上由 Jade 确认可用）。

运行: .venv-f1-adaptive/Scripts/python.exe experiments/F2_R1_ALIGN_CAND_2026-11-07_v001/run_align.py
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import sys

import numpy as np
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parents[2] / '.venv-f2/Lib/site-packages'))
import parselmouth

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
RESULTS = DEST / 'results'
AUDIO = DEST / 'audio'
if RESULTS.exists() or AUDIO.exists():
    raise SystemExit(f'输出目录已存在，拒绝覆盖: {DEST}')
RESULTS.mkdir()
AUDIO.mkdir()

sys.path.insert(0, str(ROOT / 'f2'))
from poc_phase4_realign import cost_matrix, subseq_dtw

BASE = ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2'
HOP = 0.1
PITCH_FLOOR, PITCH_CEIL = 75.0, 600.0
KEY_GRID = range(-600, 600, 100)
PREVIEW_SR = 16000

# 心墙01-对照微微跑调 已由 Jade 确认（112.8–121.1s），不在本次核对范围。
SONGS = {
    '心墙': ('心墙_郭静_vocals.wav', ['心墙01-轻度跑调']),
    '离开我的依赖': ('离开我的依赖_王艳薇_vocals.wav',
                 ['离开我的依赖01-对照微微跑调', '离开我的依赖01-轻度跑调',
                  '离开我的依赖02-对照微微跑调', '离开我的依赖02-轻度跑调']),
    '莫问归期': ('莫问归期_蒋雪儿_vocals.wav',
              ['莫问归期01-对照微微跑调', '莫问归期01-轻度跑调',
               '莫问归期02-对照微微跑调', '莫问归期02-轻度跑调']),
}


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cents_of(y, sr):
    hz = parselmouth.Sound(y, sr).to_pitch(
        time_step=HOP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEIL
    ).selected_array['frequency']
    c = np.full(len(hz), np.nan)
    v = hz > 0
    c[v] = 1200 * np.log2(hz[v] / 440.0)
    return c


def load_cents(p):
    y, sr = sf.read(str(p), dtype='float64', always_2d=True)
    y = y.mean(axis=1)          # sf 的 (N,1) 会被 parselmouth 误读为 N 声道，压成一维
    return cents_of(y, sr), sr, y


def locate(q, ref, n_candidates=2):
    """返回按代价排序的前 n 个候选；找次优时屏蔽已选参考区间。"""
    out = []
    r_work = ref.copy()
    for _ in range(n_candidates):
        best = None
        for key in KEY_GRID:
            cost, path = subseq_dtw(cost_matrix(q, r_work, float(key)))
            if best is None or cost < best[0]:
                best = (cost, path, key)
        cost, path, key = best
        rev, _ = subseq_dtw(cost_matrix(q, r_work[::-1].copy(), float(key)))
        j0, j1 = path[0][1], path[-1][1]
        out.append({'ref_start_s': round(j0 * HOP, 2), 'ref_end_s': round(j1 * HOP, 2),
                    'coarse_key_cents': int(key), 'cost': round(float(cost), 2),
                    'reversed_cost': round(float(rev), 2)})
        r_work[max(0, j0 - 2): j1 + 3] = np.nan
    return out


def to_b64_wav(y, sr):
    import subprocess
    # 预览/浏览器打不开几 MB 的内嵌 WAV：转 64k 单声道 mp3 内嵌（判断"是否同一句"足够）
    y = np.asarray(y, dtype='float64')
    if y.ndim > 1:
        y = y.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype='PCM_16', format='WAV')
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                           '-i', 'pipe:0', '-ac', '1', '-codec:a', 'libmp3lame',
                           '-b:a', '64k', '-f', 'mp3', 'pipe:1'],
                          input=buf.getvalue(), capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise SystemExit(f'mp3 转码失败: {proc.stderr.decode()[:300]}')
    return base64.b64encode(proc.stdout).decode('ascii')


def clip(y, t0, t1, sr):
    i0, i1 = max(int(t0 * sr), 0), min(int(np.ceil(t1 * sr)), len(y))
    return y[i0:i1]


def main():
    targets = sorted(p for p in (BASE / 'wav').glob('*.wav') if '严重' not in p.name)
    hashes_before = {p.name: sha256(p) for p in targets}
    refs = {p.name: sha256(p) for p in (BASE / 'ref_vocals').glob('*.wav')}

    cards, results = [], []
    for song, (ref_name, seg_names) in SONGS.items():
        ref_path = BASE / 'ref_vocals' / ref_name
        ref_cents, ref_sr, ref_y = load_cents(ref_path)
        print(f'[{song}] 参考 {ref_name} {len(ref_cents)*HOP:.1f}s', flush=True)
        for seg in seg_names:
            seg_path = BASE / 'wav' / f'{seg}.wav'
            q_cents, q_sr, q_y = load_cents(seg_path)
            cands = locate(q_cents, ref_cents)
            c1 = cands[0]
            self_b64 = to_b64_wav(q_y, q_sr)
            ref_b64 = to_b64_wav(clip(ref_y, c1['ref_start_s'], c1['ref_end_s'], ref_sr), ref_sr)
            print(f"  {seg}: 候选 {c1['ref_start_s']}-{c1['ref_end_s']}s "
                  f"key {c1['coarse_key_cents']:+d} cost {c1['cost']:.1f} "
                  f"(反转基线 {c1['reversed_cost']:.1f}); 次优 {cands[1]['ref_start_s']}-{cands[1]['ref_end_s']}s", flush=True)
            sid = f'q{len(cards)}'
            cards.append({'id': sid, 'song': song, 'seg': seg,
                          'ref_start': c1['ref_start_s'], 'ref_end': c1['ref_end_s'],
                          'self_audio': self_b64, 'ref_audio': ref_b64,
                          'dur_self': round(len(q_y) / q_sr, 1),
                          'dur_ref': round(c1['ref_end_s'] - c1['ref_start_s'], 1)})
            results.append({'song': song, 'segment': seg,
                            'target': str(seg_path.relative_to(ROOT)),
                            'reference': str(ref_path.relative_to(ROOT)),
                            'target_seconds': round(len(q_y) / q_sr, 2),
                            'candidates': cands,
                            'status': '自动候选，歌词与唱法未核对；不可作为训练标签'})

    hashes_after = {p.name: sha256(p) for p in targets}
    refs_after = {p.name: sha256(p) for p in (BASE / 'ref_vocals').glob('*.wav')}
    unchanged = (hashes_before == hashes_after) and (refs == refs_after)
    if not unchanged:
        raise SystemExit('源文件哈希发生变化，终止！')

    (RESULTS / 'candidates.json').write_text(
        json.dumps({'results': results, 'source_hashes_unchanged': unchanged,
                    'method': 'Praat F0 0.1s + 子序列DTW + 100音分调差网格（与 F2_DATA_INVENTORY_2026-11-05_v001 相同）',
                    'notes': ['候选区间是程序自动定位，歌词和唱法未核对，不可作为训练标签。',
                              'coarse_key_cents 为本人相对原唱的整体调差粗估（100 音分步长）。',
                              '次优候选已记录，供首选被否决时快速复查，不进核对页。']},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    card_html = []
    for c in cards:
        card_html.append(f'''
  <div class="card">
    <h3>{c['song']} · {c['seg'].replace(c['song'], '', 1).strip('·- ')}</h3>
    <p class="meta">原唱候选区间：{c['ref_start']} – {c['ref_end']} 秒（程序自动定位，可能有误）</p>
    <div class="row">
      <div class="col"><p>① 你的录音（{c['dur_self']} 秒）</p><audio controls preload="none" src="data:audio/mpeg;base64,{c['self_audio']}"></audio></div>
      <div class="col"><p>② 原唱候选（{c['dur_ref']} 秒）</p><audio controls preload="none" src="data:audio/mpeg;base64,{c['ref_audio']}"></audio></div>
    </div>
    <p>这两段是同一句吗？</p>
    <label><input type="radio" name="{c['id']}" value="是同一句">是同一句</label>
    <label><input type="radio" name="{c['id']}" value="不是">不是</label>
    <label><input type="radio" name="{c['id']}" value="拿不准">拿不准</label>
    <input type="text" class="note" id="{c['id']}_note" placeholder="补充（可不填）：比如漏字、顺序不一样、原唱位置不对">
  </div>''')

    ids_js = json.dumps([c['id'] for c in cards], ensure_ascii=False)
    segs_js = json.dumps({c['id']: f"{c['song']}·{c['seg']}" for c in cards}, ensure_ascii=False)
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F2 第一轮 P1 · 原唱候选核对（9 段）</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 860px; margin: 0 auto; padding: 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; }} h3 {{ margin: 0 0 4px; font-size: 16px; }}
  .head {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 14px; line-height: 1.7; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  .meta {{ color: #666; font-size: 13px; margin: 2px 0 8px; }}
  .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .col {{ flex: 1 1 260px; }}
  .col p {{ margin: 4px 0; font-size: 13px; }}
  audio {{ width: 100%; height: 36px; }}
  label {{ margin-right: 14px; font-size: 14px; }}
  .note {{ width: 100%; margin-top: 6px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; box-sizing: border-box; }}
  .song-title {{ font-size: 17px; margin: 20px 0 8px; color: #14532d; }}
  button {{ padding: 10px 22px; font-size: 15px; cursor: pointer; }}
  textarea {{ width: 100%; height: 180px; margin-top: 10px; font-size: 13px; }}
</style>
</head>
<body>
<h1>F2 第一轮 P1 · 原唱候选核对</h1>
<div class="head">
  <b>怎么听</b>：戴耳机、安静环境；每题先听①你的录音，再听②原唱候选，判断是不是同一句。<br>
  <b>说明</b>：② 的位置是程序自动找的，可能有误——找错了就选"不是"，我们会换方法，不影响你。<br>
  <b>背景</b>：程序没法听懂歌词，需要你确认"你唱的这段"和"原唱哪一段"是同一句，后面修音才拿得准参考。<br>
  <b>声明</b>：Agent 不能听音，本页全部结论以你的判断为准。预计 5–8 分钟。<br>
  已确认过、不需要再听的：《心墙》对照微微跑调版（原唱约 112.8–121.1 秒）。
</div>
{''.join(card_html)}
<div style="text-align:center; margin: 20px 0;">
  <button onclick="collect()">生成答案</button>
</div>
<textarea id="out" readonly placeholder="点"生成答案"后，全选复制这里的内容发回即可"></textarea>
<script>
const IDS = {ids_js};
const SEGS = {segs_js};
function collect() {{
  const lines = ['【F2 第一轮 P1 原唱候选核对 2026-11-07】'];
  for (const id of IDS) {{
    const v = document.querySelector('input[name="' + id + '"]:checked');
    const note = document.getElementById(id + '_note').value.trim();
    lines.push(SEGS[id] + '：' + (v ? v.value : '未答') + (note ? '｜补充：' + note : ''));
  }}
  lines.push('补充：' + (document.getElementById('overall').value.trim() || '（未填）'));
  document.getElementById('out').value = lines.join('\\n');
  document.getElementById('out').select();
}}
</script>
<input type="text" class="note" id="overall" placeholder="整体补充（可不填）" style="max-width:860px">
</body>
</html>'''
    (DEST / 'index.html').write_text(html, encoding='utf-8')
    print(f'完成：{len(cards)} 段候选；页面 {DEST / "index.html"}；源文件哈希一致: {unchanged}')


if __name__ == '__main__':
    main()
