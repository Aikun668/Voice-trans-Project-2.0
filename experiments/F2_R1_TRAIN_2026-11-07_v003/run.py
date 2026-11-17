"""F2 第一轮 · train / apply / page 三个子命令。

运行（.venv-f1-adaptive 环境）:
  python experiments/F2_R1_TRAIN_2026-11-07_v001/run.py train   # 建样本表 + 训练
  python experiments/F2_R1_TRAIN_2026-11-07_v001/run.py apply   # 测试歌修正 + 工程检查
  python experiments/F2_R1_TRAIN_2026-11-07_v001/run.py page    # 试听页

train 产出 results/{samples.csv,model.pkl,train_report.json}
apply  产出 apply/*.wav 与 results/{apply_notes.csv,qa_report.json}
page   产出 index.html
所有输出只写入本实验目录；目录已存在则拒绝运行。
"""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys

import numpy as np

import r1_pipeline as R
from poc_phase1 import read_wav_pcm16, write_wav_pcm16  # noqa: E402  (r1_pipeline 已设好 f2 路径)

DEST = Path(__file__).resolve().parent
RESULTS = DEST / 'results'
APPLY = DEST / 'apply'


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fresh(d):
    if d.exists():
        raise SystemExit(f'输出目录已存在，拒绝覆盖: {d}')
    d.mkdir(parents=True)


def load_anchors():
    """每段的对齐锚点 = P1 候选（等 Jade 确认）。心墙对照版用 Jade 已确认区间。

    同一 take 的两版（对照/轻度）唱的是同一处歌词，理应对应原唱同一区间；
    旋律重复会让两版各自配到不同段落（2026-11-07 实测莫问01、离开01）。
    处理：区间不重叠时，两版都跟随"代价/反转基线"比值更小（匹配更可信）的一版。
    """
    anchors, quality = {}, {}
    cand_file = DEST.parent / 'F2_R1_ALIGN_CAND_2026-11-07_v001' / 'results' / 'candidates.json'
    if cand_file.exists():
        for r in json.loads(cand_file.read_text(encoding='utf-8'))['results']:
            c = r['candidates'][0]
            anchors[r['segment']] = (c['ref_start_s'], c['ref_end_s'], c['coarse_key_cents'])
            quality[r['segment']] = c['cost'] / max(c['reversed_cost'], 1e-6)
    inv_file = DEST.parent / 'F2_DATA_INVENTORY_2026-11-05_v001' / 'results' / 'inventory.json'
    if inv_file.exists():
        for a in json.loads(inv_file.read_text(encoding='utf-8'))['first_song_alignment_candidates']:
            seg = Path(a['target']).stem
            anchors[seg] = (a['reference_start_s'], a['reference_end_s'], a['coarse_key_cents'])
            quality[seg] = a['cost'] / max(a['reversed_cost'], 1e-6)

    groups = {}
    for seg in list(anchors):
        song, take = R.seg_song_take(seg)
        groups.setdefault((song, take), []).append(seg)
    for (song, take), segs in groups.items():
        if len(segs) < 2:
            continue
        (a0, b0, _), (a1, b1, _) = anchors[segs[0]], anchors[segs[1]]
        overlap = min(b0, b1) - max(a0, a1)
        if overlap < -3.0:                      # 减去余量后仍不重叠 → 镜像错配
            leader = min(segs, key=lambda s: quality.get(s, 9e9))
            lt0, lt1, _ = anchors[leader]
            for seg in segs:
                _, _, k = anchors[seg]
                anchors[seg] = (lt0, lt1, k)
            print(f"[anchor] {song}{take} 两版区间不重叠，跟随可信度更高的 {leader}"
                  f"（原唱 {lt0}-{lt1}s）")
    return anchors


def cmd_train():
    fresh(RESULTS)
    from sklearn.ensemble import GradientBoostingRegressor

    anchors = load_anchors()
    marks = R.load_jade_marks()
    rows, report = [], {'segments': []}
    for seg in R.TRAIN_SEGS:
        song = seg.split('01')[0].split('02')[0].rstrip('-')
        a = R.analyze(R.seg_path(seg), R.ref_path(song), anchor=anchors.get(seg))
        n_fix = 0
        for n in a['notes']:
            jm = R.jade_mark(seg, n, marks)
            if jm == 'keep':                       # Jade 标"故意这样唱"：永不修
                label, status = 0.0, 'jade_keep'
            elif jm == 'fix':                      # Jade 标"希望修"：强制入修音集
                label, status = round(float(np.clip(-n['dev'], -R.FIX_MAX_ABS, R.FIX_MAX_ABS)), 1), 'jade_fix'
            else:
                label, status = R.rule_of(n)
            n_fix += status in ('fix', 'jade_fix')
            rows.append({'segment': seg, **{k: n[k] for k in ('t0', 't1', 'dur', 'cents')},
                         'note': n['note'], 'ref_note': n['ref_note'], 'dev': n['dev'],
                         'ref_spread': round(n['ref_spread'], 1), 'ref_voiced': n['ref_voiced'],
                         'jade_mark': jm or '-', 'label_shift': label, 'status': status,
                         **{f: round(v, 2) for f, v in zip(R.FEATURES, R.feature_row(n))}})
        report['segments'].append({'segment': seg, 'key_cents': a['key_cents'],
                                   'cost_per_frame': a['cost_per_frame'],
                                   'ref_interval_s': [a['ref_t0'], a['ref_t1']],
                                   'notes': len(a['notes']), 'fix_labeled': n_fix})
        print(f"[train] {seg}: key {a['key_cents']:+.0f} 候选区间 {a['ref_t0']}-{a['ref_t1']}s "
              f"音 {len(a['notes'])} 待修标签 {n_fix}")

    with open(RESULTS / 'samples.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    X = np.array([[r[f] for f in R.FEATURES] for r in rows])
    y = np.array([r['label_shift'] for r in rows])
    fix_idx = y != 0
    # 类别极不均衡（待修样本占少数）：模型只学"够格修的音修多少"，
    # 否则回归器会被大量 0 标签拖得不敢输出（首轮实测全部 <20 音分 → 全跳过）。
    Xf, yf = X[fix_idx], y[fix_idx]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0)
    model.fit(Xf, yf)
    pred_fix = model.predict(Xf)
    mae_fix = float(np.mean(np.abs(pred_fix - yf))) if len(yf) else None
    pred_all = np.zeros_like(y)
    pred_all[fix_idx] = pred_fix
    mae_all = float(np.mean(np.abs(pred_all - y)))
    with open(RESULTS / 'model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': R.FEATURES}, f)

    report.update({'samples': len(rows), 'fix_labeled_total': int(fix_idx.sum()),
                   'train_mae_all_cents': round(mae_all, 1),
                   'train_mae_fix_cents': round(mae_fix, 1) if mae_fix is not None else None,
                   'baseline_mae_zero_cents': round(float(np.mean(np.abs(y))), 1),
                   'model': 'GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_leaf=2)，仅在待修子集上训练',
                   'honesty': ['标签由守护规则从自动候选对齐推导，尚无 Jade 的故意唱法标注（P2 后替换标签源）。',
                               '训练样本极少（待修仅 7 个），模型只是占位基线，可信修正量以守护规则回退为准。',
                               'train_mae 是训练集拟合度，不是测试成绩，更不是听感；听感一律 pending_jade。']})
    (RESULTS / 'train_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print(f"[train] 样本 {len(rows)}（待修 {int(fix_idx.sum())}）；MAE 全部 {mae_all:.1f} / "
          f"待修子集 {mae_fix and round(mae_fix, 1)} / 全零基线 {report['baseline_mae_zero_cents']} 音分")


def cmd_apply():
    if not (RESULTS / 'model.pkl').exists():
        raise SystemExit('先运行 train')
    fresh(APPLY)
    with open(RESULTS / 'model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model, feats = bundle['model'], bundle['features']

    src_files = [R.seg_path(s) for s in R.DEMO_SEGS] + \
                [R.ref_path(R.seg_song_take(s)[0]) for s in R.DEMO_SEGS]
    hashes = {p.name: sha256(p) for p in src_files}

    note_rows, qa = [], {'segments': []}
    anchors = load_anchors()
    marks = R.load_jade_marks()
    for seg in R.DEMO_SEGS:
        song = R.seg_song_take(seg)[0]
        role = '训练内验证' if seg in R.TRAIN_SEGS else '独立测试（没学过）'
        a = R.analyze(R.seg_path(seg), R.ref_path(song), anchor=anchors.get(seg))
        x, sr = read_wav_pcm16(R.seg_path(seg))
        fixes, seg_rows = [], []
        for n in a['notes']:
            jm = R.jade_mark(seg, n, marks)
            label, status = R.rule_of(n)
            shift, source = 0.0, '-'
            if jm == 'keep':                       # Jade 标"故意"：强制不动
                status, source = 'jade_keep', 'jade'
            elif jm == 'fix':                      # Jade 标"希望修"：直接修到参考音
                shift, status, source = float(np.clip(-n['dev'], -R.FIX_MAX_ABS, R.FIX_MAX_ABS)), 'jade_fix', 'jade'
                fixes.append((n['t0'], n['t1'], shift))
            elif status == 'fix':
                pred = float(np.clip(model.predict([R.feature_row(n)])[0], -R.FIX_MAX_ABS, R.FIX_MAX_ABS))
                if abs(pred - label) > 60.0:       # 样本少，模型输出离参考目标太远就不信它
                    shift, source = label, 'model_clamped'
                else:
                    shift, source = pred, 'model'
                if abs(shift) < 30.0:              # 不敢下手 → 回退守护规则修正量
                    shift, source = label, 'rule_fallback'
                fixes.append((n['t0'], n['t1'], shift))
            seg_rows.append({'segment': seg, **{k: n[k] for k in ('t0', 't1', 'dur')},
                             'note': n['note'], 'ref_note': n['ref_note'], 'dev': n['dev'],
                             'ref_spread': round(n['ref_spread'], 1), 'ref_voiced': n['ref_voiced'],
                             'jade_mark': jm or '-', 'label_shift': label,
                             'model_shift': round(shift, 1), 'shift_source': source, 'status': status})
            note_rows.append(seg_rows[-1])

        out_raw, w = R.corrected_audio(x.astype(np.float64), sr, fixes)
        g = R.gain_match(x, out_raw)
        xg, og = x * g, out_raw * g
        stem = seg
        write_wav_pcm16(APPLY / f'{stem}_原音同响.wav', xg, sr)
        write_wav_pcm16(APPLY / f'{stem}_修正版.wav', og, sr)

        outside = w == 0.0   # 交叉淡化斜坡属于拼接区；恒等检查只看掩码严格为 0 处
        max_diff_outside = float(np.max(np.abs(out_raw - x)[outside])) if outside.any() else 0.0
        lufs_c, tp_c = R.ebur128(APPLY / f'{stem}_修正版.wav')
        lufs_a, tp_a = R.ebur128(APPLY / f'{stem}_原音同响.wav')
        qc = R.cents_of(out_raw, sr)   # 增益前比较：Praat 浊音判定吃绝对强度，混入增益会假跌
        qo = R.cents_of(x.astype(np.float64), sr)
        both = ~np.isnan(qc) & ~np.isnan(qo)
        either = ~np.isnan(qc) | ~np.isnan(qo)
        # 浊音重合率 = 双侧都浊 / 任一侧浊（轨迹完全一致时 = 1.0；ROADMAP §8.2 门槛 0.95）
        overlap = float(np.sum(both) / max(np.sum(either), 1)) if len(qc) == len(qo) else float('nan')
        qa['segments'].append({'segment': seg, 'role': role, 'key_cents': a['key_cents'],
                               'ref_interval_s': [a['ref_t0'], a['ref_t1']],
                               'notes': len(a['notes']), 'fixed': len(fixes),
                               'samples': len(x), 'samplerate': sr,
                               'max_sample_diff_outside_splice': max_diff_outside,
                               'corrected_LUFS': lufs_c, 'corrected_dbtp': tp_c,
                               'anchor_LUFS': lufs_a, 'anchor_dbtp': tp_a,
                               'voicing_overlap_corrected_vs_orig': round(overlap, 4),
                               'nan_in_output': bool(np.isnan(og).any()),
                               'peak_corrected': round(float(np.max(np.abs(og))), 4)})
        print(f"[apply] {seg}: 修 {len(fixes)}/{len(a['notes'])} 音；拼接区外最大差异 {max_diff_outside:.2e}")

    fields = list(note_rows[0].keys())
    with open(RESULTS / 'apply_notes.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(note_rows)

    hashes_after = {p.name: sha256(p) for p in src_files}
    qa['source_hashes_unchanged'] = hashes == hashes_after
    if not qa['source_hashes_unchanged']:
        raise SystemExit('源文件哈希变化，终止！')
    qa['engineering_pass'] = all(
        s['samples'] == len(read_wav_pcm16(R.seg_path(s['segment']))[0])
        and not s['nan_in_output'] and s['peak_corrected'] < 0.999
        and s['max_sample_diff_outside_splice'] < 1e-9
        for s in qa['segments'])
    (RESULTS / 'qa_report.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(f"[apply] 工程检查 {'通过' if qa['engineering_pass'] else '未通过'}；听感 pending_jade")


def cmd_page():
    qa = json.loads((RESULTS / 'qa_report.json').read_text(encoding='utf-8'))
    notes = list(csv.DictReader(open(RESULTS / 'apply_notes.csv', encoding='utf-8')))
    import io as _io
    import base64
    import soundfile as sf

    def b64(p):
        import subprocess
        # 预览/浏览器打不开几 MB 的内嵌 WAV：转 96k 单声道 mp3 内嵌（可听度足够判断音准）
        proc = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(p),
                               '-ac', '1', '-codec:a', 'libmp3lame', '-b:a', '96k',
                               '-f', 'mp3', 'pipe:1'], capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            raise SystemExit(f'mp3 转码失败: {p}\n{proc.stderr.decode()[:300]}')
        return base64.b64encode(proc.stdout).decode('ascii')

    cards = []
    for s in qa['segments']:
        stem = s['segment']
        seg_notes = [n for n in notes if n['segment'] == stem]
        fix_list = ''.join(
            f"<tr><td>{float(n['t0']):.2f}–{float(n['t1']):.2f}</td><td>{n['note']}</td>"
            f"<td>{float(n['dev']):+.0f}</td><td>{float(n['model_shift']):+.0f}</td></tr>"
            for n in seg_notes if n['status'] == 'fix') or '<tr><td colspan="4">无</td></tr>'
        reason = {'ref_unstable': '原唱这段在滑音，目标不稳', 'too_large': '偏得太多，怕修错',
                  'too_short': '音太短，修了听不出还容易出伪影', 'ok': '偏差在门槛内'}
        susp = ''.join(
            f"<tr><td>{float(n['t0']):.2f}</td><td>{n['note']}</td><td>{float(n['dev']):+.0f}</td>"
            f"<td>{reason.get(n['status'], n['status'])}</td></tr>"
            for n in seg_notes if n['status'] != 'fix' and abs(float(n['dev'])) >= 60) or \
            '<tr><td colspan="4">无</td></tr>'
        cards.append(f'''
  <div class="card">
    <h3>{stem} <span class="tag">{s['role']}</span></h3>
    <p class="meta">修了 {s['fixed']} 处（Jade 标注优先）；对齐参考区间 {s['ref_interval_s'][0]}–{s['ref_interval_s'][1]} 秒</p>
    <div class="row">
      <div class="col"><p>① 原音·同样响（参照）</p><audio controls preload="none" src="data:audio/mpeg;base64,{b64(APPLY / f'{stem}_原音同响.wav')}"></audio></div>
      <div class="col"><p>② 修音版</p><audio controls preload="none" src="data:audio/mpeg;base64,{b64(APPLY / f'{stem}_修正版.wav')}"></audio></div>
    </div>
    <details><summary>修了哪些音（时刻 / 音名 / 原来偏了多少 / 修了多少，单位音分）</summary>
      <table><tr><th>时刻(秒)</th><th>音名</th><th>偏差</th><th>修正</th></tr>{fix_list}</table>
    </details>
    <details><summary>疑似跑调但按"拿不准不修"跳过的位置（P2 需要你标注）</summary>
      <table><tr><th>时刻(秒)</th><th>音名</th><th>偏差</th><th>跳过原因</th></tr>{susp}</table>
    </details>
    <p>② 和 ① 比：</p>
    <p>长音准不准：<label><input type="radio" name="{stem}_acc" value="更准">更准</label>
      <label><input type="radio" name="{stem}_acc" value="差不多">差不多</label>
      <label><input type="radio" name="{stem}_acc" value="更差">更差</label>
      <label><input type="radio" name="{stem}_acc" value="没听出差别">没听出差别</label></p>
    <p>自不自然：<label><input type="radio" name="{stem}_nat" value="自然">自然</label>
      <label><input type="radio" name="{stem}_nat" value="有点怪">有点怪</label>
      <label><input type="radio" name="{stem}_nat" value="明显电音">明显电音</label></p>
    <p>像不像本人：<label><input type="radio" name="{stem}_id" value="像">像</label>
      <label><input type="radio" name="{stem}_id" value="不太像">不太像</label>
      <label><input type="radio" name="{stem}_id" value="不像">不像</label></p>
    <input type="text" class="note" id="{stem}_note" placeholder="补充（可不填）：比如哪一秒怪">
  </div>''')

    segs_js = json.dumps([s['segment'] for s in qa['segments']], ensure_ascii=False)
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F2 第一轮 · 修音试听（莫问归期）</title>
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
  table {{ border-collapse: collapse; font-size: 13px; margin-top: 6px; }}
  td, th {{ border: 1px solid #ddd; padding: 3px 8px; }}
  .note {{ width: 100%; margin-top: 6px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; box-sizing: border-box; }}
  button {{ padding: 10px 22px; font-size: 15px; cursor: pointer; }}
  textarea {{ width: 100%; height: 160px; margin-top: 10px; font-size: 13px; }}
  details summary {{ cursor: pointer; font-size: 13px; color: #555; }}
  .tag {{ font-size: 12px; color: #fff; background: #6b7280; border-radius: 4px; padding: 1px 6px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>F2 第一轮 · 修音试听（带你的逐段标注 · v3）</h1>
<div class="head">
  <b>这版改了什么</b>：把你 11-06 标的 17 段"希望修 / 故意这样唱"直接接进了程序——标"故意"的音一个不动，标"希望修"的强制修到参考音；其余按放开的阈值大胆修。<br>
  <b>重点听</b>：离开01 的"当你"（约 8.0–10.2 秒，你标了希望修的长音）和心墙01 的"太多"（约 4.3–5.2 秒）——这两处是最可能听出"修准了"的。<br>
  <b>莫问归期两段</b>：你标的全是"故意这样唱"，所以这版几乎不动它——<b>几乎不动才是对的</b>（上一版乱修就是你标注要防的事）。<br>
  <b>声明</b>：离开01/心墙01 是模型学过的段（训练内验证）；莫问两段是它没学过的。Agent 不能听音，一切以你为准。
</div>
{''.join(cards)}
<div style="text-align:center; margin: 20px 0;"><button onclick="collect()">生成答案</button></div>
<textarea id="out" readonly placeholder="点"生成答案"后，全选复制发回即可"></textarea>
<script>
const SEGS = {segs_js};
function collect() {{
  const lines = ['【F2 第一轮试听反馈 2026-11-07】'];
  for (const s of SEGS) {{
    const g = n => {{ const v = document.querySelector('input[name="' + s + '_' + n + '"]:checked'); return v ? v.value : '未答'; }};
    const note = document.getElementById(s + '_note').value.trim();
    lines.push(s + '｜长音：' + g('acc') + '｜自然：' + g('nat') + '｜像本人：' + g('id') + (note ? '｜补充：' + note : ''));
  }}
  document.getElementById('out').value = lines.join('\\n');
  document.getElementById('out').select();
}}
</script>
</body>
</html>'''
    if (DEST / 'index.html').exists():
        raise SystemExit('index.html 已存在，拒绝覆盖')
    (DEST / 'index.html').write_text(html, encoding='utf-8')
    print(f"[page] {DEST / 'index.html'}")


if __name__ == '__main__':
    {'train': cmd_train, 'apply': cmd_apply, 'page': cmd_page}[sys.argv[1]]()
