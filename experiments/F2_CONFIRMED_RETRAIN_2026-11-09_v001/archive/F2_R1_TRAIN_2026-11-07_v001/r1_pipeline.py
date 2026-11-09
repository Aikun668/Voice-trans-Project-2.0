"""F2 第一轮有参考修音 · 训练/应用/试听页 共用分析代码。

复用可信模块（不重写 DSP）：
- poc_phase4_realign.cents_track / cost_matrix / subseq_dtw ：F0 提取与子序列 DTW 对齐
- self_ref_align.notes_of / note_name ：稳定音分段
- poc_phase3b.apply_tapered ：PSOLA 平移（带渐变）
- poc_phase4b.splice ：只在修正区间使用重合成，其余原样保留

标签现状（诚实记录）：训练标签由守护规则从"自动候选对齐"的偏差推导，
尚无 Jade 的"故意唱法"标注；等 P2 标注页确认后标签源可替换，流程不变。
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'f2'))
sys.path.append(str(ROOT / '.venv-f2/Lib/site-packages'))
import parselmouth  # noqa: E402  (来自 .venv-f2)

from poc_phase4_realign import cents_track, cost_matrix, subseq_dtw, HOP  # noqa: E402
from self_ref_align import notes_of, note_name  # noqa: E402
from poc_phase3b import apply_tapered  # noqa: E402
from poc_phase4b import splice  # noqa: E402

BASE = ROOT / 'test_dataset/my_singing/rerecord_2026-11-04_f2'
REFS = {
    '心墙': '心墙_郭静_vocals.wav',
    '离开我的依赖': '离开我的依赖_王艳薇_vocals.wav',
    '莫问归期': '莫问归期_蒋雪儿_vocals.wav',
}
TRAIN_SEGS = ['心墙01-对照微微跑调', '心墙01-轻度跑调',
              '离开我的依赖01-对照微微跑调', '离开我的依赖01-轻度跑调',
              '离开我的依赖02-对照微微跑调', '离开我的依赖02-轻度跑调']
TEST_SEGS = ['莫问归期01-轻度跑调', '莫问归期02-轻度跑调']


def seg_song_take(seg):
    """'离开我的依赖01-轻度跑调' → ('离开我的依赖', '01')。同段两版共享 take 号。"""
    stem = seg.split('-')[0]
    for song in REFS:
        if stem.startswith(song):
            return song, stem[len(song):]
    raise ValueError(seg)

# 守护规则（沿用既有约定；≠拍脑袋）
FIX_MIN_DEV = 60.0     # 偏差小于此不修（realign 同款阈值）
FIX_MAX_ABS = 150.0    # 超过此标记 too_large，交 Jade，不自动硬修
MIN_FIX_DUR = 0.25     # 短于此的音不修（phase4c 约定：短音修了也听不出）
REF_MAX_SPREAD = 120.0  # 参考对应段离散度过大视为滑音段，不作为目标
VOICED_MIN = 0.6       # 参考段浊音占比不足不配对
FEATURES = ['dur', 'sung_cents', 'ref_c_global', 'ref_spread', 'ref_voiced', 't0']


def cents_of(y, sr):
    """波形数组 → 音分轨迹（0.02s 步长，与 cents_track 一致）。"""
    hz = parselmouth.Sound(y, sr).to_pitch(
        time_step=HOP, pitch_floor=75.0, pitch_ceiling=600.0).selected_array['frequency']
    c = np.full(len(hz), np.nan)
    v = hz > 0
    c[v] = 1200 * np.log2(hz[v] / 440.0)
    return c


def circ(v):
    return float((v + 600.0) % 1200.0 - 600.0)


def seg_path(seg):
    return BASE / 'wav' / f'{seg}.wav'


def ref_path(song):
    return BASE / 'ref_vocals' / REFS[song]


def analyze(target, reference, anchor=None):
    """对齐并返回逐音偏差表。target/reference 均为文件路径。

    anchor = (t0, t1, key)：P1 候选区间与粗调差。旋律会重复，全局细搜索会
    错配到别的段落（2026-11-07 实测《心墙》两段均错配），因此细对齐只在
    候选区间 ±3 秒、粗调差 ±150 音分内搜索；候选被 Jade 否决时换次优重跑。
    """
    q = cents_track(str(target))
    r_full = cents_track(str(reference))
    key0 = None
    if anchor is None:
        r = r_full
    else:
        a0, a1, k0 = anchor
        i0, i1 = max(int((a0 - 3.0) / HOP), 0), min(int(np.ceil((a1 + 3.0) / HOP)), len(r_full))
        r = r_full[i0:i1]
    best = None
    keys = range(-600, 600, 10) if key0 is None else range(int(k0) - 150, int(k0) + 151, 10)
    for k in keys:
        cost, path = subseq_dtw(cost_matrix(q, r, float(k)))
        if best is None or cost < best[0]:
            best = (cost, path, float(k))
    cost, path, key = best
    j_off = 0 if anchor is None else i0
    qmap = {}
    for i, j in path:
        qmap.setdefault(i, []).append(j + j_off)

    rows0 = []
    for n in notes_of(str(target)):
        i0 = int(round(n['t0'] / HOP))
        i1 = min(int(round(n['t1'] / HOP)), len(q) - 1)
        js = sorted({j for i in range(i0, i1 + 1) for j in qmap.get(i, [])})
        rc = np.asarray(r_full[js], dtype=float) if js else np.array([])
        rc = rc[~np.isnan(rc)]
        voiced = len(rc) / max(len(js), 1)
        if len(rc) < 3 or voiced < VOICED_MIN:
            continue
        seed = float(np.median(rc))       # 参考轨八度误判防护：按 1200 音分折叠
        rcf = seed + (rc - seed + 600.0) % 1200.0 - 600.0
        rows0.append({**n, 'ref_c': float(np.median(rcf)), 'ref_voiced': round(voiced, 2),
                      'ref_spread': float(np.percentile(rcf, 90) - np.percentile(rcf, 10)),
                      'map_t0': js[0] * HOP, 'map_t1': js[-1] * HOP})

    stable = [x for x in rows0 if x['ref_spread'] <= REF_MAX_SPREAD]
    if stable:                            # 稳定音二次细化调差（realign 同款）
        key += float(np.median([circ(x['cents'] - x['ref_c'] - key) for x in stable]))

    notes = []
    for x in rows0:
        dev = circ(x['cents'] - x['ref_c'] - key)
        notes.append({**x, 'dev': round(dev, 1), 'key': round(key, 1),
                      'ref_c_global': x['ref_c'] + key,
                      'note': note_name(x['cents']),
                      'ref_note': note_name(x['ref_c'] + key),
                      'map_t0': round(x['map_t0'], 2), 'map_t1': round(x['map_t1'], 2),
                      'dur': round(x['t1'] - x['t0'], 3)})
    return {'cost_per_frame': round(cost, 1), 'key_cents': round(key, 1),
            'ref_t0': round((path[0][1] + j_off) * HOP, 2),
            'ref_t1': round((path[-1][1] + j_off) * HOP, 2),
            'notes': notes}


def rule_of(n):
    """守护规则 →（修正量, 状态）。训练标签与上线守护共用同一套。"""
    if n['ref_spread'] > REF_MAX_SPREAD:
        return 0.0, 'ref_unstable'
    if n.get('ref_voiced', 0) < VOICED_MIN:
        return 0.0, 'ref_unvoiced'
    if n['dur'] < MIN_FIX_DUR:
        return 0.0, 'too_short'
    if abs(n['dev']) < FIX_MIN_DEV:
        return 0.0, 'ok'
    if abs(n['dev']) > FIX_MAX_ABS:
        return 0.0, 'too_large'
    return round(-n['dev'], 1), 'fix'


def feature_row(n):
    return [n['dur'], n['cents'], n['ref_c_global'], n['ref_spread'], n['ref_voiced'], n['t0']]


def corrected_audio(x, sr, fixes):
    """fixes=[(t0,t1,shift)] →（修正后波形, 拼接掩码）。只在拼接区用重合成。"""
    y = apply_tapered(parselmouth.Sound(x, sr), fixes)[: len(x)]
    out, w = splice(x, y, sr, [(a, b, s) for a, b, s in fixes])
    return out, w


def gain_match(x, *others):
    """与 realign 同款：整体 -16 dB RMS，峰值不超 -3 dBFS。返回统一增益。"""
    from poc_phase1 import peak_db, rms_db
    g = 10 ** ((-16.0 - rms_db(x)) / 20.0)
    peak = max(peak_db(x), max((peak_db(o) for o in others), default=-300))
    return min(g, 10 ** ((-3.0 - peak) / 20.0))


def ebur128(path):
    """FFmpeg ebur128 实测响度/真峰值（不信任内部计算，ROADMAP §8.2）。"""
    import re
    import subprocess
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path),
                        '-filter_complex', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True)
    i_vals = re.findall(r'I:\s*(-?\d+\.?\d*)\s*LUFS', p.stderr)
    tp_vals = re.findall(r'(-?\d+\.?\d*)\s*dBTP', p.stderr)
    if not tp_vals:  # 本机 ffmpeg 摘要只给 dBFS；逐帧 TPK 即 4 倍过采样真峰值
        tp_lines = re.findall(r'TPK:\s*(-?\d+\.?\d*)\s*dBFS', p.stderr)
        tp_vals = tp_lines
    if not i_vals or not tp_vals:
        raise SystemExit(f'ebur128 解析失败: {path}\n{p.stderr[-800:]}')
    return float(i_vals[-1]), float(tp_vals[-1])
