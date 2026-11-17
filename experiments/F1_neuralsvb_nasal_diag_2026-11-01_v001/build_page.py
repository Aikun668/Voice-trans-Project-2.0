"""Build a static, 8-song comparison page for the nasal diagnosis (no JS, no selector).

Layout follows F1_neuralsvb_8songs_2026-11-01_v001/listening/index.html:
one card per recording, one player row per version, candidate times in the card meta.
Read-only: audio and metrics are not modified.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "F1_neuralsvb_nasal_diag_2026-11-01_v001"
METRICS = json.loads((EXP / "run-003" / "metrics.json").read_text(encoding="utf-8"))
SRC_REPORT = json.loads((ROOT / "experiments" / "F1_neuralsvb_8songs_2026-11-01_v001" / "report.json")
                        .read_text(encoding="utf-8"))
PAGE = EXP / "listening" / "index.html"
if PAGE.exists():
    PAGE.unlink()  # 页面可从 run-003 重建；指标与音频不改

REL_AUDIO = "../../F1_neuralsvb_8songs_2026-11-01_v001/listening/audio"
THIN = {"A01", "A03", "A05", "A07"}
DUR = {s["sample_id"]: s["input_stats"]["duration_seconds"] for s in SRC_REPORT["samples"]}


def esc(value):
    return html.escape(str(value))


def select_segments(r):
    ws = r["windows"]
    ranked = sorted(ws, key=lambda w: w["a2p"]["mid_emphasis_db"], reverse=True)
    picked = []
    for w in ranked:
        if all(abs(w["start_s"] - p["start_s"]) >= 1.0 for p in picked):
            picked.append(w)
        if len(picked) == 3:
            break
    for w in sorted(ws, key=lambda w: w["a2p"]["mid_emphasis_db"]):
        if all(abs(w["start_s"] - p["start_s"]) >= 1.0 for p in picked):
            picked.append(w)
            break
    labels = ["候选 1（最强）", "候选 2", "候选 3", "弱对照（最低）"]
    return [{
        "label": labels[i] if i < len(labels) else f"候选 {i + 1}",
        "start": w["start_s"], "end": w["end_s"],
        "score": w["a2p"]["mid_emphasis_db"],
    } for i, w in enumerate(picked)]


def row(label, src):
    return (f'<div class="row"><label>{esc(label)}</label>'
            f'<audio controls preload="metadata" src="{esc(src)}"></audio></div>')


DATA = []
summary_rows = []
for r in METRICS["results"]:
    sid = r["sample_id"]
    d = r["deltas"]["a2p_vs_original"]
    v = r["deltas"]["vocoder_vs_original"]
    item = {
        "id": sid, "song": r["song"], "variant": r["variant"],
        "group": "thin" if sid in THIN else "normal",
        "normal": r["normal_reference"], "duration": DUR.get(sid),
        "segments": select_segments(r),
        "delta": d, "vocoder_delta": v,
        "mid_emphasis": r["mid_emphasis_a2p"],
        "mid_emphasis_vocoder": r["mid_emphasis_vocoder"],
        "window": r["window_score_stats"],
        "eq": r["colour_match_estimate"],
    }
    DATA.append(item)
    top = r["top_mid_emphasis_windows"][0]
    summary_rows.append(
        "<tr>"
        f"<td>{esc(sid)}</td><td>{esc(r['song'])}·{esc(r['variant'])}</td>"
        f"<td>{'薄音' if sid in THIN else '正常'}</td>"
        f"<td>{d['low_0_500']:+.1f}</td><td>{d['mid1_500_1500']:+.1f}</td><td>{d['high_4000_8000']:+.1f}</td>"
        f"<td>{r['mid_emphasis_a2p']:+.1f}</td><td>{r['mid_emphasis_vocoder']:+.1f}</td>"
        f"<td>{top['start_s']:.1f}–{top['end_s']:.1f}s</td>"
        "</tr>")


def card(item):
    sid = item["id"]
    is_thin = item["group"] == "thin"
    orig_label = "薄音原音（48kHz）" if is_thin else "正常原音（48kHz）"
    seg_text = "；".join(
        f"{s['label']} {s['start']:.1f}–{s['end']:.1f}s（{s['score']:+.1f}dB）"
        for s in item["segments"])
    st = item["window"]
    meta = (
        f"{sid} {item['song']}·{item['variant']} · {item['duration']:.2f}s<br>"
        f"a2p 相对原音：低频 {item['delta']['low_0_500']:+.1f}dB · "
        f"0.5–1.5k {item['delta']['mid1_500_1500']:+.1f}dB · "
        f"4–8k {item['delta']['high_4000_8000']:+.1f}dB · "
        f"中频强调 {item['mid_emphasis']:+.1f}dB"
        f"（2s 窗 {st['mean']:+.1f}±{st['std']:.1f}）<br>"
        f"重点听：{seg_text}")
    rows = row(orig_label, f"{REL_AUDIO}/{sid}_original.wav")
    rows += row("NeuralSVB a2p（22.05kHz）", f"{REL_AUDIO}/{sid}_a2p.wav")
    rows += row("仅声码器对照（22.05kHz）", f"{REL_AUDIO}/{sid}_vocoder_only.wav")
    if is_thin and item["normal"]:
        rows += row(f"同曲正常版参考（{item['normal']} 原音）",
                    f"{REL_AUDIO}/{item['normal']}_original.wav")
    return (f'<div class="card"><h3>{esc(item["song"])}（{"薄音版" if is_thin else "正常版"}）</h3>'
            f'<p class="meta">{meta}</p>{rows}</div>')


cards_thin = "".join(card(x) for x in DATA if x["group"] == "thin")
cards_normal = "".join(card(x) for x in DATA if x["group"] == "normal")

page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuralSVB a2p 鼻音定位 · 8 条对照（只读）</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}h2{{font-size:20px;margin-top:26px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0;font-size:14px}}.row{{display:grid;grid-template-columns:215px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #dce0e6;padding:6px 8px;text-align:right}}th{{background:#eef1f5;text-align:center}}td:first-child,td:nth-child(2){{text-align:left}}</style>
<h1>NeuralSVB a2p 鼻音定位 · 8 条对照（只读）</h1>
<div class="note"><b>本页只做定位，不改音频，也没有加 EQ。</b>每张卡片直接列出 原音 / a2p / 仅声码器（薄音卡片再加同曲正常版参考），和上一版 8 条页的对照方式一致。<br>
<b>测量结论：</b>8 条 a2p 都有“低频和 4–8kHz 相对减弱、0.5–1.5kHz 相对抬高”的倾斜；仅声码器对照在 ±1dB 内。所以来源是声学模型/映射层，不是声码器，也不是单纯 22.05kHz 带宽。<br>
<b>怎么听：</b>同一张卡片先听原音，再听 a2p；如果怀疑是声码器，切到仅声码器；薄音卡片最后听同曲正常版参考。固定播放器音量。每张卡片都写了“重点听”的候选时间段，先听那里。<br>
<b>已知：</b>这批文件沿用上一轮工程结果（0/16 通过：音高诊断/动态跌落）；本轮不重跑模型、不改阈值、不动原素材。Agent 无听音能力，下面的时间点只是测量挑出的候选，最终以你的听感为准。</div>
<h2>一、薄音版（4 条）</h2>{cards_thin}
<h2>二、正常版（4 条）</h2>{cards_normal}
<h2>三、测量总览</h2>
<table><tr><th>文件</th><th>素材</th><th>组</th><th>a2p 低频 Δ/dB</th><th>a2p 0.5–1.5k Δ/dB</th><th>a2p 4–8k Δ/dB</th><th>a2p 中频强调/dB</th><th>仅声码器中频强调/dB</th><th>最强 2s 窗</th></tr>{''.join(summary_rows)}</table>
<p class="meta">Δ 是相对原音、以 1.5–4kHz 为基准的长期平均频谱差；中频强调 = −(低频Δ + 高频Δ)/2。正数表达“低频/高频相对变弱”。这些是描述性测量，不是“更难听”的判定。</p>
<h2>四、如果确认了位置，下一步怎么调</h2>
<div class="note">既有 <b>colour_match</b> 的校准估计：低频 +1.3 至 +4.4 dB、4–8k +2.7 至 +7.4 dB、中频 −1.8 至 +1.3 dB（zero-phase EQ，可以直接作用在已生成的 a2p 上，不用重跑模型）。<br>
风险：它会把模型抬高的中频一起削弱，可能伤到你觉得“更饱满”的部分；而且 A07 的倾斜随时间变化最大（−3.0 至 +6.7 dB），固定 EQ 可能过校正。<br>
所以下一步建议只在你确认的那一条上做 <b>100% / 50% 两个校准版本 A/B</b>，其他 7 条不动、权重不改、工程阈值不放宽。先确认“哪首、哪几秒、哪个版本开始有鼻音”，再决定是否值得做这一轮校准。</div>
<p class="meta">原始素材：test_dataset/f1_dataset_v2/A_thin/self/wav（哈希校验，未改动）。详细测量：run-003/metrics.json。</p></html>'''

PAGE.parent.mkdir(parents=True, exist_ok=True)
PAGE.write_text(page, encoding="utf-8")
print(PAGE)
