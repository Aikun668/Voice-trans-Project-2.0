"""Run fixed v0.5 candidates on two Jade recordings and one GTSinger example."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from f1_poc import experiment, export_wav, read_pcm16, run_ffmpeg


def build(out, dataset="normal"):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Use a NEW output directory")
    sources = [
        ("shige", "本人 · 是谁在唱歌", Path("test_dataset/my_singing/shige_normal.wav")),
        ("zuixuan", "本人 · 最炫民族风", Path("test_dataset/my_singing/zuixuan_normal.wav")),
        ("web_gaobai", "网络 · 告白气球（女中音）", Path("test_dataset/web_singing/gtsinger_alto_control/告白气球_normal.wav")),
    ]
    if dataset == "offpitch":
        sources = [
            ("shige_medium", "本人 · 是谁在唱歌 · 中度跑调", Path("test_dataset/my_singing/shige_medium.wav")),
            ("shige_severe", "本人 · 是谁在唱歌 · 严重跑调", Path("test_dataset/my_singing/shige_severe.wav")),
            ("web_shinian", "网络 · 十年（女中音，新片段）", Path("test_dataset/web_singing/gtsinger_alto_control/十年_normal.wav")),
        ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for _, _, p in sources}
    out.mkdir(parents=True)
    manifest, cards = [], []
    for key, title, original in sources:
        source = original
        converted = key.startswith("web")
        if converted:
            source = out / f"{key}_input_pcm16.wav"
            run_ffmpeg(["-v", "error", "-f", "wav", "-i", str(original),
                        "-af", "aresample=44100:dither_method=triangular", "-ac", "1",
                        "-c:a", "pcm_s16le", "-n", str(source)])
        report = experiment(source, out / key, profile="v5_transfer")
        entry = {"id": key, "title": title, "original_source": str(original.resolve()),
                 "original_sha256": hashes[str(original)], "converted_copy": converted,
                 "technical_pass": report["technical_pass"], "errors": report["errors"],
                 "subjective_result": "pending_jade"}
        entry["dataset"] = dataset
        entry["profile"] = "v5_transfer"
        manifest.append(entry)
        if not report["technical_pass"]:
            cards.append(f'<section><h2>{html.escape(title)}</h2><p>技术检查未通过，本条暂不交付试听。</p></section>')
            continue
        clips = []
        for item, label in zip(report["outputs"], ["01 原音（仅匹配音量）", "02 明显处理", "03 强处理"]):
            name = item["file"]
            clips.append(f'<p>{label} · {item["loudness"]["integrated_lufs"]:.2f} LUFS</p>'
                         f'<audio controls preload="metadata" data-group="{key}" src="{key}/{name}"></audio>')
        reference, sr = read_pcm16(source)
        # Choose the most energetic 3-second phrase from the source only.
        starts = range(0, max(1, len(reference) - 3 * sr + 1), sr // 4)
        start = max(starts, key=lambda i: np.mean(reference[i:i + 3*sr] ** 2))
        tracks = [read_pcm16(out / key / item["file"])[0][start:start + 3*sr] for item in report["outputs"]]
        gap = np.zeros(round(sr*.35))
        montage = np.concatenate([tracks[0], gap, tracks[1], gap, tracks[2]])
        export_wav(montage, sr, 0, out / key / "04_same_phrase.wav")
        entry["phrase_seconds"] = [start / sr, start / sr + 3]
        entry["duration_seconds"] = report["input_samples"]["duration_seconds"]
        note = ("来源：GTSinger 女中音。原始 48 kHz / 24-bit 保留，测试副本转换为 44.1 kHz / 16-bit；只与同一人的转换后原音比较。"
                "音高诊断沿用男声 60–500 Hz 范围，不能据此确认女声音高完整保留。" if converted else "本人手机录音，保留完整时长和原始素材。")
        cards.append(f'<section id="{key}"><h2>{html.escape(title)}</h2><p class="note">{note}</p>'
                     '<h3>先听同一句：原音 → 明显处理 → 强处理</h3>'
                     f'<p>取原音 {start/sr:.2f}–{start/sr+3:.2f} 秒；每段 3 秒，间隔 0.35 秒。</p>'
                     f'<audio controls preload="metadata" data-montage="true" src="{key}/04_same_phrase.wav"></audio>'
                     '<h3>完整录音</h3>' + ''.join(clips) +
                     f'<p><a href="{key}/qa_report.json">本条技术检查数据</a></p></section>')
        (out / key / "试听说明.md").write_text(
            title + "\n\n" + note + "\n\n固定使用 v0.5 两档，不按本素材重新调参。技术检查通过，听感待 Jade。\n"
            "短句对照为原音→明显处理→强处理；完整输出不裁剪。不移调、无混响、无降噪。\n"
            "请反馈更喜欢哪档、有无发闷/变尖/变薄、是否比茉莉花更容易听出变化。\n", encoding="utf-8")
    for _, _, p in sources:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)], "Original source changed"
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,"><title>Jade · F1 换素材验证 01</title>
<style>body{font:16px/1.7 system-ui;background:#f4f5f7;color:#202530;max-width:800px;margin:32px auto;padding:0 20px}section{background:white;border:1px solid #dce0e6;border-radius:12px;padding:24px;margin:24px 0}audio{width:100%}h1{font-size:26px}h2{font-size:22px}h3{font-size:17px}.note{color:#596579}a{color:#2953a6}</style>
<h1>F1 换素材验证 · 固定 v0.5 参数</h1>
<p>先听自己的两首，再听网络女声。各首内部已匹配响度；不把不同人的音色相互打分。</p>
<p>本轮只换素材，未加强参数，也不修音高。先比较：哪一首更容易听出改善？更喜欢原音、明显处理还是强处理？无区别也是有效结果。</p>
<p>技术检查通过不代表更好听；主观试听由 Jade 验收。</p>
''' + ''.join(cards) + '''<script>
let active=null;
document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{
 if(active && active!==p){
  const t=active.ended||active.dataset.montage||p.dataset.montage||active.dataset.group!==p.dataset.group?0:active.currentTime;
  active.pause();p.currentTime=Math.min(t,p.duration||t);
 }active=p;
}));</script></html>'''
    if dataset == "offpitch":
        page = page.replace("换素材验证 01", "跑调素材验证 01")
        page = page.replace("F1 换素材验证 · 固定 v0.5 参数", "F1 中度／严重跑调素材 · 固定 v0.5 参数")
        page = page.replace("先听自己的两首，再听网络女声。", "先听本人同一首歌的中度、严重跑调版，再听新换的网络女声《十年》。")
        page = page.replace("本轮只换素材，未加强参数，也不修音高。", "本轮只换素材，不改 EQ、压缩阈值或其他参数，也不修音高。请评价声音质感是否改善，跑调本身仍会保留。")
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True))
    return all(item["technical_pass"] for item in manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=["normal", "offpitch"], default="normal")
    args = parser.parse_args()
    raise SystemExit(0 if build(args.out, dataset=args.dataset) else 1)
