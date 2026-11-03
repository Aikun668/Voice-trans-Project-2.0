"""NeuralSVB 官方权重 · 8 条本人录音批量试听（4 薄音 / 4 同曲正常版）。

复用（不修改）：
- neuralsvb_infer.py：官方 1030_vae_mle 声学模型 + 1009 PPG/ASR + 1012 HiFi-GAN-NSF 声码器，
  a2p 美化，F0 保持输入自身曲线（不改音高/节奏）。
- f1_poc.py / run_poc_batch.py::check_output：项目既有工程检查，阈值不放宽。
不训练、不微调、不加 DSP。输出为模型原生 22.05 kHz mono PCM16；
所有试听文件统一到一个全局响度目标（常数增益，无限制器）。
听感状态始终 pending_jade：测量变化不等于变好听。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# f1_poc 的响度/导出依赖 ffmpeg；WinGet 安装目录通常不在 PATH 中。
if not shutil.which("ffmpeg"):
    links = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if (links / "ffmpeg.exe").exists():
        os.environ["PATH"] = f"{links}{os.pathsep}{os.environ['PATH']}"
if not shutil.which("ffmpeg"):
    raise SystemExit("ffmpeg.exe not found; add its directory to PATH.")

import f1_poc as qa  # noqa: E402
import neuralsvb_infer as nsvb  # noqa: E402
from run_poc_batch import check_output, common_target, sha256  # noqa: E402

DATA = ROOT / "test_dataset" / "f1_dataset_v2"
VARIANTS = ("a2p", "vocoder_only")
MODE_OF = {"a2p": "a2p", "vocoder_only": "vocoder"}
PAIRS = [
    {"song": "平凡之路", "thin": "A_self_01", "normal": "A_self_02"},
    {"song": "我们的歌", "thin": "A_self_03", "normal": "A_self_04"},
    {"song": "雅俗共赏", "thin": "A_self_05", "normal": "A_self_06"},
    {"song": "因为刚好遇见你", "thin": "A_self_07", "normal": "A_self_08"},
]
ALL_IDS = [x for p in PAIRS for x in (p["thin"], p["normal"])]
VARIANT_NAMES = {"a2p": "NeuralSVB a2p", "vocoder_only": "声码器对照（非候选）"}

# ---- 谐波描述（与 experiments/F1-EXP-011/run.py 同一测量；仅供描述，不作通过判定）----
FMIN, FMAX = 90.0, 450.0
ANA_WIN_S, ANA_HOP_S = 0.060, 0.020
R_GATE, FLOOR_DB = 0.35, 40.0
K_PROFILE = 12


def frame_f0_ac(win, sr):
    w = np.hanning(len(win)) * (win - win.mean())
    if not np.any(w):
        return float("nan"), 0.0
    c = signal.correlate(w, w, method="fft")[len(w) - 1:]
    if c[0] <= 0:
        return float("nan"), 0.0
    c = c / c[0]
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    if hi >= len(c):
        return float("nan"), 0.0
    k = int(np.argmax(c[lo:hi + 1])) + lo
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        k_f, r = k + d, float(y1)
    else:
        k_f, r = float(k), float(c[k])
    f0 = sr / k_f
    return (float(f0), float(r)) if FMIN <= f0 <= FMAX else (float("nan"), float(r))


def harmonic_profile(x, sr):
    win, hop = int(round(sr * ANA_WIN_S)), int(round(sr * ANA_HOP_S))
    if len(x) < win:
        raise RuntimeError("too short for profile")
    starts = list(range(0, len(x) - win + 1, hop))
    rms_db = np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x[s:s + win] ** 2))), 1e-12))
                       for s in starts])
    gate = rms_db.max() - FLOOR_DB
    rel = {k: [] for k in range(1, K_PROFILE + 1)}
    n_voiced, nfft = 0, 1 << 15
    for s, rdb in zip(starts, rms_db):
        if rdb < gate:
            continue
        seg = x[s:s + win]
        f0, r = frame_f0_ac(seg, sr)
        if not np.isfinite(f0) or r < R_GATE:
            continue
        n_voiced += 1
        w = np.hanning(win) * seg
        spec = np.fft.rfft(w, nfft)
        gain = np.sum(np.hanning(win))
        a1 = 20.0 * np.log10(max(2.0 * abs(spec[int(round(f0 * nfft / sr))]) / gain, 1e-10))
        for k in range(1, K_PROFILE + 1):
            f = k * f0
            if f > 8000.0:
                break
            idx = int(round(f * nfft / sr))
            if idx >= len(spec):
                break
            ak = 20.0 * np.log10(max(2.0 * abs(spec[idx]) / gain, 1e-10))
            rel[k].append(ak - a1)
    if n_voiced < 50:
        raise RuntimeError(f"voiced frames too few: {n_voiced}")
    return {k: float(np.median(v)) for k, v in rel.items() if len(v) >= 30}, n_voiced


def band_fraction_2_4k(x, sr):
    f, p = signal.welch(x, sr, nperseg=min(4096, len(x)))
    total = float(np.sum(p[(f >= 20) & (f < 8000)]))
    band = float(np.sum(p[(f >= 2000) & (f < 4000)]))
    return band / max(total, 1e-20)


def safe_metrics(x, sr):
    try:
        P, n_voiced = harmonic_profile(x, sr)
        rich = [10 ** (P[k] / 10) for k in range(2, K_PROFILE + 1) if k in P]
        return {
            "h2_minus_h1_db": round(P.get(2, float("nan")), 2) if 2 in P else None,
            "richness_k2_12_db": round(10.0 * np.log10(max(sum(rich), 1e-10)), 2) if rich else None,
            "presence_2_4k_fraction": round(band_fraction_2_4k(x, sr), 4),
            "voiced_frames": n_voiced,
        }
    except Exception as exc:  # 描述性测量，失败只记录，不影响试听文件
        return {"error": str(exc)}


def load_clip(did, item):
    wav_path = DATA / item["wav"]["path"]
    if sha256(wav_path) != item["wav"]["sha256"]:
        raise ValueError(f"input WAV changed: {wav_path}")
    x, sr = qa.read_pcm16(wav_path)
    if len(x) != item["wav"]["frames"]:
        raise ValueError(f"frame mismatch: {dict(id=did, expected=item['wav']['frames'], got=len(x))}")
    return x, sr, wav_path


def sid_of(did):
    return did.replace("A_self_", "A")


def esc(value):
    return html.escape(str(value))


def variant_label(name, processed=True):
    if name == "original":
        return "原音"
    base = VARIANT_NAMES[name]
    return f"{base}（22.05kHz）" if processed else base


def row(label, src, note=""):
    extra = f'<br><span style="font-size:12px">{esc(note)}</span>' if note else ""
    return (f'<div class="row"><label>{esc(label)}{extra}</label>'
            f'<audio controls preload="metadata" src="{esc(src)}"></audio></div>')


def metrics_text(m):
    if not m or "error" in m or m.get("h2_minus_h1_db") is None:
        return "谐波描述不可用"
    return (f"H2−H1 {m['h2_minus_h1_db']:+.1f} dB · H2–H12 {m['richness_k2_12_db']:.1f} dB · "
            f"2–4kHz 占比 {m['presence_2_4k_fraction'] * 100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="新的实验输出目录（已存在则拒绝）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="冒烟测试用：逗号分隔的 dataset id，如 A_self_01")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; use a new directory")
    if not out.parent.exists():
        raise SystemExit(f"parent directory does not exist: {out.parent}")
    audio_dir = out / "listening" / "audio"
    audio_dir.mkdir(parents=True)

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in manifest["items"]}
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    ids = only or ALL_IDS
    missing = [x for x in ids if x not in items]
    if missing:
        raise SystemExit(f"unknown ids: {missing}")

    import torch  # noqa: E402  (延迟 import，便于快速看到参数错误)

    engine = nsvb.NeuralSVB(f0_norm=None)
    clips = []
    for did in ids:
        item = items[did]
        x, sr, wav_path = load_clip(did, item)
        variants = {}
        for name in VARIANTS:
            torch.manual_seed(args.seed)  # vocoder 分支内部返回较早，这里统一固定随机种子
            wav, f0_in, f0_voc = engine.run(str(wav_path), mode=MODE_OF[name], seed=args.seed)
            variants[name] = {"raw": np.asarray(wav, dtype=np.float64), "f0_in": f0_in, "f0_voc": f0_voc}
        clips.append({"did": did, "sid": sid_of(did), "item": item, "x": x, "sr": sr,
                      "wav_path": wav_path, "variants": variants})
        print(f"inferred {sid_of(did)} {did} {item.get('song')} {item.get('variant')}", flush=True)

    # 一个全局响度目标：原音与全部处理版共用，保证同一曲的薄音/正常版也在同一响度。
    measured = {}
    for clip in clips:
        measured[(clip["sid"], "original")] = qa.measure_loudness(clip["x"], clip["sr"])
        for name, v in clip["variants"].items():
            measured[(clip["sid"], name)] = qa.measure_loudness(v["raw"], nsvb.SAMPLE_RATE)
    target = common_target(list(measured.values()))
    print(f"common_target_lufs={target:.3f}", flush=True)

    records = []
    for clip in clips:
        sid, did, item = clip["sid"], clip["did"], clip["item"]
        orig_export = audio_dir / f"{sid}_original.wav"
        qa.export_wav(clip["x"], clip["sr"],
                      target - measured[(sid, "original")]["integrated_lufs"], orig_export)
        orig_m, rate_o = qa.read_pcm16(orig_export)
        loud_o = qa.measure_loudness(orig_m, rate_o)
        record = {
            "sample_id": sid, "dataset_id": did, "song": item.get("song"), "variant": item.get("variant"),
            "input_wav": str(clip["wav_path"].relative_to(ROOT)).replace("\\", "/"),
            "input_sha256": sha256(clip["wav_path"]),
            "input_stats": qa.sample_stats(clip["x"], clip["sr"]),
            "input_loudness": measured[(sid, "original")],
            "common_target_lufs": target,
            "original_matched": {"file": str(orig_export.relative_to(out)).replace("\\", "/"),
                                 "sha256": sha256(orig_export)},
            "variants": {},
            "metrics": {"original": safe_metrics(orig_m, rate_o)},
        }
        for name, v in clip["variants"].items():
            path = audio_dir / f"{sid}_{name}.wav"
            qa.export_wav(v["raw"], nsvb.SAMPLE_RATE,
                          target - measured[(sid, name)]["integrated_lufs"], path)
            out_m, rate_out = qa.read_pcm16(path)
            checks = check_output(engine, orig_m, rate_o, loud_o, out_m, rate_out)
            f0_in, f0_voc = v["f0_in"][v["f0_in"] > 0], v["f0_voc"][v["f0_voc"] > 0]
            record["variants"][name] = {
                "file": str(path.relative_to(out)).replace("\\", "/"),
                "sha256": sha256(path),
                "gain_db": target - measured[(sid, name)]["integrated_lufs"],
                "f0_median_in_hz": float(np.median(f0_in)) if len(f0_in) else None,
                "f0_median_vocoder_hz": float(np.median(f0_voc)) if len(f0_voc) else None,
                "checks": checks,
            }
            record["metrics"][name] = safe_metrics(out_m, rate_out)
            print(f"{sid} {name}: technical_pass={checks['technical_pass']} {checks['errors']}", flush=True)
        records.append(record)

    by_id = {r["sample_id"]: r for r in records}
    n_checks = sum(len(r["variants"]) for r in records)
    n_pass = sum(1 for r in records for v in r["variants"].values() if v["checks"]["technical_pass"])
    errors = sorted({e for r in records for v in r["variants"].values() for e in v["checks"]["errors"]})
    tech_summary = (f"{n_pass}/{n_checks} 个处理输出通过全部工程检查"
                    if errors else f"{n_pass}/{n_checks} 个处理输出通过全部工程检查")
    if errors:
        tech_summary += "；未通过项：" + "；".join(errors)

    contract = {
        "original_problem": "A01/A03/A05/A07 为薄音/清唱版（自评薄）；A02/A04/A06/A08 为同曲正常版对照",
        "clips": "A01–A08 全长（13.14–15.00s），不截取；原始 WAV 哈希校验，未改动",
        "expected_change": "若模型有效，薄音版处理后谐波结构与 2–4kHz 占比应更接近同曲正常版；正常版不应被明显破坏",
        "rejected_side_effects": "换人感、沙哑、鼻音、变嗲、音高跳变/游移、机械感、齿音变重；只变响或只变亮不算改善",
        "prior_context": "2026-10-27 同一官方权重在 SHIGE/zuixuan 上被判不适配（沙哑/跳变/变嗲）；"
                         "本轮按 Jade 要求在新 8 条素材上复测，未训练、未改权重、未放宽阈值",
        "technical": tech_summary,
        "listening_status": "pending_jade",
    }

    # ---------------- 试听页（正常标注，不盲听） ----------------
    def harm_summary(sid, name):
        m = by_id.get(sid, {}).get("metrics", {}).get(name) if sid in by_id else None
        return f"{sid} {variant_label(name)}: {metrics_text(m)}"

    def card(pair, r):
        sid, nid = sid_of(pair["thin"]), sid_of(pair["normal"])
        lines = [harm_summary(sid, "original"), harm_summary(sid, "a2p")]
        if nid in by_id:
            lines.append(harm_summary(nid, "original") + "（同曲正常版）")
        meta = (f"{sid} {pair['song']}·薄音/清唱 · {r['input_stats']['duration_seconds']:.2f}s"
                f" · 谐波描述：{'；'.join(lines)}")
        rows = (
            row("薄音原音（48kHz）", f"audio/{sid}_original.wav") +
            row("薄音 + NeuralSVB a2p", f"audio/{sid}_a2p.wav") +
            row("薄音 + 声码器对照", f"audio/{sid}_vocoder_only.wav", "非候选；只说明重建本身的影响") +
            (row("同曲正常版参考", f"audio/{nid}_original.wav", f"{nid}；不同一次演唱，仅作音色参考")
             if nid in by_id else "")
        )
        return f'<div class="card"><h3>{esc(pair["song"])}（薄音 → 正常对照）</h3><p class="meta">{esc(meta)}</p>{rows}</div>'

    def normal_card(pair, r):
        sid = sid_of(pair["normal"])
        meta = (f"{sid} {pair['song']}·正常版 · {r['input_stats']['duration_seconds']:.2f}s"
                f" · 谐波描述：{harm_summary(sid, 'original')}；{harm_summary(sid, 'a2p')}")
        rows = (
            row("正常原音（48kHz）", f"audio/{sid}_original.wav") +
            row("正常 + NeuralSVB a2p", f"audio/{sid}_a2p.wav") +
            row("正常 + 声码器对照", f"audio/{sid}_vocoder_only.wav", "非候选；只说明重建本身的影响")
        )
        return f'<div class="card"><h3>{esc(pair["song"])}（正常版）</h3><p class="meta">{esc(meta)}</p>{rows}</div>'

    def check_table():
        head = ("<tr><th>文件</th><th>变体</th><th>时长差 s</th><th>响度 LUFS</th><th>真峰值 dBTP</th>"
                "<th>削波</th><th>音高中位/P95 音分</th><th>活跃段跨度 dB</th><th>跳变比</th><th>结果</th></tr>")
        body = []
        for r in records:
            for name, v in r["variants"].items():
                c = v["checks"]
                pc = c["pitch_vs_original"]
                pitch = (f"{pc.get('median_abs_cents', float('nan')):.1f} / {pc.get('p95_abs_cents', float('nan')):.1f}"
                         if "median_abs_cents" in pc else "不可用")
                body.append(
                    "<tr>"
                    f"<td>{esc(r['sample_id'])}</td><td>{esc(VARIANT_NAMES[name])}</td>"
                    f"<td>{c['duration_gap_s']:.4f}</td>"
                    f"<td>{c['loudness']['integrated_lufs']:.2f}</td>"
                    f"<td>{c['loudness']['true_peak_dbtp']:.2f}</td>"
                    f"<td>{c['samples']['clipped_samples']}</td>"
                    f"<td>{esc(pitch)}</td>"
                    f"<td>{c['active_gain']['spread_db']:.2f}</td>"
                    f"<td>{c['max_sample_jump_ratio']:.2f}</td>"
                    f"<td>{'通过' if c['technical_pass'] else '未通过'}</td>"
                    "</tr>")
        return f"<table>{head}{''.join(body)}</table>"

    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuralSVB 官方权重 · 8 条本人录音试听</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:940px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}h2{{font-size:20px;margin-top:26px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0;font-size:14px}}.row{{display:grid;grid-template-columns:200px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px;margin-bottom:14px;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #dce0e6;padding:6px 8px;text-align:right}}th{{background:#eef1f5;text-align:center}}td:first-child,td:nth-child(2){{text-align:left}}</style>
<h1>NeuralSVB 官方权重 · 8 条本人录音试听（4 薄音 / 4 正常）</h1>
<div class="note"><b>本轮登记（先于生成音频）：</b>原音问题：A01/A03/A05/A07 为薄音/清唱版（自评薄），A02/A04/A06/A08 为同曲正常版对照。<br>
<b>处理：</b>官方 NeuralSVB（1030_vae_mle@200k + 1009 PPG@136k + 1012 HiFi-GAN-NSF@1.17M），a2p 美化；F0 保持输入自身曲线（不改音高/节奏）；seed {args.seed}；不训练、不微调、不加 DSP。<br>
<b>预期可听变化：</b>若模型有效，薄音版的谐波结构与 2–4kHz 占比应更接近同曲正常版，听起来更实/更饱满；正常版不应被明显破坏。<br>
<b>否决副作用：</b>换人感、沙哑、鼻音、变嗲、音高跳变/游移、机械感、齿音变重；只变响或只变亮不算改善。<br>
<b>已知背景：</b>2026-10-27 同一官方权重在 SHIGE/zuixuan 上已被判不适配（沙哑/跳变/变嗲）；本轮按你的要求在 8 条新素材上复测，结论仍以你的听感为准。<br>
<b>工程状态：</b>{esc(tech_summary)}。处理版为模型原生 22.05kHz（带宽上限约 11kHz），原音 48kHz；全部试听文件统一到 {target:.2f} LUFS（常数增益，无限制器）。Agent 无听音能力，技术通过不等于变好听；听感状态 <b>pending_jade</b>。</div>
<h2>一、薄音版处理（同曲正常版作为参考）</h2>{''.join(card(p, by_id[sid_of(p['thin'])]) for p in PAIRS if sid_of(p['thin']) in by_id)}
<h2>二、正常版处理</h2>{''.join(normal_card(p, by_id[sid_of(p['normal'])]) for p in PAIRS if sid_of(p['normal']) in by_id)}
<h2>三、工程自检（阈值沿用 f1_poc / run_poc_batch，未放宽）</h2>
<div class="card">{check_table()}
<p class="meta">说明：音高诊断比较的是“响度匹配后的原音 vs 处理版”；声码器重建会对 F0 细结构产生抖动，这是上一轮已记录的已知限制，本轮不放宽阈值、也不把它写成“音高保持”。时长差 ≤23ms 来自模型 4 帧分组与尾部截断。谐波与 2–4kHz 数字仅为描述，不构成听感判定。</p></div>
<p class="meta">原始素材：test_dataset/f1_dataset_v2/A_thin/self/wav（哈希校验，未改动）。详细记录：report.json。</p></html>'''
    (out / "listening").mkdir(parents=True, exist_ok=True)
    (out / "listening" / "index.html").write_text(page, encoding="utf-8")

    report = {
        "experiment": "F1_neuralsvb_8songs_2026-11-01_v001",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "NeuralSVB MleSVBVAE 1030_vae_mle@200000 + 1009_pretrain_asr_english@136000"
                 " + 1012_hifigan_all_songs_nsf@1170000",
        "model_source": "third_party_neuralsvb/src/checkpoints",
        "code_reused": ["neuralsvb_f1/neuralsvb_infer.py", "f1_poc.py", "neuralsvb_f1/run_poc_batch.py::check_output"],
        "seed": args.seed, "sample_rate_out": nsvb.SAMPLE_RATE,
        "f0_policy": "identity: model pitch input and vocoder F0 = input's own contour",
        "listening_status": "pending_jade", "technical_summary": tech_summary,
        "technical_pass_count": n_pass, "technical_check_count": n_checks,
        "common_target_lufs": target, "listening_contract": contract,
        "samples": records,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "target_lufs": target, "technical": tech_summary,
                      "preview": str(out / "listening" / "index.html")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
