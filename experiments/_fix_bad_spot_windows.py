# -*- coding: utf-8 -*-
"""Clamp F2 min-effort bad-spot listen windows to Jade dry clips; rebuild index.html."""
import csv
import json
import wave
from html import escape
from pathlib import Path

ROOT = Path(r"C:\Users\杨玉璋\.codex\worktrees\f2-initial-plan\Voice-trans-Project-2.0")
OUT = ROOT / "experiments" / "F2_MIN_EFFORT_PACK_2026-11-05_v001"
RES = OUT / "results"
PAD = 0.35
LISTEN_DUR = 1.4

PROPOSED = json.loads((RES / "proposed_lyrics.json").read_text(encoding="utf-8"))
summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
bad = list(csv.DictReader((RES / "bad_spots.csv").open(encoding="utf-8-sig")))


def wav_dur(rel: str) -> float:
    with wave.open(str(ROOT / rel), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def rel_url(p: str) -> str:
    return "../../" + p.replace("\\", "/")


def pick_dry(row: dict) -> str:
    """Prefer Jade dry clip for the same group/version; remap 原唱* to self_control."""
    src = row["source"].replace("\\", "/")
    group = row["group"]
    version = row["version"]
    meta = PROPOSED[group]
    if "/wav/" in src:
        return src
    # ref_vocals full song → Jade dry segment of same lyric group
    if "轻度" in version:
        return meta["self_mild"]
    return meta["self_control"]


stats = {
    "n": len(bad),
    "was_full_ref": 0,
    "was_play_oob": 0,
    "clamped_usable": 0,
    "out_of_clip": 0,
    "rows": [],
}

fixed = []
for i, r in enumerate(bad):
    src_old = r["source"].replace("\\", "/")
    ws = float(r["window_start_s"])
    start = float(r["start_s"])
    end = float(r["end_s"])
    local = start - ws
    local_end = end - ws

    # Old HTML bug: play_from = max(0, local-PAD) used as absolute currentTime on source.
    old_play_from = max(0.0, local - PAD)
    was_full_ref = "/ref_vocals/" in src_old
    if was_full_ref:
        stats["was_full_ref"] += 1
        # On full file, old play started at local offset (e.g. 6s) not window+local → outside 原唱片段
        stats["was_play_oob"] += 1
    else:
        old_src_dur = wav_dur(src_old)
        if old_play_from >= old_src_dur or start > old_src_dur + 1e-6:
            stats["was_play_oob"] += 1

    dry = pick_dry(r)
    dry_dur = wav_dur(dry)

    out_of = local < -1e-6 or local >= dry_dur - 1e-6
    # also if zero-duration label sits past clip end
    if local_end < -1e-6:
        out_of = True

    if out_of:
        listen_start = None
        listen_end = None
        listen_status = "OUT_OF_CLIP"
        ui_note = "Qwen 标点已超出本人干声片段时长；默认选「丢掉/不可用」或「对不上」"
        stats["out_of_clip"] += 1
        default_choice = "drop"
    else:
        listen_start = max(0.0, min(local - PAD, max(0.0, dry_dur - 0.05)))
        listen_end = min(dry_dur, listen_start + LISTEN_DUR)
        # ensure window non-empty
        if listen_end - listen_start < 0.2:
            listen_start = max(0.0, dry_dur - min(LISTEN_DUR, dry_dur))
            listen_end = dry_dur
        listen_status = "CLAMPED_OK"
        ui_note = "只听本人干声片段内；字大致落在这段=大概对"
        stats["clamped_usable"] += 1
        default_choice = ""

    row = dict(r)
    row["listen_source"] = dry
    row["listen_local_s"] = round(local, 4)
    row["clip_duration_s"] = round(dry_dur, 4)
    row["listen_start_s"] = "" if listen_start is None else round(listen_start, 4)
    row["listen_end_s"] = "" if listen_end is None else round(listen_end, 4)
    row["listen_status"] = listen_status
    row["ui_note"] = ui_note
    row["was_full_ref"] = "1" if was_full_ref else "0"
    row["default_choice"] = default_choice
    # keep original source/times for audit; playback uses listen_*
    fixed.append(row)
    stats["rows"].append(
        {
            "i": i + 1,
            "group": r["group"],
            "version": r["version"],
            "char": r["char"],
            "old_source": src_old.split("/")[-1],
            "listen_source": dry.split("/")[-1],
            "local_s": round(local, 3),
            "clip_dur": round(dry_dur, 3),
            "listen_status": listen_status,
            "listen_start": None if listen_start is None else round(listen_start, 3),
        }
    )

# write bad_spots.csv (augmented)
fieldnames = list(fixed[0].keys())
with (RES / "bad_spots.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(fixed)

summary["bad_spot_fix"] = {
    "n_bad_spots": len(fixed),
    "was_full_ref_n": stats["was_full_ref"],
    "was_play_oob_n": stats["was_play_oob"],
    "clamped_usable_n": stats["clamped_usable"],
    "out_of_clip_n": stats["out_of_clip"],
    "note": "Playback remapped to Jade dry wav clips; windows clamped to [0, duration]. OUT_OF_CLIP = Qwen local time past dry clip.",
}
summary["round"] = "v001-clamp"
(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(RES / "bad_spot_fix_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

# --- rebuild index.html (from _build_html_pack.py, with listen_* fix) ---
bad_cards = []
for r in fixed:
    bad_cards.append({**r, "src_rel": rel_url(r["listen_source"])})

parts = []
for gid, meta in PROPOSED.items():
    refs = []
    for w in meta["ref_windows"]:
        refs.append(
            '<button type="button" class="btn" data-play="%s" data-t="%s">原唱%s @%ss</button>'
            % (escape(rel_url(meta["ref"])), w["start"], escape(w["id"]), w["start"])
        )
    done = meta["jade_status"] == "已确认"
    disabled = "disabled" if done else ""
    pre = "已确认，可跳过" if done else "待你点选"
    checked = "checked" if done else ""
    mw = ""
    if gid == "莫问归期01":
        mw = (
            '<div class="row">原唱窗口：'
            '<label><input type="radio" name="mw01_ref" value="A"> A 30.6–42.9s</label>'
            '<label><input type="radio" name="mw01_ref" value="B"> B 113.1–125.4s</label>'
            '<label><input type="radio" name="mw01_ref" value="both_pending"> 都待定</label>'
            '<label><input type="radio" name="mw01_ref" value="unsure"> 拿不准</label>'
            "</div>"
        )
    opacity = ' style="opacity:.5"' if done else ""
    parts.append(
        '<section class="card" id="g-%s">'
        '<h2>%s <span class="tag">%s</span></h2>'
        '<p class="muted">来源：%s</p>'
        "<p><b>候选歌词（已去标点，字数=%s）：</b></p>"
        '<textarea id="text-%s" rows="2" %s>%s</textarea>'
        '<p class="muted">提示：%s<br>ASR 草稿：%s</p>'
        '<div class="row">'
        '<button type="button" class="btn" data-play="%s" data-t="0">听本人对照</button>'
        '<button type="button" class="btn" data-play="%s" data-t="0">听本人轻度</button>'
        "%s"
        "</div>%s"
        '<div class="row"%s>'
        '<label><input type="radio" name="st-%s" value="ok" %s %s> 歌词正确</label>'
        '<label><input type="radio" name="st-%s" value="edited" %s> 我改过上面的字</label>'
        '<label><input type="radio" name="st-%s" value="unsure" %s> 拿不准</label>'
        "</div>"
        '<label>备注 <input id="note-%s" type="text" style="width:70%%" %s></label>'
        "</section>"
        % (
            escape(gid),
            escape(gid),
            escape(pre),
            escape(meta["source"]),
            meta["n"],
            escape(gid),
            disabled,
            escape(meta["text"]),
            escape(meta["note"]),
            escape(meta["asr_hint"]),
            escape(rel_url(meta["self_control"])),
            escape(rel_url(meta["self_mild"])),
            " ".join(refs),
            mw,
            opacity,
            escape(gid),
            checked,
            disabled,
            escape(gid),
            disabled,
            escape(gid),
            disabled,
            escape(gid),
            disabled,
        )
    )

bad_parts = []
for i, r in enumerate(bad_cards):
    ooc = r["listen_status"] == "OUT_OF_CLIP"
    drop_checked = "checked" if ooc else ""
    if ooc:
        btn = '<button type="button" class="btn" disabled>超出本人干声片段 · 不可播</button>'
        extra = '<p class="warn">OUT_OF_CLIP：Qwen 时间已落在本人干声片段之外 → 请选「丢掉/不可用」</p>'
        stop_attr = ""
        play_t = "0"
    else:
        play_t = r["listen_start_s"]
        stop_len = round(float(r["listen_end_s"]) - float(r["listen_start_s"]), 3)
        btn = (
            '<button type="button" class="btn primary" data-play="%s" data-t="%s" data-stop="%s">'
            "听本人干声片段内约%.1f秒</button>"
            % (escape(r["src_rel"]), play_t, stop_len, stop_len)
        )
        extra = (
            '<p class="muted">干声 %s · 片段内 %.2f–%.2fs（clip=%.2fs）· 原 Qwen 源文件 %.2f–%.2fs</p>'
            % (
                escape(Path(r["listen_source"]).name),
                float(r["listen_start_s"]),
                float(r["listen_end_s"]),
                float(r["clip_duration_s"]),
                float(r["start_s"]),
                float(r["end_s"]),
            )
        )
    badge = "OUT_OF_CLIP" if ooc else r["status"]
    bad_parts.append(
        '<div class="card bad">'
        "<div><b>#%s</b> %s · %s · 字#%s「%s」· <span class=\"warn\">%s</span></div>"
        "%s%s"
        '<div class="row">'
        '<label><input type="radio" name="bad-%s" value="wrong_time"> 对不上/时间不对</label>'
        '<label><input type="radio" name="bad-%s" value="ok_enough"> 大概对（字在这段里）</label>'
        '<label><input type="radio" name="bad-%s" value="unsure"> 拿不准</label>'
        '<label><input type="radio" name="bad-%s" value="drop" %s> 丢掉/不可用</label>'
        "</div></div>"
        % (
            i + 1,
            escape(r["group"]),
            escape(r["version"]),
            r["index"],
            escape(r["char"]),
            escape(badge),
            extra,
            btn,
            i,
            i,
            i,
            i,
            drop_checked,
        )
    )

group_keys_js = json.dumps(list(PROPOSED.keys()), ensure_ascii=False)
html = """<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>F2 最小操作包 · Jade</title>
<style>
body{font-family:system-ui,sans-serif;max-width:880px;margin:24px auto;padding:0 16px;line-height:1.45}
h1{font-size:1.35rem} h2{font-size:1.1rem;margin:0 0 8px}
.card{border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin:14px 0;background:#fff}
.bad{border-color:#f0b4b4;background:#fff8f8}
.muted{color:#666;font-size:.92rem}
.warn{color:#b00020;font-weight:600}
.tag{font-size:.8rem;background:#eef;padding:2px 8px;border-radius:999px}
.row{display:flex;flex-wrap:wrap;gap:10px 14px;margin:8px 0;align-items:center}
.btn{padding:6px 10px;border:1px solid #888;border-radius:8px;background:#f7f7f7;cursor:pointer}
.btn.primary{background:#1a73e8;color:#fff;border-color:#1a73e8}
.btn:disabled{opacity:.55;cursor:not-allowed}
textarea{width:100%;font-size:1rem;padding:8px}
#player{position:sticky;bottom:0;background:#111;color:#fff;padding:10px 14px;border-radius:10px 10px 0 0}
#out{white-space:pre-wrap;background:#f4f4f4;padding:12px;border-radius:8px}
.steps{background:#f0f7ff;border:1px solid #cfe0ff;border-radius:10px;padding:12px 14px}
</style>
</head>
<body>
<h1>F2 最小操作包（你只需做这两步）</h1>
<div class=\"steps\">
<ol>
<li><b>确认另外四组歌词</b>（心墙已确认可跳过）。每组听一遍本人对照，对就点「歌词正确」，有改口就改字再点「我改过」。</li>
<li><b>只听坏点</b>：下面 BADN 处。每处只播<strong>本人干声片段内</strong>约 1.4 秒；字大致落在这段就选「大概对」。落在窗外/OUT_OF_CLIP →「丢掉/不可用」。</li>
</ol>
<p class=\"muted\">已修复：不再用整首下载原曲的全局时间轴抽听。源文件未等响，先调低音量。点底部「生成答案」复制发我。</p>
<p class=\"muted\">工程统计：总字时标 ROWS 行；STATUS；坏点修复 FIXNOTE。</p>
</div>
<audio id=\"audio\" controls style=\"width:100%;margin:8px 0\"></audio>
<h2>第一步：歌词确认</h2>
GROUPS
<h2>第二步：坏点抽听（BADN）</h2>
BADS
<div class=\"card\">
  <button type=\"button\" class=\"btn primary\" id=\"gen\">生成答案并复制</button>
  <button type=\"button\" class=\"btn\" id=\"dl\">下载 answers.txt</button>
  <pre id=\"out\"></pre>
</div>
<div id=\"player\">正在听：<span id=\"now\">未播放</span></div>
<script>
const audio = document.getElementById('audio');
let stopAt = null;
audio.addEventListener('timeupdate', () => {
  if (stopAt != null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; }
});
document.body.addEventListener('click', (e) => {
  const b = e.target.closest('[data-play]');
  if (!b) return;
  const src = b.getAttribute('data-play');
  const t = parseFloat(b.getAttribute('data-t') || '0');
  const dur = parseFloat(b.getAttribute('data-stop') || '0');
  document.getElementById('now').textContent = src.split('/').pop() + ' @' + t + 's';
  if (audio.getAttribute('src') !== src) { audio.src = src; }
  const start = () => { audio.currentTime = t; stopAt = dur ? t + dur : null; audio.play(); };
  if (audio.readyState >= 1) start(); else audio.onloadedmetadata = start;
});
const groups = GROUPKEYS;
const badN = BADN;
function collect() {
  const lines = [];
  lines.push('F2_MIN_EFFORT_PACK answers');
  lines.push('date: ' + new Date().toISOString());
  lines.push('pack: v001-clamp');
  lines.push('=== LYRICS ===');
  for (const g of groups) {
    const st = document.querySelector('input[name=\"st-' + g + '\"]:checked');
    const text = document.getElementById('text-' + g).value.trim();
    const note = document.getElementById('note-' + g).value.trim();
    let extra = '';
    if (g === '莫问归期01') {
      const r = document.querySelector('input[name=\"mw01_ref\"]:checked');
      extra = ' ref_pick=' + (r ? r.value : 'unset');
    }
    lines.push(g + ' | status=' + (st ? st.value : 'unset') + ' | text=' + text + ' | note=' + note + extra);
  }
  lines.push('=== BAD_SPOTS ===');
  for (let i=0;i<badN;i++) {
    const st = document.querySelector('input[name=\"bad-' + i + '\"]:checked');
    lines.push('bad#' + (i+1) + ' | ' + (st ? st.value : 'unset'));
  }
  return lines.join('\\n');
}
document.getElementById('gen').onclick = async () => {
  const t = collect();
  document.getElementById('out').textContent = t;
  try { await navigator.clipboard.writeText(t); alert('已复制，直接粘贴发我'); }
  catch(e) { alert('复制失败，请手动复制下方文本'); }
};
document.getElementById('dl').onclick = () => {
  const t = collect();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([t], {type:'text/plain'}));
  a.download = 'F2_MIN_EFFORT_answers.txt';
  a.click();
};
</script>
</body>
</html>
"""
fixnote = (
    "原唱候选曾误播整首ref；已改本人干声。"
    "越界原=%s 可用=%s OUT_OF_CLIP=%s"
    % (stats["was_play_oob"], stats["clamped_usable"], stats["out_of_clip"])
)
html = (
    html.replace("GROUPS", "\n".join(parts))
    .replace("BADS", "\n".join(bad_parts))
    .replace("BADN", str(len(bad_cards)))
    .replace("ROWS", str(summary["n_rows"]))
    .replace("STATUS", escape(str(summary["status_counts"])))
    .replace("FIXNOTE", escape(fixnote))
    .replace("GROUPKEYS", group_keys_js)
)
(OUT / "index.html").write_text(html, encoding="utf-8")

jade = """# 你只要做这些

打开：
`C:\\Users\\杨玉璋\\.codex\\worktrees\\f2-initial-plan\\Voice-trans-Project-2.0\\experiments\\F2_MIN_EFFORT_PACK_2026-11-05_v001\\index.html`

1. 四组歌词：听本人对照 → 点「正确」或改字。
2. 莫问归期 01：选原唱 A / B / 都待定。
3. 17 个坏点：只听**本人干声片段内**；字落在这段=大概对；标 OUT_OF_CLIP / 窗外=丢掉。
4. 「生成答案并复制」→ 粘贴发我。

已修复：不再用整首下载原曲全局时间轴抽听。
"""
(OUT / "JADE_ONLY.md").write_text(jade, encoding="utf-8")

print("DONE")
print("was_full_ref", stats["was_full_ref"])
print("was_play_oob", stats["was_play_oob"])
print("clamped_usable", stats["clamped_usable"])
print("out_of_clip", stats["out_of_clip"])
for row in stats["rows"]:
    print(
        "#{i} {group}|{version}「{char}」 local={local_s} clip={clip_dur} -> {listen_status} listen@{listen_start} file={listen_source}".format(
            **row
        )
    )
