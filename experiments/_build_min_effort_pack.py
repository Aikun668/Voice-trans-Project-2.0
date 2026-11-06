# -*- coding: utf-8 -*-
import json, csv
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Users\杨玉璋\.codex\worktrees\f2-initial-plan\Voice-trans-Project-2.0")
EXP = ROOT / "experiments"
OUT = EXP / "F2_MIN_EFFORT_PACK_2026-11-05_v001"
RES = OUT / "results"
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)

PROPOSED = {
    "心墙01": {
        "text": "我学着不去担心得太远不计划太多反而能勇敢冒险",
        "n": 22,
        "jade_status": "已确认",
        "source": "Jade 原话确认；Qwen 已用该句",
        "note": "轻度版按命名同组；首字本人两版零时长",
        "self_control": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-对照微微跑调.wav",
        "self_mild": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-轻度跑调.wav",
        "ref": "test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/心墙_郭静_vocals.wav",
        "ref_windows": [{"id": "A", "start": 112.8, "end": 121.1}],
        "asr_hint": "对照识别曾把「去」听成「许」；已按确认句纠正",
    },
    "离开我的依赖01": {
        "text": "我蒙上了眼仿佛你在身旁当你不再应答",
        "n": 17,
        "jade_status": "待确认",
        "source": "https://www.kugeci.com/song/mqpSKkac （约 00:55–01:07）",
        "note": "ASR 常听成「梦上了眼」；请听本人对照版确认是「蒙」还是你实际唱的字",
        "self_control": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/离开我的依赖01-对照微微跑调.wav",
        "self_mild": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/离开我的依赖01-轻度跑调.wav",
        "ref": "test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/离开我的依赖_王艳薇_vocals.wav",
        "ref_windows": [{"id": "A", "start": 53.4, "end": 68.8}],
        "asr_hint": "本人对照识别：我梦上了眼 仿佛你在身旁 当你不再应答",
    },
    "离开我的依赖02": {
        "text": "我来不及道声不安有点混乱有点缓慢才发现承诺是谎话",
        "n": 24,
        "jade_status": "待确认",
        "source": "https://www.kugeci.com/song/mqpSKkac （约 01:09–01:20）",
        "note": "ASR 把「道声/混乱/承诺」听乱了；以官网词为候选，听一句确认即可",
        "self_control": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/离开我的依赖02-对照微微跑调.wav",
        "self_mild": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/离开我的依赖02-轻度跑调.wav",
        "ref": "test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/离开我的依赖_王艳薇_vocals.wav",
        "ref_windows": [{"id": "A", "start": 68.5, "end": 80.9}],
        "asr_hint": "本人对照识别：我来不及大声不安 有点浑软 有点缓慢 才发现沉默是谎话",
    },
    "莫问归期01": {
        "text": "藏进心口的刺不枉寻也如此沉默有时最后因你放肆",
        "n": 22,
        "jade_status": "待确认",
        "source": "https://www.mulanci.org/lyric/sl323329/ 第1–3行",
        "note": "原唱有两段重复；请选更像你这一段的 A 或 B（或都待定）",
        "self_control": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期01-对照微微跑调.wav",
        "self_mild": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期01-轻度跑调.wav",
        "ref": "test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/莫问归期_蒋雪儿_vocals.wav",
        "ref_windows": [
            {"id": "A", "start": 30.6, "end": 42.9},
            {"id": "B", "start": 113.1, "end": 125.4},
        ],
        "asr_hint": "本人对照识别：藏尽心口的词…（「刺」常被听成「词/字」）",
    },
    "莫问归期02": {
        "text": "沉默有时念想有时谁诀别相思成疾莫问天涯也莫问归期",
        "n": 24,
        "jade_status": "待确认",
        "source": "https://www.mulanci.org/lyric/sl323329/ 第6–8行附近",
        "note": "听本人对照版一句即可；有漏字/改口在备注写",
        "self_control": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期02-对照微微跑调.wav",
        "self_mild": "test_dataset/my_singing/rerecord_2026-11-04_f2/wav/莫问归期02-轻度跑调.wav",
        "ref": "test_dataset/my_singing/rerecord_2026-11-04_f2/ref_vocals/莫问归期_蒋雪儿_vocals.wav",
        "ref_windows": [{"id": "A", "start": 133.4, "end": 146.4}],
        "asr_hint": "本人对照识别：沉默有事,想念有事…莫问归期",
    },
}

for g, meta in PROPOSED.items():
    assert len(meta["text"]) == meta["n"], (g, len(meta["text"]), meta["n"])

xq = json.loads((EXP / "F2_XINQIANG_ALIGN_2026-11-05_v001/results/alignment.json").read_text(encoding="utf-8"))
fg = json.loads((EXP / "F2_FOUR_GROUPS_ALIGN_2026-11-05_v001/results/alignment.json").read_text(encoding="utf-8"))
rw = json.loads((EXP / "F2_REFERENCE_WINDOW_2026-11-05_v001/results/alignment.json").read_text(encoding="utf-8"))

DRIFT_THRESH = 0.35

def status_for(start, end, drift=None, lyric_confirmed=False):
    if start == end:
        return "零时长", "start==end；不可直接训练"
    if drift is not None and drift >= DRIFT_THRESH:
        return "漂移可疑", f"扩窗后起点变化 {drift:.2f}s"
    if lyric_confirmed:
        return "OK候选", "歌词已确认；仍待抽听后才可进可靠表"
    return "OK候选", "机器粗对齐；歌词或听感未确认，暂不可训练"

drift_map = {}
first_ref = {}
for it in fg["items"]:
    if "原唱" in it["label"]:
        first_ref[(it["group"], it["label"])] = {t["index"]: t["start_s"] for t in it["tokens"]}

for it in rw.get("items", []):
    prev = None
    matched_label = it["label"]
    for k, v in first_ref.items():
        if k[0] != it["group"]:
            continue
        if ("A" in it["label"] and "A" in k[1]) or ("B" in it["label"] and "B" in k[1]):
            prev = v
            matched_label = k[1]
            break
        if "A" not in it["label"] and "B" not in it["label"] and "A" not in k[1] and "B" not in k[1]:
            prev = v
            matched_label = k[1]
            break
    if prev is None:
        continue
    for t in it["tokens"]:
        if t["index"] in prev:
            d = abs(t["start_s"] - prev[t["index"]])
            drift_map[(it["group"], matched_label, t["index"])] = max(
                drift_map.get((it["group"], matched_label, t["index"]), 0.0), d
            )

rows = []
bad = []

for r in xq["results"]:
    if "对照" in r["label"]:
        version = "对照微微跑调"
    elif "轻度" in r["label"]:
        version = "轻度跑调"
    else:
        version = "原唱候选"
    for i, u in enumerate(r["units"], 1):
        s, e = u["start_source_s"], u["end_source_s"]
        st, reason = status_for(s, e, None, lyric_confirmed=True)
        row = {
            "group": "心墙01",
            "version": version,
            "index": i,
            "char": u["text"],
            "start_s": s,
            "end_s": e,
            "duration_s": round(e - s, 4),
            "time_base": "source_file",
            "source": r["source"].replace("\\", "/"),
            "window_start_s": r.get("source_start_s", 0.0),
            "status": st,
            "reason": reason,
            "lyric_status": "已确认",
            "round": "v001",
        }
        rows.append(row)
        if st in ("零时长", "漂移可疑"):
            bad.append(row)

for it in fg["items"]:
    group = it["group"]
    text = PROPOSED[group]["text"]
    chars = list(text)
    assert len(chars) == len(it["tokens"]), (group, it["label"], len(chars), len(it["tokens"]))
    if "对照" in it["label"]:
        version = "对照微微跑调"
    elif "轻度" in it["label"]:
        version = "轻度跑调"
    elif "B" in it["label"]:
        version = "原唱候选B"
    elif "A" in it["label"]:
        version = "原唱候选A"
    else:
        version = "原唱候选"
    lyric_ok = PROPOSED[group]["jade_status"] == "已确认"
    for t, ch in zip(it["tokens"], chars):
        s, e = t["start_s"], t["end_s"]
        drift = None
        if "原唱" in version:
            for lab in (it["label"], version, "原唱候选", "原唱候选A", "原唱候选B"):
                if (group, lab, t["index"]) in drift_map:
                    drift = drift_map[(group, lab, t["index"])]
                    break
        st, reason = status_for(s, e, drift, lyric_confirmed=lyric_ok)
        row = {
            "group": group,
            "version": version,
            "index": t["index"],
            "char": ch,
            "start_s": s,
            "end_s": e,
            "duration_s": round(e - s, 4),
            "time_base": "source_file",
            "source": it["source"].replace("\\", "/"),
            "window_start_s": it.get("source_start_s", 0.0),
            "status": st,
            "reason": reason,
            "lyric_status": PROPOSED[group]["jade_status"],
            "round": "v001",
        }
        rows.append(row)
        if st in ("零时长", "漂移可疑"):
            bad.append(row)

fields = list(rows[0].keys())
with (RES / "labeled_status.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
with (RES / "bad_spots.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(bad)

reliable = [r for r in rows if r["group"] == "心墙01" and r["status"] == "OK候选"]
with (RES / "reliable_pairs_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
    rf = fields + ["jade_listen", "keep_for_train"]
    w = csv.DictWriter(f, fieldnames=rf)
    w.writeheader()
    for r in reliable:
        rr = dict(r); rr["jade_listen"] = "pending"; rr["keep_for_train"] = "no_until_spotcheck"
        w.writerow(rr)

(RES / "proposed_lyrics.json").write_text(json.dumps(PROPOSED, ensure_ascii=False, indent=2), encoding="utf-8")
c = Counter(r["status"] for r in rows)
bc = Counter(f"{r['group']}|{r['version']}" for r in bad)
summary = {
    "n_rows": len(rows),
    "status_counts": dict(c),
    "n_bad_spots": len(bad),
    "bad_by_group_version": dict(bc),
    "drift_threshold_s": DRIFT_THRESH,
    "note": "OK候选不等于可训练；零时长/漂移可疑默认不可训练。",
}
(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

smoke = []
seen = set()
for r in bad:
    if r["status"] != "零时长":
        continue
    key = (r["group"], r["version"], r["source"])
    if key in seen:
        continue
    seen.add(key)
    zidx = [x["index"] for x in bad if x["group"]==r["group"] and x["version"]==r["version"] and x["status"]=="零时长"]
    smoke.append({
        "group": r["group"], "version": r["version"], "source": r["source"],
        "window_start_s": r["window_start_s"], "text": PROPOSED[r["group"]]["text"],
        "zero_indices": zidx, "suggested_tools": ["MFA", "HubertFA"],
        "priority": 1 if r["group"]=="心墙01" else 2,
    })
(RES / "a3_smoke_manifest.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
print("rows", len(rows), "bad", len(bad), "status", dict(c), "smoke", len(smoke))
