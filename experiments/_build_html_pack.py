# -*- coding: utf-8 -*-
"""Rebuild F2 min-effort index.html.
IMPORTANT: bad-spot playback MUST use listen_* columns from bad_spots.csv
(produced by _fix_bad_spot_windows.py). Do NOT recompute play_from = local
and point it at ref_vocals full files — that played outside Jade's dry clip.
"""
import json, csv
from pathlib import Path
from html import escape

ROOT = Path(r"C:\Users\杨玉璋\.codex\worktrees\f2-initial-plan\Voice-trans-Project-2.0")
OUT = ROOT / "experiments" / "F2_MIN_EFFORT_PACK_2026-11-05_v001"
RES = OUT / "results"
PROPOSED = json.loads((RES / "proposed_lyrics.json").read_text(encoding="utf-8"))
bad = list(csv.DictReader((RES / "bad_spots.csv").open(encoding="utf-8-sig")))
summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))

def rel(p):
    return "../../" + p.replace("\\", "/")

# Prefer fixed listen_* fields; fall back only for dry wav rows.
bad_cards = []
for r in bad:
    listen_src = r.get("listen_source") or r["source"]
    status = r.get("listen_status") or ""
    if status == "OUT_OF_CLIP" or not r.get("listen_start_s"):
        play_from = None
        play_stop = None
    else:
        play_from = float(r["listen_start_s"])
        play_stop = round(float(r["listen_end_s"]) - float(r["listen_start_s"]), 3)
    bad_cards.append({**r, "play_from": play_from, "play_stop": play_stop, "src_rel": rel(listen_src), "listen_status": status or r.get("status","")})

print("bad cards", len(bad_cards), "OUT_OF_CLIP", sum(1 for c in bad_cards if c.get("listen_status")=="OUT_OF_CLIP"))
print("Rebuild full HTML via: python experiments/_fix_bad_spot_windows.py")
print("(This stub only validates CSV listen_* fields; full HTML writer lives in the fix script.)")
