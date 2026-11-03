"""列出 Phase-3c 的配对音符, 按时长×修正量排序(感知显著性代理)。"""
import csv

rows = list(csv.DictReader(open(r"f2/outputs/phase3c_shige_origref_v001/pairs.csv", encoding="utf-8")))
print(f"总配对 {len(rows)} 音")
print(f"{'时间':>12} {'时长s':>6} {'唱的音':>6} {'原唱':>6} {'修正':>8} {'时长x修正':>10}")
ordered = sorted(rows, key=lambda z: -(float(z["t1"]) - float(z["t0"])) * abs(float(z["applied"])))
for r in ordered[:12]:
    dur = float(r["t1"]) - float(r["t0"])
    app = float(r["applied"])
    sal = dur * abs(app)
    mark = "修" if app != 0 else "  "
    print(f"{mark} {float(r['t0']):5.2f}-{float(r['t1']):5.2f} {dur:6.2f} {r['note']:>6} "
          f"{r['ref_note']:>6} {app:+8.1f} {sal:10.0f}")
