"""列出 v002 的修正明细, 便于判断是否合理。"""
import csv
import statistics

rows = list(csv.DictReader(open(r"f2/outputs/phase3_shige_v002/deviations.csv", encoding="utf-8")))
rel = [abs(float(r["relative"])) for r in rows]
print(f"音符 {len(rows)}  中位|相对偏差| {statistics.median(rel):.1f}  最大 {max(rel):.1f}")
print()
for r in rows:
    a = float(r["applied"])
    flag = "修" if a != 0 else "  "
    print(f"{flag} {float(r['t0']):5.2f}s  {r['hz']:>6}Hz {r['note']:>4} -> {r['ref_note']:>4}  "
          f"相对 {float(r['relative']):+7.1f}  修正 {a:+7.1f}")
