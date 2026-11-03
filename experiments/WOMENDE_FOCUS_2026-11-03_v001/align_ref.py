"""把 我们的歌 薄音版/正常版 各自对齐到王力宏原唱人声轨，得到对应区间与调差（.venv-f2）。"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "f2"))
from poc_phase4_realign import HOP, align, cents_track, cost_matrix, subseq_dtw  # noqa: E402
SELF = ROOT / "test_dataset/f1_dataset_v2/A_thin/self/wav"
REF = ROOT / "test_dataset/reference/vocals/我们的歌_王力宏_vocals.wav"
r = cents_track(REF)
out = {}
for k, f in {"thin": SELF / "我们的歌-薄音版.wav", "normal": SELF / "我们的歌正常版.wav"}.items():
    q = cents_track(f)
    cost, path, key = align(q, r)
    base, _ = subseq_dtw(cost_matrix(q, r[::-1].copy(), key))
    out[k] = {"ref_t0": path[0][1] * HOP, "ref_t1": path[-1][1] * HOP, "key_cents": key, "cost": round(cost, 1), "reversed": round(base, 1),
              "path": [(int(i), int(j)) for i, j in path]}
    print(k, {x: out[k][x] for x in ("ref_t0", "ref_t1", "key_cents", "cost", "reversed")})
(Path(__file__).parent / "align_ref.json").write_text(json.dumps(out), encoding="utf-8")
