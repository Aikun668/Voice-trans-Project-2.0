"""F1-EXP-001: single-module EQ baseline (Phase-2). Dev only."""
import csv, hashlib, json, pathlib, platform, random, shutil, subprocess, sys
from datetime import datetime, timezone
import numpy as np
import soundfile as sf
from pedalboard import HighShelfFilter, LowShelfFilter, Pedalboard
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats
OUT = ROOT / "experiments/F1-EXP-001"
if (OUT / "summary.json").exists():
    raise FileExistsError("F1-EXP-001 outputs already exist")
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}
DEV = [("A01","A_self_01","Self","平凡之路 / 薄音版"),("A02","A_self_03","Self","我们的歌 / 薄音版"),("A03","A_self_05","Self","雅俗共赏 / 清唱版"),("A05","A_gt_02","GTSinger","女A-002（来源未保留）"),("B01","B_gt_02","GTSinger","女B-002（来源未保留）"),("B02","B_gt_06","GTSinger","男B-003（来源未保留）")]
SEALED = ["A04","A06","B03","C03","C01","C02"]
SEED = 20261031
SCHEMES = {"E1":{"purpose":"中低频主体感：轻微宽带低架","filter":LowShelfFilter(cutoff_frequency_hz=220,gain_db=2.0,q=0.7),"config":{"type":"LowShelfFilter","cutoff_hz":220,"gain_db":2.0,"q":0.7}},"E2":{"purpose":"中高频存在感：轻微宽带高架","filter":HighShelfFilter(cutoff_frequency_hz=3000,gain_db=1.5,q=0.7),"config":{"type":"HighShelfFilter","cutoff_hz":3000,"gain_db":1.5,"q":0.7}}}
FFMPEG = subprocess.run(["ffmpeg","-version"],capture_output=True,text=True)
if FFMPEG.returncode != 0: raise RuntimeError("ffmpeg not on PATH")
FFMPEG_VERSION = FFMPEG.stdout.splitlines()[0]
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def db(v): return float(20*np.log10(max(float(v),1e-12)))
raw_root = OUT / "outputs/processed_raw"
listen_root = OUT / "listening/audio"
qa_dir = OUT / "qa"
for d in (raw_root/"E1", raw_root/"E2", listen_root, qa_dir):
    if d.exists(): raise FileExistsError(d)
    d.mkdir(parents=True)
inputs = []
for sid,did,source,display in DEV:
    item = items[did]
    wav = DATA/item["wav"]["path"]; raw = DATA/item["raw"]["path"]
    actual = sha(raw)
    if actual != item["raw"]["sha256"]: raise ValueError(f"Input raw hash changed: {raw}")
    inputs.append({"sample_id":sid,"dataset_id":did,"source":source,"display":display,"wav_file":str(wav.relative_to(ROOT)).replace("\\\\","/"),"raw_file":str(raw.relative_to(ROOT)).replace("\\\\","/"),"wav_sha256":sha(wav),"raw_sha256":actual,"split":"dev"})
(OUT/"inputs.json").write_text(json.dumps({"experiment":"F1-EXP-001","phase":"F1-P2","dev":inputs,"sealed_not_processed":SEALED,"note":"Dev only; sealed samples are not read or processed by this script."},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
records = []; raw_metrics = {}
for inp in inputs:
    sid = inp["sample_id"]
    x,sr = read_pcm16(DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2"))
    raw_metrics[(sid,"Original")] = measure_loudness(x,sr)
    entry = {"sample_id":sid,"dataset_id":inp["dataset_id"],"source":inp["source"],"display":inp["display"],"input_wav":inp["wav_file"],"input_sha256":inp["wav_sha256"],"sample_rate":sr,"frames":len(x),"length_preserved":True,"nonfinite":False,"raw_clipping":False,"abnormal_gain":False,"schemes":{},"notes":[]}
    for scheme,spec in SCHEMES.items():
        y = np.asarray(Pedalboard([spec["filter"]])(np.asarray(x,dtype=np.float32),sr),dtype=np.float64).reshape(-1)
        if len(y) != len(x): raise ValueError(f"Length changed for {sid}/{scheme}")
        if not np.all(np.isfinite(y)): raise ValueError(f"Non-finite output for {sid}/{scheme}")
        stats = sample_stats(y,sr); loud = measure_loudness(y,sr)
        raw_path = raw_root/scheme/f"{sid}.wav"
        export_wav(y,sr,0.0,raw_path)
        exported,exported_sr = read_pcm16(raw_path); exported_loud = measure_loudness(exported,exported_sr)
        clipped = int(np.count_nonzero(np.abs(exported) >= 32767/32768))
        lufs_gain = exported_loud["integrated_lufs"] - raw_metrics[(sid,"Original")]["integrated_lufs"]
        entry["schemes"][scheme] = {"filter":spec["config"],"purpose":spec["purpose"],"raw_file":str(raw_path.relative_to(ROOT)).replace("\\\\","/"),"raw_sha256":sha(raw_path),"raw_frames":len(exported),"raw_peak_dbfs":stats["sample_peak_dbfs"],"raw_rms_dbfs":stats["rms_dbfs"],"raw_clipped_samples":clipped,"raw_integrated_lufs":exported_loud["integrated_lufs"],"raw_true_peak_dbtp":exported_loud["true_peak_dbtp"],"lufs_gain_vs_original_db":lufs_gain}
        raw_metrics[(sid,scheme)] = exported_loud
        if clipped: entry["raw_clipping"] = True
        if abs(lufs_gain) > 3.0: entry["abnormal_gain"] = True
    if entry["raw_clipping"]: entry["notes"].append("raw processed output clipped")
    if entry["abnormal_gain"]: entry["notes"].append("raw processed loudness gain >3 dB")
    records.append(entry)
target = min(-18.0, *[m["integrated_lufs"]-3.2-m["true_peak_dbtp"] for m in raw_metrics.values()])
rng = random.Random(SEED); blind_key = {}; listen_manifest = []
for inp in inputs:
    sid = inp["sample_id"]; a_is_e1 = bool(rng.getrandbits(1))
    blind_key[sid] = {"version_A":"E1" if a_is_e1 else "E2","version_B":"E2" if a_is_e1 else "E1"}
    entry = next(e for e in records if e["sample_id"] == sid)
    for label, scheme in (("Original",None),("E1","E1"),("E2","E2")):
        source_file = DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2") if scheme is None else raw_root/scheme/f"{sid}.wav"
        x,sr = read_pcm16(source_file); meter = measure_loudness(x,sr); gain = target-meter["integrated_lufs"]
        if scheme is None:
            dest = listen_root/f"{sid}_original.wav"
        else:
            is_a = (scheme=="E1" and a_is_e1) or (scheme=="E2" and not a_is_e1)
            dest = listen_root/f"{sid}_{'A' if is_a else 'B'}.wav"
        export_wav(x,sr,gain,dest); exported,exported_sr = read_pcm16(dest); check = measure_loudness(exported,exported_sr)
        rec = {"file":str(dest.relative_to(ROOT)).replace("\\\\","/"),"gain_db":gain,"lufs":check["integrated_lufs"],"true_peak_dbtp":check["true_peak_dbtp"],"frames":len(exported),"sha256":sha(dest)}
        if scheme is None:
            entry["original_matched"] = rec
        else:
            entry["schemes"][scheme]["matched"] = rec
            entry["schemes"][scheme]["listening_label"] = "A" if ((scheme=="E1" and a_is_e1) or (scheme=="E2" and not a_is_e1)) else "B"
    listen_manifest.append({"sample_id":sid,"source":inp["source"],"display":inp["display"],"original":entry["original_matched"]["file"],"A":entry["schemes"]["E1" if a_is_e1 else "E2"]["matched"]["file"],"B":entry["schemes"]["E2" if a_is_e1 else "E1"]["matched"]["file"]})
    entry["listening"] = {"target_lufs":target,"blind_key_file":"qa/blind_key.json"}
bad_peaks = [(e["sample_id"],k,v["matched"]["true_peak_dbtp"]) for e in records for k,v in e["schemes"].items() if v["matched"]["true_peak_dbtp"] > -2.9]
if bad_peaks: raise RuntimeError(f"Matched true peak too hot: {bad_peaks}")
qa = {"experiment":"F1-EXP-001","phase":"F1-P2","created_utc":datetime.now(timezone.utc).isoformat(),"dev":[i["sample_id"] for i in inputs],"sealed_not_processed":SEALED,"schemes":{k:v["config"] for k,v in SCHEMES.items()},"target_lufs":target,"blind_seed":SEED,"technical_pass":not any(e["raw_clipping"] or e["abnormal_gain"] or not e["length_preserved"] for e in records),"records":records}
(OUT/"qa/qa_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"qa/blind_key.json").write_text(json.dumps({"seed":SEED,"mapping":blind_key},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"listening/manifest.json").write_text(json.dumps({"target_lufs":target,"samples":listen_manifest},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with open(OUT/"qa/levels.csv","w",encoding="utf-8-sig",newline="") as f:
    w = csv.writer(f); w.writerow(["sample_id","scheme","raw_peak_dbfs","raw_rms_dbfs","raw_lufs","raw_true_peak_dbtp","raw_clipped","lufs_gain_vs_original_db","matched_lufs","matched_true_peak_dbtp","matched_gain_db"])
    for e in records:
        o = e["original_matched"]; w.writerow([e["sample_id"],"Original","","",o["lufs"],o["true_peak_dbtp"],0,0,o["lufs"],o["true_peak_dbtp"],o["gain_db"]])
        for scheme,s in e["schemes"].items(): w.writerow([e["sample_id"],scheme,s["raw_peak_dbfs"],s["raw_rms_dbfs"],s["raw_integrated_lufs"],s["raw_true_peak_dbtp"],s["raw_clipped_samples"],s["lufs_gain_vs_original_db"],s["matched"]["lufs"],s["matched"]["true_peak_dbtp"],s["matched"]["gain_db"]])
def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
by_sid = {m["sample_id"]:m for m in listen_manifest}
cards = "".join(f'<section class="card"><h3>{inp["sample_id"]} · {inp["source"]} · {esc(inp["display"])}</h3><p class="meta">Dev · 原音 / 版本 A / 版本 B（A/B 已随机，不显示处理方向）</p><div class="row"><label>原音</label><audio controls preload="metadata" src="audio/{esc(by_sid[inp["sample_id"]]["original"].replace("\\","/").split("/")[-1])}"></audio></div><div class="row"><label>版本 A</label><audio controls preload="metadata" src="audio/{esc(by_sid[inp["sample_id"]]["A"].replace("\\","/").split("/")[-1])}"></audio></div><div class="row"><label>版本 B</label><audio controls preload="metadata" src="audio/{esc(by_sid[inp["sample_id"]]["B"].replace("\\","/").split("/")[-1])}"></audio></div></section>' for inp in inputs)
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P2 EQ 试听</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:70px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}}</style><h1>F1-P2 · 单模块 EQ 试听</h1><div class="note"><b>听法：</b>每个样本先听“原音”，再听 A/B。A/B 的处理方向已随机隐藏；请分别判断：饱满度是否更好、自然度是否下降、是否仍像本人、是否更愿意听。所有版本已等响，只是固定增益，没有压缩/饱和/混响/谐波。</div>{cards}<p class="meta">本页只含 Dev；A04/A06/B03/C03 已封存，未处理。</p></html>'''
(OUT/"listening/index.html").write_text(page,encoding="utf-8")
params = {"experiment":"F1-EXP-001","phase":"F1-P2","dev":[i["sample_id"] for i in inputs],"sealed":SEALED,"schemes":{k:v["config"] for k,v in SCHEMES.items()},"target_lufs":target,"blind_seed":SEED,"code":"experiments/F1-EXP-001/process.py","no_stacking":True}
(OUT/"params.json").write_text(json.dumps(params,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary = {"experiment":"F1-EXP-001","phase":"F1-P2","status":"pending_jade","dev_count":len(inputs),"sealed_count":len(SEALED),"target_lufs":target,"technical_pass":qa["technical_pass"],"preview_page":"listening/index.html","blind_key":"qa/blind_key.json","raw_outputs":"outputs/processed_raw","listening_audio":"listening/audio"}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"run_log.txt").write_text(f"created_utc: {qa['created_utc']}\nexperiment: F1-EXP-001\nphase: F1-P2\npython: {sys.version.split()[0]}\nplatform: {platform.platform()}\nffmpeg: {FFMPEG_VERSION}\ndev: {', '.join(i['sample_id'] for i in inputs)}\nsealed: {', '.join(SEALED)}\nschemes: {json.dumps({k:v['config'] for k,v in SCHEMES.items()},ensure_ascii=False)}\ntarget_lufs: {target}\ntechnical_pass: {qa['technical_pass']}\nnote: original inputs not modified; holdout not processed\n",encoding="utf-8")
(OUT/"RESULT.md").write_text("# F1-EXP-001 结果（pending_jade）\n\n本实验只跑 Dev 的 E1/E2 初始参数；Holdout 未处理。\n\n技术检查已完成，具体见 qa/qa_report.json。听感结论尚未产生，状态保持 pending_jade；用户在 listening/index.html 试听后，再决定保留、修订或停止。\n",encoding="utf-8")
print(json.dumps({"summary":summary,"qa":qa["technical_pass"],"target":target},ensure_ascii=False,indent=1))