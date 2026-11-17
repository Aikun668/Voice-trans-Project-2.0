"""F1-EXP-003: Phase-2.2 final strength diagnostic (+4 vs +8 dB high shelf)."""
import csv, hashlib, json, pathlib, platform, random, subprocess, sys
from datetime import datetime, timezone
import numpy as np
from pedalboard import HighShelfFilter, Pedalboard
from scipy import signal
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats
OUT = ROOT / "experiments/F1-EXP-003"
if (OUT / "summary.json").exists(): raise FileExistsError("F1-EXP-003 outputs already exist")
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}
TEST = [("A01","A_self_01","Self","平凡之路 / 薄音版"),("A02","A_self_03","Self","我们的歌 / 薄音版"),("A05","A_gt_02","GTSinger","女A-002（来源未保留）"),("B01","B_gt_02","GTSinger","女B-002（来源未保留）")]
SEALED = ["A03","A04","A06","B02","B03","C01","C02","C03"]
SCHEMES = {"E2_4p0":{"filter":HighShelfFilter(cutoff_frequency_hz=3000,gain_db=4.0,q=0.7),"config":{"type":"HighShelfFilter","cutoff_hz":3000,"gain_db":4.0,"q":0.7}},"E2_8p0":{"filter":HighShelfFilter(cutoff_frequency_hz=3000,gain_db=8.0,q=0.7),"config":{"type":"HighShelfFilter","cutoff_hz":3000,"gain_db":8.0,"q":0.7}}}
FFMPEG = subprocess.run(["ffmpeg","-version"],capture_output=True,text=True)
if FFMPEG.returncode != 0: raise RuntimeError("ffmpeg not on PATH")
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def process(x,sr,flt): return np.asarray(Pedalboard([flt])(np.asarray(x,dtype=np.float32),sr),dtype=np.float64).reshape(-1)
def band_gains(x,y,sr):
    f,p1 = signal.welch(x,sr,nperseg=min(4096,len(x))); f,p2 = signal.welch(y,sr,nperseg=min(4096,len(x)))
    def g(lo,hi):
        m=(f>=lo)&(f<hi); return float(10*np.log10(np.mean(p2[m])/max(np.mean(p1[m]),1e-20)))
    return {"low_80_500":g(80,500),"high_4000_8000":g(4000,8000),"tilt":g(4000,8000)-g(80,500)}
def fr(ir,sr,freqs):
    spec=np.fft.rfft(ir); f=np.fft.rfftfreq(len(ir),1/sr); return {str(hz):float(20*np.log10(max(abs(spec[int(np.argmin(np.abs(f-hz)))]),1e-20))) for hz in freqs}
raw_root=OUT/"outputs/processed_raw"; listen_root=OUT/"listening/audio"; qa_dir=OUT/"qa"
for d in (raw_root/"E2_4p0",raw_root/"E2_8p0",listen_root,qa_dir):
    if d.exists(): raise FileExistsError(d)
    d.mkdir(parents=True)
inputs=[]
for sid,did,source,display in TEST:
    item=items[did]; wav=DATA/item["wav"]["path"]; raw=DATA/item["raw"]["path"]
    if sha(raw)!=item["raw"]["sha256"]: raise ValueError(f"Input raw hash changed: {raw}")
    inputs.append({"sample_id":sid,"dataset_id":did,"source":source,"display":display,"wav_file":str(wav.relative_to(ROOT)).replace("\\\\","/"),"raw_file":str(raw.relative_to(ROOT)).replace("\\\\","/"),"wav_sha256":sha(wav),"raw_sha256":sha(raw)})
(OUT/"inputs.json").write_text(json.dumps({"experiment":"F1-EXP-003","phase":"F1-P2.2","test":inputs,"sealed_not_processed":SEALED},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
# Response + linearity + raw processing.
impulse=np.zeros(2*48000); impulse[0]=1.0; freqs=[100,500,1000,2000,3000,4000,6000,8000,12000,16000]
response={}; linearity={}
for scheme,spec in SCHEMES.items():
    ir=process(impulse,48000,spec["filter"]); response[scheme]={"frequency_response_db":fr(ir,48000,freqs),"decay_ms":float(np.flatnonzero(np.abs(ir)>max(np.max(np.abs(ir))*1e-4,1e-9))[-1]/48)}
    x1=np.random.default_rng(0).normal(0,0.2,96000); x2=np.random.default_rng(1).normal(0,0.2,96000); y1=process(x1,48000,spec["filter"]); y2=process(x2,48000,spec["filter"]); yscale=process(2*x1,48000,spec["filter"]); yadd=process(x1+x2,48000,spec["filter"]); yneg=process(-x1,48000,spec["filter"]); linearity[scheme]={"scale":float(np.max(np.abs(yscale-2*y1))),"additivity":float(np.max(np.abs(yadd-y1-y2))),"negation":float(np.max(np.abs(yneg+y1))),"pass":bool(max(np.max(np.abs(yscale-2*y1)),np.max(np.abs(yadd-y1-y2)),np.max(np.abs(yneg+y1)))<1e-5)}
records=[]; raw_metrics={}
for inp in inputs:
    sid=inp["sample_id"]; x,sr=read_pcm16(DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2")); raw_metrics[(sid,"Original")]=measure_loudness(x,sr)
    entry={"sample_id":sid,"dataset_id":inp["dataset_id"],"source":inp["source"],"display":inp["display"],"input_wav":inp["wav_file"],"input_sha256":inp["wav_sha256"],"sample_rate":sr,"frames":len(x),"raw_clipping":False,"abnormal_gain":False,"schemes":{},"notes":[]}
    for scheme,spec in SCHEMES.items():
        y=process(x,sr,spec["filter"]); stats=sample_stats(y,sr); loud=measure_loudness(y,sr); raw_path=raw_root/scheme/f"{sid}.wav"; export_wav(y,sr,0.0,raw_path); exported,esr=read_pcm16(raw_path); exported_loud=measure_loudness(exported,esr); clipped=int(np.count_nonzero(np.abs(exported)>=32767/32768)); gain=exported_loud["integrated_lufs"]-raw_metrics[(sid,"Original")]["integrated_lufs"]
        entry["schemes"][scheme]={"filter":spec["config"],"raw_file":str(raw_path.relative_to(ROOT)).replace("\\\\","/"),"raw_sha256":sha(raw_path),"raw_peak_dbfs":stats["sample_peak_dbfs"],"raw_rms_dbfs":stats["rms_dbfs"],"raw_clipped_samples":clipped,"raw_lufs":exported_loud["integrated_lufs"],"raw_true_peak_dbtp":exported_loud["true_peak_dbtp"],"lufs_gain_db":gain,"raw_bands":band_gains(x,exported,esr)}
        raw_metrics[(sid,scheme)]=exported_loud
        if clipped: entry["raw_clipping"]=True
        if abs(gain)>3.0: entry["abnormal_gain"]=True
    records.append(entry)
target=min(-18.0,*[m["integrated_lufs"]-3.2-m["true_peak_dbtp"] for m in raw_metrics.values()])
rng=random.Random(20261031); blind_key={}; listen_manifest=[]
for inp in inputs:
    sid=inp["sample_id"]; a_is_4=bool(rng.getrandbits(1)); blind_key[sid]={"version_A":"E2_4p0" if a_is_4 else "E2_8p0","version_B":"E2_8p0" if a_is_4 else "E2_4p0"}; entry=next(e for e in records if e["sample_id"]==sid)
    for scheme in (None,"E2_4p0","E2_8p0"):
        source_file=DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2") if scheme is None else raw_root/scheme/f"{sid}.wav"
        x,sr=read_pcm16(source_file); meter=measure_loudness(x,sr); gain=target-meter["integrated_lufs"]
        if scheme is None: dest=listen_root/f"{sid}_original.wav"
        else:
            is_a=(scheme=="E2_4p0" and a_is_4) or (scheme=="E2_8p0" and not a_is_4); dest=listen_root/f"{sid}_{'A' if is_a else 'B'}.wav"
        export_wav(x,sr,gain,dest); exported,esr=read_pcm16(dest); check=measure_loudness(exported,esr); rec={"file":str(dest.relative_to(ROOT)).replace("\\\\","/"),"gain_db":gain,"lufs":check["integrated_lufs"],"true_peak_dbtp":check["true_peak_dbtp"],"frames":len(exported),"sha256":sha(dest)}
        if scheme is None: entry["original_matched"]=rec
        else: entry["schemes"][scheme]["matched"]=rec; entry["schemes"][scheme]["label"]="A" if ((scheme=="E2_4p0" and a_is_4) or (scheme=="E2_8p0" and not a_is_4)) else "B"
    listen_manifest.append({"sample_id":sid,"source":inp["source"],"display":inp["display"],"original":entry["original_matched"]["file"],"A":entry["schemes"]["E2_4p0" if a_is_4 else "E2_8p0"]["matched"]["file"],"B":entry["schemes"]["E2_8p0" if a_is_4 else "E2_4p0"]["matched"]["file"]})
# Matched tilt preservation and A/B ranking.
matched=[]; ab=[]
for e in records:
    sid=e["sample_id"]; base=DATA/pathlib.Path(next(i["wav_file"] for i in inputs if i["sample_id"]==sid)).relative_to("test_dataset/f1_dataset_v2"); xom,som=read_pcm16(OUT/pathlib.Path(e["original_matched"]["file"]).relative_to("experiments/F1-EXP-003"))
    for scheme,s in e["schemes"].items():
        xm,sm=read_pcm16(OUT/pathlib.Path(s["matched"]["file"]).relative_to("experiments/F1-EXP-003")); mb=band_gains(xom,xm,sm); s["matched_tilt_db"]=mb["tilt"]; s["tilt_preserved_db"]=mb["tilt"]-s["raw_bands"]["tilt"]; matched.append({"sample_id":sid,"scheme":scheme,"raw_tilt":s["raw_bands"]["tilt"],"matched_tilt":mb["tilt"],"difference":s["tilt_preserved_db"],"pass":abs(s["tilt_preserved_db"])<1.0})
for m in listen_manifest:
    sid=m["sample_id"]; xo,so=read_pcm16(OUT/pathlib.Path(m["original"]).relative_to("experiments/F1-EXP-003")); xa,sa=read_pcm16(OUT/pathlib.Path(m["A"]).relative_to("experiments/F1-EXP-003")); xb,sb=read_pcm16(OUT/pathlib.Path(m["B"]).relative_to("experiments/F1-EXP-003")); ga=band_gains(xo,xa,sa)["high_4000_8000"]; gb=band_gains(xo,xb,sb)["high_4000_8000"]; a_s=blind_key[sid]["version_A"]; b_s=blind_key[sid]["version_B"]; exp_a=8.0 if a_s=="E2_8p0" else 4.0; exp_b=8.0 if b_s=="E2_8p0" else 4.0; ab.append({"sample_id":sid,"A_scheme":a_s,"B_scheme":b_s,"A_high_gain":ga,"B_high_gain":gb,"rank_pass":bool((ga>gb)==(exp_a>exp_b))})
# Page.
def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
by={m["sample_id"]:m for m in listen_manifest}; cards=[]
for inp in inputs:
    m=by[inp["sample_id"]]
    def src(p): return "audio/"+p.replace("\\","/").split("/")[-1]
    cards.append(f'<section class="card"><h3>{inp["sample_id"]} · {inp["source"]} · {esc(inp["display"])}</h3><p class="meta">最后一档强度诊断 · 原音 / A / B（A/B 随机）</p><div class="row"><label>原音</label><audio controls preload="metadata" src="{esc(src(m["original"]))}"></audio></div><div class="row"><label>版本 A</label><audio controls preload="metadata" src="{esc(src(m["A"]))}"></audio></div><div class="row"><label>版本 B</label><audio controls preload="metadata" src="{esc(src(m["B"]))}"></audio></div></section>')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P2.2 强度诊断</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:70px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}}</style><h1>F1-P2.2 · 最后一档强度</h1><div class="note"><b>听法：</b>每段先听原音，再听 A/B。请回答：有没有明显变化？更饱满？只是更亮？刺耳/发紧/不自然？是否仍像本人？A/B 的处理强度已随机隐藏；这是最后一档诊断，不代表最终美化参数。</div>{''.join(cards)}<p class="meta">四段诊断；Holdout 未处理。</p></html>'''
(OUT/"listening/index.html").write_text(page,encoding="utf-8")
(OUT/"listening/manifest.json").write_text(json.dumps({"target_lufs":target,"samples":listen_manifest},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"qa/blind_key.json").write_text(json.dumps({"seed":20261031,"mapping":blind_key},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
chain={"experiment":"F1-EXP-003","phase":"F1-P2.2","frequency_response":response,"linearity":linearity,"matched":matched,"ab_mapping":ab,"raw_clipping":any(e["raw_clipping"] for e in records),"abnormal_gain":any(e["abnormal_gain"] for e in records),"technical_pass":all(v["pass"] for v in linearity.values()) and all(x["pass"] for x in matched) and all(x["rank_pass"] for x in ab) and not any(e["raw_clipping"] or e["abnormal_gain"] for e in records)}
(OUT/"qa/chain_checks.json").write_text(json.dumps(chain,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"qa/qa_report.json").write_text(json.dumps({"experiment":"F1-EXP-003","phase":"F1-P2.2","target_lufs":target,"records":records,"chain":chain,"status":"pending_jade"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with open(OUT/"qa/levels.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["sample_id","scheme","raw_peak_dbfs","raw_rms_dbfs","raw_lufs","raw_true_peak_dbtp","raw_clipped","lufs_gain_db","raw_high_gain_db","raw_tilt_db","matched_lufs","matched_true_peak_dbtp","matched_gain_db","matched_tilt_db","tilt_preserved_db"])
    for e in records:
        o=e["original_matched"]; w.writerow([e["sample_id"],"Original","","",o["lufs"],o["true_peak_dbtp"],0,0,"","",o["lufs"],o["true_peak_dbtp"],o["gain_db"],"",""])
        for scheme,s in e["schemes"].items(): w.writerow([e["sample_id"],scheme,s["raw_peak_dbfs"],s["raw_rms_dbfs"],s["raw_lufs"],s["raw_true_peak_dbtp"],s["raw_clipped_samples"],s["lufs_gain_db"],s["raw_bands"]["high_4000_8000"],s["raw_bands"]["tilt"],s["matched"]["lufs"],s["matched"]["true_peak_dbtp"],s["matched"]["gain_db"],s["matched_tilt_db"],s["tilt_preserved_db"]])
params={"experiment":"F1-EXP-003","phase":"F1-P2.2","test":[i["sample_id"] for i in inputs],"sealed":SEALED,"schemes":{k:v["config"] for k,v in SCHEMES.items()},"target_lufs":target,"blind_seed":20261031,"code":"experiments/F1-EXP-003/run.py","no_stacking":True}
(OUT/"params.json").write_text(json.dumps(params,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary={"experiment":"F1-EXP-003","phase":"F1-P2.2","status":"pending_jade","test_count":len(inputs),"target_lufs":target,"technical_pass":chain["technical_pass"],"preview_page":"listening/index.html","blind_key":"qa/blind_key.json","raw_outputs":"outputs/processed_raw"}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"run_log.txt").write_text(f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-003\nphase: F1-P2.2\ntest: {', '.join(i['sample_id'] for i in inputs)}\nsealed: {', '.join(SEALED)}\ntarget_lufs: {target}\ntechnical_pass: {chain['technical_pass']}\nnote: original inputs not modified; holdout not processed\n",encoding="utf-8")
(OUT/"RESULT.md").write_text("# F1-EXP-003 结果（pending_jade）\n\nE2 +4/+8 dB 最后一档诊断；四段试听已生成，Holdout 未处理。技术检查见 qa/chain_checks.json；听感待用户试听。\n",encoding="utf-8")
print(json.dumps({"summary":summary,"chain_pass":chain["technical_pass"]},ensure_ascii=False,indent=1))