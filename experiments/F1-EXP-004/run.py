"""F1-EXP-004: P3A soft harmonic saturation (single module, wet/dry, anti-aliased)."""
import csv, hashlib, json, pathlib, platform, random, subprocess, sys
from datetime import datetime, timezone
import numpy as np
from scipy import signal
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats
OUT = ROOT / "experiments/F1-EXP-004"
if (OUT / "summary.json").exists(): raise FileExistsError("F1-EXP-004 outputs already exist")
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}
TEST = [("A01","A_self_01","Self","平凡之路 / 薄音版"),("A05","A_gt_02","GTSinger","女A-002（来源未保留）")]
SEALED = ["A02","A03","A04","A06","B01","B02","B03","C01","C02","C03"]
SCHEMES = {"Light":{"drive":1.5,"wet":0.5},"Medium":{"drive":3.0,"wet":0.5},"Strong":{"drive":6.0,"wet":0.5}}
OS=4; CUTOFF=20000; SEED=20261031
FFMPEG = subprocess.run(["ffmpeg","-version"],capture_output=True,text=True)
if FFMPEG.returncode != 0: raise RuntimeError("ffmpeg not on PATH")
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def db(v): return float(20*np.log10(max(float(v),1e-12)))
def sat(x,sr,drive,wet=0.5,os=OS,cutoff=CUTOFF):
    up=signal.resample_poly(x,os,1,window=('kaiser',5.0))
    y=np.tanh(drive*up)/drive
    sos=signal.butter(8,cutoff/(sr*os/2),btype='low',output='sos')
    y=signal.sosfiltfilt(sos,y)
    down=signal.resample_poly(y,1,os,window=('kaiser',5.0))
    if len(down)!=len(x):
        down=down[:len(x)] if len(down)>len(x) else np.pad(down,(0,len(x)-len(down)))
    return wet*down+(1-wet)*x
def spectral(x,sr):
    f,p=signal.welch(x,sr,nperseg=min(4096,len(x))); m=(f>=20)&(f<8000); f,p=f[m],p[m]; total=max(float(np.sum(p)),1e-20); centroid=float(np.sum(f*p)/total); flat=float(np.exp(np.mean(np.log(np.maximum(p,1e-20))))/max(np.mean(p),1e-20)); return {"centroid_hz":centroid,"flatness":flat}
raw_root=OUT/"outputs/processed_raw"; listen_root=OUT/"listening/audio"; qa_dir=OUT/"qa"
for d in (raw_root/"Light",raw_root/"Medium",raw_root/"Strong",listen_root,qa_dir):
    if d.exists(): raise FileExistsError(d)
    d.mkdir(parents=True)
inputs=[]
for sid,did,source,display in TEST:
    item=items[did]; wav=DATA/item["wav"]["path"]; raw=DATA/item["raw"]["path"]
    if sha(raw)!=item["raw"]["sha256"]: raise ValueError(f"Input raw changed: {raw}")
    inputs.append({"sample_id":sid,"dataset_id":did,"source":source,"display":display,"wav_file":str(wav.relative_to(ROOT)).replace("\\\\","/"),"raw_file":str(raw.relative_to(ROOT)).replace("\\\\","/"),"wav_sha256":sha(wav),"raw_sha256":sha(raw)})
(OUT/"inputs.json").write_text(json.dumps({"experiment":"F1-EXP-004","phase":"F1-P3A","test":inputs,"sealed_not_processed":SEALED},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
records=[]; raw_metrics={}; chain_align=[]
for inp in inputs:
    sid=inp["sample_id"]; x,sr=read_pcm16(DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2")); raw_metrics[(sid,"Original")]=measure_loudness(x,sr); entry={"sample_id":sid,"dataset_id":inp["dataset_id"],"source":inp["source"],"display":inp["display"],"input_wav":inp["wav_file"],"input_sha256":inp["wav_sha256"],"sample_rate":sr,"frames":len(x),"schemes":{},"notes":[]}
    for scheme,cfg in SCHEMES.items():
        y=sat(x,sr,cfg["drive"],cfg["wet"]); stats=sample_stats(y,sr); loud=measure_loudness(y,sr); raw_path=raw_root/scheme/f"{sid}.wav"; export_wav(y,sr,0.0,raw_path); exported,esr=read_pcm16(raw_path); exported_loud=measure_loudness(exported,esr); clipped=int(np.count_nonzero(np.abs(exported)>=32767/32768)); gain=exported_loud["integrated_lufs"]-raw_metrics[(sid,"Original")]["integrated_lufs"]
        # alignment on first 2 s; resample_poly is zero-phase in practice, verify lag.
        seg=x[:min(len(x),2*sr)]; yseg=exported[:len(seg)]; corr=signal.correlate(yseg,seg,mode='full'); lag=int(np.argmax(np.abs(corr))-(len(seg)-1)); chain_align.append({"sample_id":sid,"scheme":scheme,"lag_samples":lag})
        entry["schemes"][scheme]={"drive":cfg["drive"],"wet":cfg["wet"],"raw_file":str(raw_path.relative_to(ROOT)).replace("\\\\","/"),"raw_sha256":sha(raw_path),"raw_frames":len(exported),"raw_peak_dbfs":stats["sample_peak_dbfs"],"raw_rms_dbfs":stats["rms_dbfs"],"raw_clipped_samples":clipped,"raw_lufs":exported_loud["integrated_lufs"],"raw_true_peak_dbtp":exported_loud["true_peak_dbtp"],"lufs_gain_db":gain,"spectral":spectral(exported,esr),"lag_samples":lag}
        raw_metrics[(sid,scheme)]=exported_loud
    records.append(entry)
target=min(-18.0,*[m["integrated_lufs"]-3.2-m["true_peak_dbtp"] for m in raw_metrics.values()])
rng=random.Random(SEED); blind_key={}; listen_manifest=[]
for inp in inputs:
    sid=inp["sample_id"]; entry=next(e for e in records if e["sample_id"]==sid); names=rng.sample(list(SCHEMES.keys()),3); mapping={code:name for code,name in zip(["A","B","C"],names)}; blind_key[sid]=mapping
    for label,scheme in ((None,"Original"),("A",mapping["A"]),("B",mapping["B"]),("C",mapping["C"])):
        source_file=DATA/pathlib.Path(inp["wav_file"]).relative_to("test_dataset/f1_dataset_v2") if scheme=="Original" else raw_root/scheme/f"{sid}.wav"; x,sr=read_pcm16(source_file); meter=measure_loudness(x,sr); gain=target-meter["integrated_lufs"]
        dest=listen_root/f"{sid}_original.wav" if label is None else listen_root/f"{sid}_{label}.wav"
        export_wav(x,sr,gain,dest); exported,esr=read_pcm16(dest); check=measure_loudness(exported,esr); rec={"file":str(dest.relative_to(ROOT)).replace("\\\\","/"),"gain_db":gain,"lufs":check["integrated_lufs"],"true_peak_dbtp":check["true_peak_dbtp"],"frames":len(exported),"sha256":sha(dest)}
        if scheme=="Original": entry["original_matched"]=rec
        else: entry["schemes"][scheme]["matched"]=rec; entry["schemes"][scheme]["label"]=label
    listen_manifest.append({"sample_id":sid,"source":inp["source"],"display":inp["display"],"original":entry["original_matched"]["file"],"A":entry["schemes"][mapping["A"]]["matched"]["file"],"B":entry["schemes"][mapping["B"]]["matched"]["file"],"C":entry["schemes"][mapping["C"]]["matched"]["file"]})
bad_peaks=[(e["sample_id"],k,v["matched"]["true_peak_dbtp"]) for e in records for k,v in e["schemes"].items() if v["matched"]["true_peak_dbtp"]>-2.9]
if bad_peaks: raise RuntimeError(f"Matched true peak too hot: {bad_peaks}")
# Page.
def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
by={m["sample_id"]:m for m in listen_manifest}; cards=[]
for inp in inputs:
    m=by[inp["sample_id"]]
    def src(p): return "audio/"+p.replace("\\","/").split("/")[-1]
    cards.append(f'<section class="card"><h3>{inp["sample_id"]} · {inp["source"]} · {esc(inp["display"])}</h3><p class="meta">P3A 谐波饱和 · 原音 / 版本 A/B/C（强度随机隐藏）</p><div class="row"><label>原音</label><audio controls preload="metadata" src="{esc(src(m["original"]))}"></audio></div><div class="row"><label>版本 A</label><audio controls preload="metadata" src="{esc(src(m["A"]))}"></audio></div><div class="row"><label>版本 B</label><audio controls preload="metadata" src="{esc(src(m["B"]))}"></audio></div><div class="row"><label>版本 C</label><audio controls preload="metadata" src="{esc(src(m["C"]))}"></audio></div></section>')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-P3A 谐波饱和试听</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:860px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:16px;margin:14px 0}}h3{{font-size:17px;margin:0 0 8px}}.meta{{color:#596579;margin:4px 0}}.row{{display:grid;grid-template-columns:70px 1fr;gap:10px;align-items:center;margin:8px 0}}label{{color:#596579;font-size:14px}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}}</style><h1>F1-P3A · 谐波饱和试听</h1><div class="note"><b>听法：</b>先听原音，再听 A/B/C。请分别回答：饱满度是否提升？是否只是更亮/更响？是否自然、像本人？是否刺耳、沙哑或失真？三个处理强度已随机隐藏；本轮只测饱和，没有 EQ/压缩/混响。</div>{''.join(cards)}<p class="meta">仅 A01/A05；Holdout 未处理。</p></html>'''
(OUT/"listening/index.html").write_text(page,encoding="utf-8")
(OUT/"listening/manifest.json").write_text(json.dumps({"target_lufs":target,"samples":listen_manifest},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"qa/blind_key.json").write_text(json.dumps({"seed":SEED,"mapping":blind_key},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
chain={"experiment":"F1-EXP-004","phase":"F1-P3A","oversampling":OS,"lowpass_hz":CUTOFF,"wet":0.5,"drives":{k:v["drive"] for k,v in SCHEMES.items()},"alignment":chain_align,"alignment_pass":all(abs(x["lag_samples"])<=1 for x in chain_align),"raw_clipping":any(e["schemes"][k]["raw_clipped_samples"]>0 for e in records for k in SCHEMES),"ab_mapping_seed":SEED,"technical_pass":all(abs(x["lag_samples"])<=1 for x in chain_align) and not any(e["schemes"][k]["raw_clipped_samples"]>0 for e in records for k in SCHEMES)}
(OUT/"qa/chain_checks.json").write_text(json.dumps(chain,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"qa/qa_report.json").write_text(json.dumps({"experiment":"F1-EXP-004","phase":"F1-P3A","target_lufs":target,"records":records,"chain":chain,"status":"pending_jade"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with open(OUT/"qa/levels.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["sample_id","scheme","drive","wet","raw_peak_dbfs","raw_rms_dbfs","raw_lufs","raw_true_peak_dbtp","raw_clipped","lufs_gain_db","centroid_hz","flatness","matched_lufs","matched_true_peak_dbtp","matched_gain_db","lag_samples"])
    for e in records:
        o=e["original_matched"]; w.writerow([e["sample_id"],"Original","","", "", "",o["lufs"],o["true_peak_dbtp"],0,0,"","",o["lufs"],o["true_peak_dbtp"],o["gain_db"],0])
        for scheme,s in e["schemes"].items(): w.writerow([e["sample_id"],scheme,s["drive"],s["wet"],s["raw_peak_dbfs"],s["raw_rms_dbfs"],s["raw_lufs"],s["raw_true_peak_dbtp"],s["raw_clipped_samples"],s["lufs_gain_db"],s["spectral"]["centroid_hz"],s["spectral"]["flatness"],s["matched"]["lufs"],s["matched"]["true_peak_dbtp"],s["matched"]["gain_db"],s["lag_samples"]])
params={"experiment":"F1-EXP-004","phase":"F1-P3A","test":[i["sample_id"] for i in inputs],"sealed":SEALED,"algorithm":"soft tanh: tanh(drive*x)/drive; 4x oversampling; 20kHz lowpass before downsample; wet=0.5","schemes":SCHEMES,"target_lufs":target,"blind_seed":SEED,"no_stacking":True}
(OUT/"params.json").write_text(json.dumps(params,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary={"experiment":"F1-EXP-004","phase":"F1-P3A","status":"pending_jade","test_count":len(inputs),"target_lufs":target,"technical_pass":chain["technical_pass"],"preview_page":"listening/index.html","blind_key":"qa/blind_key.json","raw_outputs":"outputs/processed_raw"}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"run_log.txt").write_text(f"created_utc: {datetime.now(timezone.utc).isoformat()}\nexperiment: F1-EXP-004\nphase: F1-P3A\ntest: {', '.join(i['sample_id'] for i in inputs)}\nsealed: {', '.join(SEALED)}\ntarget_lufs: {target}\ntechnical_pass: {chain['technical_pass']}\nnote: original inputs not modified; holdout not processed; only saturation\n",encoding="utf-8")
(OUT/"RESULT.md").write_text("# F1-EXP-004 结果（pending_jade）\n\nSoft Saturation 三强度（Light/Medium/Strong）已生成；A01/A05；Holdout 未处理。技术检查见 qa/chain_checks.json；听感待用户试听。\n",encoding="utf-8")
print(json.dumps({"summary":summary,"chain_pass":chain["technical_pass"],"alignment":chain_align},ensure_ascii=False,indent=1))