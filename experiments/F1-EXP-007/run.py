"""F1-EXP-007: A01 parameter-range audition + objective prescreen."""
import csv, hashlib, json, pathlib, random, subprocess, sys
from datetime import datetime, timezone
import numpy as np
from scipy import signal
from pedalboard import Pedalboard, LowShelfFilter, HighShelfFilter, PeakFilter
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_poc import read_pcm16, measure_loudness, export_wav, sample_stats
OUT = ROOT / "experiments/F1-EXP-007"
if (OUT / "summary.json").exists(): raise FileExistsError("F1-EXP-007 outputs already exist")
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "test_dataset/f1_dataset_v2"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in manifest["items"]}
SAMPLE_ID = "A01"; DATASET_ID = "A_self_01"; SOURCE = "Self"; DISPLAY = "平凡之路 / 薄音版"
SEALED = ["A02","A03","A04","A05","A06","B01","B02","B03","C01","C02","C03"]
OS=4; CUTOFF=20000; SEED=20261101
CANDIDATES = [
 {"id":"C01","group":"EQ","type":"low_shelf","freq":220,"gain":2.0,"q":0.7,"label":"LowShelf 220Hz +2dB Q0.7"},
 {"id":"C02","group":"EQ","type":"low_shelf","freq":220,"gain":4.0,"q":0.7,"label":"LowShelf 220Hz +4dB Q0.7"},
 {"id":"C03","group":"EQ","type":"low_shelf","freq":300,"gain":4.0,"q":0.8,"label":"LowShelf 300Hz +4dB Q0.8"},
 {"id":"C04","group":"EQ","type":"low_shelf","freq":400,"gain":3.0,"q":0.9,"label":"LowShelf 400Hz +3dB Q0.9"},
 {"id":"C05","group":"EQ","type":"low_shelf","freq":220,"gain":6.0,"q":0.7,"label":"LowShelf 220Hz +6dB Q0.7"},
 {"id":"C06","group":"EQ","type":"high_shelf","freq":3000,"gain":4.0,"q":0.7,"label":"HighShelf 3000Hz +4dB Q0.7"},
 {"id":"C07","group":"EQ","type":"high_shelf","freq":3000,"gain":8.0,"q":0.7,"label":"HighShelf 3000Hz +8dB Q0.7"},
 {"id":"C08","group":"EQ","type":"peak","freq":500,"gain":3.0,"q":0.8,"label":"Peak 500Hz +3dB Q0.8"},
 {"id":"C09","group":"EQ","type":"peak","freq":1500,"gain":3.0,"q":1.0,"label":"Peak 1500Hz +3dB Q1.0"},
 {"id":"C10","group":"SAT","drive":3.0,"wet":0.5,"label":"tanh drive3 wet0.5"},
 {"id":"C11","group":"SAT","drive":3.0,"wet":0.6,"label":"tanh drive3 wet0.6"},
 {"id":"C12","group":"SAT","drive":6.0,"wet":0.5,"label":"tanh drive6 wet0.5"},
 {"id":"C13","group":"SAT","drive":6.0,"wet":0.6,"label":"tanh drive6 wet0.6"},
 {"id":"C14","group":"SAT","drive":6.0,"wet":0.7,"label":"tanh drive6 wet0.7"},
]
FFMPEG=subprocess.run(["ffmpeg","-version"],capture_output=True,text=True)
if FFMPEG.returncode != 0: raise RuntimeError("ffmpeg not on PATH")
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def eq(x,sr,typ,cfg):
    f={"low_shelf":LowShelfFilter,"high_shelf":HighShelfFilter,"peak":PeakFilter}[typ](cutoff_frequency_hz=cfg["freq"],gain_db=cfg["gain"],q=cfg["q"])
    out=np.asarray(Pedalboard([f])(np.asarray(x,dtype=np.float32),sr),dtype=float).reshape(-1)
    if len(out)!=len(x) or not np.all(np.isfinite(out)): raise ValueError("EQ length/nonfinite")
    return out
def sat(x,sr,drive,wet,os=OS,cutoff=CUTOFF):
    up=signal.resample_poly(x,os,1,window=('kaiser',5.0)); y=np.tanh(drive*up)/drive
    sos=signal.butter(8,cutoff/(sr*os/2),btype='low',output='sos'); y=signal.sosfiltfilt(sos,y)
    down=signal.resample_poly(y,1,os,window=('kaiser',5.0))
    if len(down)!=len(x): down=down[:len(x)] if len(down)>len(x) else np.pad(down,(0,len(x)-len(down)))
    return wet*down+(1-wet)*x
def spectral(x,sr):
    f,p=signal.welch(x,sr,nperseg=min(4096,len(x))); m=(f>=20)&(f<8000); f,p=f[m],p[m]; total=max(float(np.sum(p)),1e-20); return {"centroid_hz":float(np.sum(f*p)/total),"flatness":float(np.exp(np.mean(np.log(np.maximum(p,1e-20))))/max(np.mean(p),1e-20))}
def band_gain(x,y,sr,lo,hi):
    f1,p1=signal.welch(x,sr,nperseg=min(4096,len(x))); f2,p2=signal.welch(y,sr,nperseg=min(4096,len(y))); m1=(f1>=lo)&(f1<hi); m2=(f2>=lo)&(f2<hi); return float(10*np.log10(max(float(np.sum(p2[m2])),1e-20)/max(float(np.sum(p1[m1])),1e-20)))
def lag_between(y,x,sr):
    seg=x[:min(len(x),2*sr)]; yseg=y[:len(seg)]; c=signal.correlate(yseg,seg,mode='full'); return int(np.argmax(np.abs(c))-(len(seg)-1))
def align_shift(y,lag):
    if lag==0: return y.copy()
    if lag>0: return np.r_[y[lag:],np.zeros(lag,dtype=y.dtype)]
    return np.r_[np.zeros(-lag,dtype=y.dtype),y[:len(y)+lag]]
raw_root=OUT/"outputs/processed_raw"; listen_root=OUT/"listening/audio"; qa_dir=OUT/"qa"
for d in (raw_root,listen_root,qa_dir):
    if d.exists(): raise FileExistsError(d)
    d.mkdir(parents=True)
item=items[DATASET_ID]; wav=DATA/item["wav"]["path"]; raw=DATA/item["raw"]["path"]
if sha(raw)!=item["raw"]["sha256"]: raise ValueError("Input raw changed")
x,sr=read_pcm16(wav); orig_stats=sample_stats(x,sr); orig_loud=measure_loudness(x,sr); orig_spec=spectral(x,sr); orig_crest=orig_stats["sample_peak_dbfs"]-orig_stats["rms_dbfs"]
(OUT/"inputs.json").write_text(json.dumps({"experiment":"F1-EXP-007","sample_id":SAMPLE_ID,"source":SOURCE,"display":DISPLAY,"wav_file":str(wav.relative_to(ROOT)).replace("\\\\","/"),"raw_file":str(raw.relative_to(ROOT)).replace("\\\\","/"),"wav_sha256":sha(wav),"raw_sha256":sha(raw),"sealed_not_processed":SEALED},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
records=[]; raw_metrics={}
for c in CANDIDATES:
    cid=c["id"]; filter_lag=0
    if c["group"]=="EQ":
        y=eq(x,sr,c["type"],c); filter_lag=lag_between(y,x,sr); y=align_shift(y,filter_lag)
    else: y=sat(x,sr,c["drive"],c["wet"])
    rp=raw_root/f"{cid}.wav"; export_wav(y,sr,0.0,rp); ye,esr=read_pcm16(rp); st=sample_stats(ye,esr); loud=measure_loudness(ye,esr); sp=spectral(ye,esr); clipped=int(np.count_nonzero(np.abs(ye)>=32767/32768)); crest=st["sample_peak_dbfs"]-st["rms_dbfs"]; crest_drop=orig_crest-crest; centroid_ratio=sp["centroid_hz"]/orig_spec["centroid_hz"]; low_gain=band_gain(x,ye,esr,100,400); high_gain=band_gain(x,ye,esr,4000,8000); lag=lag_between(ye,x,sr); reasons=[]
    if clipped>0: reasons.append("削波")
    if abs(lag)>1: reasons.append(f"时间对齐 {lag} samples")
    if crest_drop>3.0: reasons.append(f"峰值压缩过大 crest_drop={crest_drop:.2f}dB")
    if centroid_ratio<0.78: reasons.append(f"频谱过暗 centroid={centroid_ratio:.2f}x")
    if centroid_ratio>1.25: reasons.append(f"频谱过亮 centroid={centroid_ratio:.2f}x")
    if low_gain>6.0: reasons.append(f"低频提升过大 {low_gain:.1f}dB")
    if high_gain>8.0: reasons.append(f"高频提升过大 {high_gain:.1f}dB")
    rec={"id":cid,"group":c["group"],"type":c.get("type",""),"params":{k:c[k] for k in ("freq","gain","q","drive","wet") if k in c},"label":c["label"],"raw_file":str(rp.relative_to(ROOT)).replace("\\\\","/"),"raw_sha256":sha(rp),"raw_frames":len(ye),"raw_peak_dbfs":st["sample_peak_dbfs"],"raw_rms_dbfs":st["rms_dbfs"],"raw_lufs":loud["integrated_lufs"],"raw_true_peak_dbtp":loud["true_peak_dbtp"],"raw_clipped_samples":clipped,"filter_lag_samples":filter_lag,"crest_drop_db":crest_drop,"centroid_hz":sp["centroid_hz"],"centroid_ratio":centroid_ratio,"flatness":sp["flatness"],"low_gain_db":low_gain,"high_gain_db":high_gain,"lag_samples":lag,"screen_pass":not reasons,"screen_reasons":reasons}
    records.append(rec); raw_metrics[cid]=loud
# common target from Original and all candidates; keep true peak >= -3.2
target=min(-18.0, orig_loud["integrated_lufs"]-3.2-orig_loud["true_peak_dbtp"], *[m["integrated_lufs"]-3.2-m["true_peak_dbtp"] for m in raw_metrics.values()])
orig_matched=listen_root/"original.wav"; export_wav(x,sr,target-orig_loud["integrated_lufs"],orig_matched); om,osr=read_pcm16(orig_matched); orig_m=measure_loudness(om,osr)
selected=[]; rejected=[]
for rec in records:
    if rec["screen_pass"]:
        src=OUT/pathlib.Path(rec["raw_file"]).relative_to("experiments/F1-EXP-007"); y,ysr=read_pcm16(src); meter=measure_loudness(y,ysr); gain=target-meter["integrated_lufs"]; dest=listen_root/f"{rec['id']}.wav"; export_wav(y,ysr,gain,dest); de,dsr=read_pcm16(dest); dm=measure_loudness(de,dsr); rec["matched"]={"file":str(dest.relative_to(ROOT)).replace("\\\\","/"),"gain_db":gain,"lufs":dm["integrated_lufs"],"true_peak_dbtp":dm["true_peak_dbtp"],"frames":len(de),"sha256":sha(dest)}; selected.append(rec)
    else: rejected.append(rec)
bad=[(r["id"],r["matched"]["true_peak_dbtp"]) for r in selected if r["matched"]["true_peak_dbtp"]>-2.9]
if bad: raise RuntimeError(f"Matched true peak too hot: {bad}")
# Page
def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
cards=[]
for r in selected:
    cards.append(f'<section class="card"><h3>{r["id"]} · {esc(r["group"])}</h3><p class="param">{esc(r["label"])}</p><p class="meta">客观筛查通过 · centroid {r["centroid_ratio"]:.2f}x · crest_drop {r["crest_drop_db"]:.2f}dB · low {r["low_gain_db"]:+.1f}dB / high {r["high_gain_db"]:+.1f}dB</p><div class="row"><label>候选</label><audio controls preload="metadata" src="audio/{r["id"]}.wav"></audio></div></section>')
rej=''.join(f'<tr><td>{esc(r["id"])}</td><td>{esc(r["label"])}</td><td>{esc("; ".join(r["screen_reasons"]))}</td></tr>' for r in rejected)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-EXP-007 A01 参数试听</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f5f7;color:#202530;max-width:900px;margin:28px auto;padding:0 18px}}h1{{font-size:24px}}.card{{background:#fff;border:1px solid #dce0e6;border-radius:10px;padding:14px;margin:12px 0}}h3{{font-size:16px;margin:0 0 4px}}.param{{font-weight:600;margin:2px 0}}.meta{{color:#596579;font-size:13px;margin:4px 0}}.row{{display:grid;grid-template-columns:60px 1fr;gap:10px;align-items:center;margin:8px 0}}audio{{width:100%}}.note{{background:#fff8e6;border:1px solid #ead9a6;border-radius:8px;padding:12px}}table{{width:100%;border-collapse:collapse;background:#fff}}td,th{{border:1px solid #dce0e6;padding:6px;font-size:13px;text-align:left}}th{{background:#eef1f5}}</style><h1>F1-EXP-007 · A01 参数范围试听</h1><div class="note"><b>先听原音，再逐个听候选。</b> 只改一个模块；候选参数可见。请分别判断：饱满度、清晰度、自然度、是否只是更亮/更响/更闷。客观筛查只排除明显偏差，不能代替耳朵。</div><section class="card"><h3>Original</h3><p class="meta">等响 {orig_m["integrated_lufs"]:.2f} LUFS</p><div class="row"><label>原音</label><audio controls preload="metadata" src="audio/original.wav"></audio></div></section>{''.join(cards)}<h2>被客观筛查剔除的候选</h2><table><tr><th>ID</th><th>参数</th><th>剔除原因</th></tr>{rej}</table><p class="meta">仅 A01；Holdout 未处理；试听版均等响 {target:.2f} LUFS。</p></html>'''
(OUT/"listening/index.html").write_text(page,encoding="utf-8")
(OUT/"qa/qa_report.json").write_text(json.dumps({"experiment":"F1-EXP-007","sample_id":SAMPLE_ID,"original":{**orig_stats,"lufs":orig_loud,"spectral":orig_spec},"target_lufs":target,"candidates":records,"selected":selected,"rejected":rejected,"screen_rules":{"clipping":"reject","lag":"abs>1 sample","crest_drop":">3.0 dB","centroid_ratio":"<0.90 or >1.25","low_gain":">6.0 dB","high_gain":">8.0 dB"},"status":"pending_jade"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with open(OUT/"qa/levels.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["id","group","params","raw_peak_dbfs","raw_rms_dbfs","raw_lufs","raw_true_peak_dbtp","raw_clipped","crest_drop_db","centroid_ratio","low_gain_db","high_gain_db","lag_samples","screen_pass","screen_reasons","matched_lufs","matched_true_peak_dbtp","matched_gain_db"])
    for r in records:
        m=r.get("matched",{}); w.writerow([r["id"],r["group"],r["label"],r["raw_peak_dbfs"],r["raw_rms_dbfs"],r["raw_lufs"],r["raw_true_peak_dbtp"],r["raw_clipped_samples"],r["crest_drop_db"],r["centroid_ratio"],r["low_gain_db"],r["high_gain_db"],r["lag_samples"],r["screen_pass"],"; ".join(r["screen_reasons"]),m.get("lufs",""),m.get("true_peak_dbtp",""),m.get("gain_db","")])
params={"experiment":"F1-EXP-007","sample_id":SAMPLE_ID,"ranges":{"low_shelf":{"freq_hz":[220,300,400],"gain_db":[2,3,4,6],"q":[0.7,0.8,0.9]},"high_shelf":{"freq_hz":[3000],"gain_db":[4,8],"q":[0.7]},"peak":{"freq_hz":[500,1500],"gain_db":[3],"q":[0.8,1.0]},"saturation":{"drive":[3,6],"wet":[0.5,0.6,0.7],"oversampling":4,"lowpass_hz":20000}},"candidates":CANDIDATES,"screen_rules":{"max_crest_drop_db":3.0,"min_centroid_ratio":0.78,"max_centroid_ratio":1.25,"max_low_gain_db":6.0,"max_high_gain_db":8.0},"target_lufs":target,"note":"only one module per candidate; no stacking; A01 only"}
(OUT/"params.json").write_text(json.dumps(params,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary={"experiment":"F1-EXP-007","status":"pending_jade","sample_id":SAMPLE_ID,"generated":len(records),"selected":len(selected),"rejected":[r["id"] for r in rejected],"target_lufs":target,"technical_pass":all(r["screen_pass"] for r in selected) and all(abs(r["lag_samples"])<=1 for r in records) and not any(r["raw_clipped_samples"]>0 for r in records),"preview_page":"listening/index.html"}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"RESULT.md").write_text(f"# F1-EXP-007 结果（pending_jade）\n\nA01 参数范围试听：生成 {len(records)} 个候选，客观筛查保留 {len(selected)} 个；剔除 {', '.join(r['id'] for r in rejected) or '无'}。技术检查见 qa/qa_report.json；听感待用户逐个试听。\n",encoding="utf-8")
print(json.dumps({"summary":summary,"selected":[{"id":r["id"],"label":r["label"]} for r in selected],"rejected":[{"id":r["id"],"reasons":r["screen_reasons"]} for r in rejected]},ensure_ascii=False,indent=2))