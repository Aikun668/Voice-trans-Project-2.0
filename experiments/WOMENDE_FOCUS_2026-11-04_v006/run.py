"""v006: one targeted tail revision and a shift-only diagnostic. No overwrite."""
import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT / '.venv-f2/Lib/site-packages'))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
old = load('v005', ROOT / 'experiments/WOMENDE_FOCUS_2026-11-04_v005/run.py')
v2 = load('v002', ROOT / 'experiments/WOMENDE_FOCUS_2026-11-03_v002/run.py')
master = load('master_v1', ROOT / 'experiments/F1_demo_chain_2026-11-02_v001/run.py')
V1 = ROOT / 'experiments/WOMENDE_FOCUS_2026-11-03_v001/work'
V5 = ROOT / 'experiments/WOMENDE_FOCUS_2026-11-04_v005/audio'
SRC = v2.SRC
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def apply_soft_tail(x, sr):
    mask, _ = old.decay_mask(x, sr)
    # 120 ms Gaussian sigma, shared gain across all frequencies; no EQ or high-pass.
    db = gaussian_filter1d(np.where(mask, -6., 0.), sigma=24, mode='nearest')
    t = np.arange(len(x)) / sr
    gain = np.interp(t, np.arange(len(db))*old.HOP_S, db)
    return x * 10**(gain/20), db
def sibilance(x, sr):
    n=int(.02*sr); nframes=len(x)//n
    def frames(a): return 10*np.log10(np.mean(a[:nframes*n].reshape(-1,n)**2,axis=1)+1e-12)
    full=frames(x); hi=frames(sosfilt(butter(4,[5000,9000],btype='bandpass',fs=sr,output='sos'),x))
    active=full>np.percentile(full,95)-30
    return round(float(np.percentile((hi-full)[active],99)),2)
def main():
    stage=HERE / 'work_output'
    if stage.exists() or (HERE/'audio').exists() or (HERE/'report.json').exists():
        raise SystemExit('输出已存在，拒绝覆盖')
    stage.mkdir()
    source_hash=sha(SRC)
    source,sr=sf.read(str(SRC),dtype='float64')
    key,srk=sf.read(str(V1/'thin_keyup.wav'),dtype='float64')
    tilt,srt=sf.read(str(V1/'c2_keyup_tilt_raw.wav'),dtype='float64')
    assert sr==srk==srt and len(source)==len(key)==len(tilt)
    baseline,_=sf.read(str(V5/'op_replay.wav'),dtype='float64')
    anchor,_=sf.read(str(V5/'anchor.wav'),dtype='float64')
    soft,gain=apply_soft_tail(tilt,sr)
    # Verify the archived pre-master input reproduces the reference, before using it.
    check, _=old.master_round(master, np.asarray(tilt,dtype=np.float32),sr)
    lock=old.corr(check,baseline)
    print('raw tilt → baseline correlation',lock,flush=True)
    if lock < .999: raise SystemExit('原始补谐波与基线不匹配')
    only,only_info=old.master_round(master,np.asarray(key,dtype=np.float32),sr)
    revised,soft_info=old.master_round(master,np.asarray(soft,dtype=np.float32),sr)
    items=[('anchor',anchor,0,'原音·同样响'),('baseline',baseline,310,'上次有力那版'),('key_only',only,310,'只升调·定位对照'),('soft_tail',revised,310,'有力版·缓和字尾')]
    f0move=old.median_shift_cents(tilt,soft,sr)
    rows=[]
    for name,y,target,label in items:
        path=stage/(name+'.wav')
        sf.write(str(path),y,sr,subtype='PCM_16')
        back, rate=sf.read(str(path),dtype='float64')
        lufs,tp=master.ebur(path)
        pitch=v2.pitch_delta(SRC,path,target)
        stats=master.level_stats(back,sr)
        decay=old.decay_lift(source,back,sr)
        failures=[]
        if len(back)!=len(source) or rate!=sr: failures.append('样本数或采样率不一致')
        if not np.isfinite(back).all(): failures.append('NaN/Inf')
        clipped=int(np.sum(np.abs(back)>=.9999))
        if clipped: failures.append('削波')
        if abs(lufs+13)>.4: failures.append('响度超限')
        if tp>-1: failures.append('真峰值超限')
        if pitch['median_cents'] is None or abs(pitch['median_cents']-target)>=5: failures.append('音高偏差')
        if pitch['voiced_overlap']<.95: failures.append('浊音重合低于0.95')
        if name=='soft_tail' and (f0move['median_cents'] is None or abs(f0move['median_cents'])>=5): failures.append('F1改动音高')
        rows.append(dict(id=name,name=label,file=name+'.wav',samples=len(back),sample_rate=rate,finite=bool(np.isfinite(back).all()),clipped_samples=clipped,lufs=lufs,true_peak_dbtp=tp,pitch_vs_source=pitch,decay_lift_db=round(decay,2),gap_floor_relative_db=round(stats['floor_p05_all']-stats['p95'],2),sibilance_5_9khz_p99_db=sibilance(back,sr),sha256=sha(path),failures=failures,technical_pass=not failures,listening='pending_jade'))
        print(name, lufs,tp,pitch['median_cents'],pitch['voiced_overlap'],failures,flush=True)
    times=np.arange(len(gain))*old.HOP_S; sel=(times>=11)&(times<12)
    oldmask,_=old.decay_mask(tilt,sr)
    oldgain=uniform_filter1d(np.where(oldmask,-10.,0.),size=6)
    report=dict(round=HERE.name,technical_pass=all(x['technical_pass'] for x in rows) and sha(SRC)==source_hash,listening='pending_jade',agent_cannot_hear=True,source=dict(path=str(SRC.relative_to(ROOT)),sha256=source_hash,hash_unchanged=sha(SRC)==source_hash,samples=len(source),sample_rate=sr),baseline_lock_correlation=lock,f1_pitch_change_vs_tilt=f0move,diagnosis_11_12s=dict(old_gain_range_db=[float(np.min(oldgain[sel])),float(np.max(oldgain[sel]))],old_steps_ge_1db_per_5ms=int(np.sum(np.abs(np.diff(oldgain[sel]))>=1)),revised_gain_range_db=[round(float(np.min(gain[sel])),2),round(float(np.max(gain[sel])),2)],revised_steps_ge_1db_per_5ms=int(np.sum(np.abs(np.diff(gain[sel]))>=1)),note='时间重合不证明听到的电音由字尾增益引起；升调和谐波处理仍可能参与。'),parameters=dict(f2k_cents=310,f2k_default='off',tail_depth_db=-6,tail_smoothing_gaussian_sigma_ms=120,tail_operation='全频同幅时间增益，无单独削低频或高通',no_eq_cut_below_250=True),master=dict(key_only=only_info,soft_tail=soft_info),versions=rows)
    (HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['technical_pass']: raise SystemExit('工程门未全过；音频留在 work_output，不生成试听页')
    stage.rename(HERE/'audio')
    print('technical_pass true',flush=True)
if __name__=='__main__': main()
