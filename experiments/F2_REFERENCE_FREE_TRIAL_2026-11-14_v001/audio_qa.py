"""Actual FFmpeg levels and independent pitch-following checks for delivered WAVs."""
import re
import subprocess
import numpy as np
import parselmouth


def levels(audio,rate):
    command=['ffmpeg','-hide_banner','-nostats','-f','f64le','-ar',str(rate),'-ac','1','-i','pipe:0',
             '-af','ebur128=peak=true','-f','null','-']
    result=subprocess.run(command,input=np.asarray(audio,dtype='<f8').tobytes(),capture_output=True,check=True)
    log=result.stderr.decode('utf-8',errors='replace')
    lufs=re.findall(r'I:\s*(-?[\d.]+)\s+LUFS',log)
    peak=re.findall(r'Peak:\s*(-?[\d.]+)\s+dBFS',log)
    if not lufs or not peak:raise ValueError('Cannot read actual FFmpeg measurements')
    return {'lufs':float(lufs[-1]),'true_peak_dbtp':float(peak[-1])}


def pitch(audio,rate):
    p=parselmouth.Sound(audio,rate).to_pitch_ac(time_step=.01,pitch_floor=65,pitch_ceiling=850,voicing_threshold=.5)
    f=p.selected_array['frequency'];m=np.full(len(f),np.nan);v=f>0;m[v]=1200*np.log2(f[v]/440)
    return p.xs(),m


def compare(source,processed,rate,grid,command):
    t,before=pitch(source,rate);u,after=pitch(processed,rate)
    if not np.array_equal(t,u):raise ValueError('Pitch analysis grids differ')
    shared=np.isfinite(before)&np.isfinite(after);union=np.isfinite(before)|np.isfinite(after)
    intended=np.interp(t,grid,command);active=shared&(np.abs(intended)>=20)
    error=np.abs((after-before-intended)[active])
    result={'voiced_overlap':float(shared.sum()/max(union.sum(),1)),
            'changed_pitch_frames':int(active.sum()),'command_error_median_cents':float(np.median(error)) if len(error) else None,
            'command_error_p90_cents':float(np.percentile(error,90)) if len(error) else None,
            'actual_shift_range_cents':[float(np.min((after-before)[active])),float(np.max((after-before)[active]))] if len(error) else [0.,0.]}
    result['pitch_technical_pass']=bool(len(error)>=5 and result['voiced_overlap']>=.95 and np.median(error)<5)
    return result
