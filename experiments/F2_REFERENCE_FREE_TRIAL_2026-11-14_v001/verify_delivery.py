"""Verify the delivered files against the actual reloaded model prediction and source."""
import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from trial import HERE,sha,load_model,model_features,predict_array,render_command,dump


def main():
    torch.set_num_threads(2)
    record=json.loads((HERE/'inference_practice_v001/record.json').read_text(encoding='utf-8'))
    delivery=json.loads((HERE/'delivery_v001/delivery.json').read_text(encoding='utf-8'))
    assert sha(record['checkpoint'])==record['checkpoint_sha256']==delivery['checkpoint_sha256']
    assert sha(record['input'])==record['input_sha256']==delivery['source_sha256']
    assert sha(delivery['reference_display']['source'])==delivery['reference_display']['source_sha256']
    with np.load(io.BytesIO((HERE/'inference_practice_v001/inference.npz').read_bytes()),allow_pickle=False) as z:
        model=load_model(record['checkpoint']);f0=z['source_f0'];cc=z['source_cc']
        predicted=predict_array(model,model_features(model,f0))
        assert np.array_equal(predicted,z['prediction'])
        command=render_command(f0,predicted)
        agreement=(f0>0)&(cc>0)&(np.abs(1200*np.log2(np.maximum(f0,1)/np.maximum(cc,1)))<=50)
        command[~agreement]=0
        assert np.array_equal(command,z['command'])
        assert np.array_equal(z['source'][z['mask']==0],z['rendered'][z['mask']==0])
    for key,path in delivery['files'].items():
        actual=HERE/'delivery_v001'/path;data,rate=sf.read(actual,dtype='int16')
        qa=delivery['audio_checks'][key]
        assert sha(actual)==qa['sha256']
        assert len(data)==qa['samples'] and rate==qa['sample_rate']
        assert not np.any(np.abs(data.astype(np.int32))>=32767)
    assert delivery['technical_pass'] is True
    assert delivery['effectiveness_pass'] is False and delivery['diagnostic_only'] is True
    text=(HERE/'browser_test_run.log').read_text(encoding='utf-8-sig')
    browser=json.JSONDecoder().raw_decode(text.split('### Result',1)[1].lstrip())[0]
    assert browser['script_errors']==[] and len(browser['audio_versions'])==3
    feedback=(HERE/'output/playwright/test-feedback.txt').read_text(encoding='utf-8')
    assert '\n音准：拿不准\n' in feedback and '不代表用户反馈' in feedback
    result={'source_hash_unchanged':True,'reference_source_hash_unchanged':True,'checkpoint_hash_matches':True,
            'reloaded_prediction_exact':True,'reconstructed_application_curve_exact':True,
            'outside_generated_mask_exact':True,'delivered_audio_hashes_and_headers_verified':3,
            'browser':browser,'feedback_export_has_real_newlines':True,'technical_pass':True,
            'effectiveness_pass':False,'diagnostic_only':True,'listening':'pending_jade'}
    dump(HERE/'verification.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
