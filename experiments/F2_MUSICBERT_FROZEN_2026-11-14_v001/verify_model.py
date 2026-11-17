"""Independent numerical replay of official checkpoint operations, no HF layers."""
import argparse
import json
from pathlib import Path
import numpy as np
from adapter import HERE,dump,new_output,FrozenMusicBert


def reference_logits(model,ids):
    torch=model.torch
    import torch.nn.functional as F
    s=model.original_state;prefix='encoder.sentence_encoder.'
    tokens=torch.as_tensor(ids,dtype=torch.long);batch,flat=tokens.shape
    dimension=model.model.config.hidden_size;seq=flat//8
    def linear(x,name):return F.linear(x,s[name+'.weight'],s[name+'.bias'])
    def norm(x,name):return F.layer_norm(x,(dimension,),s[name+'.weight'],s[name+'.bias'],1e-5)
    with torch.inference_mode():
        x=F.embedding(tokens,s[prefix+'embed_tokens.weight']).view(batch,seq,dimension*8)
        x=linear(x,prefix+'downsampling.0')
        pos=torch.arange(2,seq+2)[None]
        x=norm(x+F.embedding(pos,s[prefix+'embed_positions.weight']),prefix+'emb_layer_norm')
        heads=model.model.config.num_attention_heads;head_dim=dimension//heads
        for i in range(model.model.config.num_hidden_layers):
            name=prefix+f'layers.{i}.'
            q,k,v=[linear(x,name+'self_attn.'+part+'_proj').view(batch,seq,heads,head_dim).transpose(1,2) for part in ('q','k','v')]
            attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(batch,seq,dimension)
            x=norm(x+linear(attn,name+'self_attn.out_proj'),name+'self_attn_layer_norm')
            x=norm(x+linear(F.gelu(linear(x,name+'fc1')),name+'fc2'),name+'final_layer_norm')
        x=linear(x,prefix+'upsampling.0').view(batch,flat,dimension)
        x=norm(F.gelu(linear(x,'encoder.lm_head.dense')),'encoder.lm_head.layer_norm')
        return F.linear(x,s['encoder.lm_head.weight'],s['encoder.lm_head.bias'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.out);model=FrozenMusicBert()
    notes=[{'start':i*.5,'end':(i+1)*.5,'pitch':p,'center':float(p),'eligible':True,'agreement':1.} for i,p in enumerate([60,62,64,65,67,65,64,62])]
    ids=model.codec.encode(notes);masked=np.tile(ids,(3,1))
    for row,idx in enumerate((1,3,6)):masked[row,(idx+1)*8+3]=model.codec.mask
    actual=model.logits(masked);expected=reference_logits(model,masked)
    error=(actual-expected).abs()
    maximum=float(error.max());mean=float(error.mean())
    if maximum>=1e-4:raise ValueError(f'HF adaptation disagrees with independent raw-checkpoint replay: {maximum}')
    p=model.probabilities(notes,[3])[0]
    altered=[{**n,'pitch':n['pitch']+5,'center':n['center']+5} for n in notes]
    altered[3]=notes[3]
    q=model.probabilities(altered,[3])[0]
    difference=float(np.abs(p-q).sum())
    if difference<1e-5:raise ValueError('Context has no measurable model effect')
    summary={'max_absolute_logit_error':maximum,'mean_absolute_logit_error':mean,
             'official_operations_replayed':True,'raw_checkpoint_only_in_reference':True,
             'hf_equivalence_pass':True,'context_probability_l1_difference':difference,
             'all_parameters_frozen':all(not p.requires_grad for p in model.model.parameters()),
             'weight_digest_unchanged':model.parameter_digest()==model.before,'training_steps':0}
    dump(out/'model_equivalence.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
