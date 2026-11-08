#!/usr/bin/env python3
"""Text-only Qwen3.5 CPU prototype using assembly kernels and Python orchestration.

This is intentionally labelled a prototype: loading, tokenization, residuals,
position angles and full-attention orchestration are Python/NumPy, not assembly.
"""
import argparse
import ctypes
import json
import mmap
import struct
import sys
import time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--prompt', default='Hello! Introduce yourself briefly.')
ap.add_argument('--tokens', type=int, default=1)
ap.add_argument('--max-context', type=int, default=262144)
ap.add_argument('--raw-token-ids', help='comma-separated input IDs instead of chat tokenization')
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(root/'model', local_files_only=True)
if args.raw_token_ids:
    ids = [int(i) for i in args.raw_token_ids.split(',')]
else:
    ids = tok.apply_chat_template([{'role':'user','content':args.prompt}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)['input_ids']
if not 1 <= args.max_context <= 262144:
    raise SystemExit('--max-context must be between 1 and 262144')
if len(ids)+args.tokens > args.max_context:
    raise SystemExit('prompt and output exceed --max-context')
print(f'Input tokens: {len(ids)}', file=sys.stderr, flush=True)

f = open(root/'model/model.safetensors-00001-of-00001.safetensors','rb')
hlen = struct.unpack('<Q', f.read(8))[0]
hdr = json.loads(f.read(hlen))
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
def tensor(name):
    d = hdr[name]; dt = {'BF16':'<u2','F32':'<f4'}[d['dtype']]
    return np.ndarray(d['shape'], dtype=dt, buffer=mm, offset=8+hlen+d['data_offsets'][0])
def f32(name):
    a = tensor(name)
    return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype == np.uint16 else a.copy()
base='model.language_model.'
emb=tensor(base+'embed_tokens.weight')
final_norm=f32(base+'norm.weight')
lib=ctypes.CDLL(str(root/'build/libkernels.so'))
P,N,F=ctypes.c_void_p,ctypes.c_size_t,ctypes.c_float
lib.matvec_bf16_f32.argtypes=[P,P,P,N,N]
lib.mlp_bf16.argtypes=[P]*5+[N,N,P]
lib.rmsnorm_f32.argtypes=[P,P,P,N,F,F]
lib.model_step.argtypes=[N,P,N]
lib.model_step.restype=ctypes.c_int64
lib.decoder_layer.argtypes=[P]*6
lib.delta_layer.argtypes=[P,P,P,P,P]
lib.full_attention_layer.argtypes=[P,P,P,P,P]
lib.sigmoid_f32.argtypes=[P,P,N]
lib.rope_f32.argtypes=[P,P,P,N]
lib.attention_f32.argtypes=[P]*4+[N,N,P]

def mat(w,x):
    y=np.empty(w.shape[0],np.float32)
    lib.matvec_bf16_f32(w.ctypes.data,x.ctypes.data,y.ctypes.data,w.shape[0],w.shape[1])
    return y
def norm(x,w):
    y=np.empty_like(x)
    lib.rmsnorm_f32(x.ctypes.data,w.ctypes.data,y.ctypes.data,x.size,1e-6,1.)
    return y
def sigmoid(x):
    x=np.ascontiguousarray(x,dtype=np.float32); y=np.empty_like(x)
    lib.sigmoid_f32(x.ctypes.data,y.ctypes.data,x.size)
    return y

layers=[]
for i in range(24):
    p=base+f'layers.{i}.'
    d={'pre':f32(p+'input_layernorm.weight'), 'post':f32(p+'post_attention_layernorm.weight'),
       'mlp':[tensor(p+'mlp.'+n+'_proj.weight') for n in ('gate','up','down')]}
    if i%4 != 3:
        prefix=p+'linear_attn.'
        names=['in_proj_qkv.weight','in_proj_z.weight','in_proj_b.weight','in_proj_a.weight',
               'conv1d.weight','A_log','dt_bias','norm.weight','out_proj.weight']
        arrays=[f32(prefix+n) if n in ('A_log','dt_bias','norm.weight') else tensor(prefix+n) for n in names]
        d['arrays']=arrays; d['wp']=(P*9)(*(a.ctypes.data for a in arrays))
        d['conv']=np.zeros((6144,4),np.float32); d['state']=np.zeros((16,128,128),np.float32)
        d['sp']=(P*2)(d['conv'].ctypes.data,d['state'].ctypes.data)
        d['scratch']=np.empty(16800,np.float32)
    else:
        prefix=p+'self_attn.'
        for n in ('q_proj','k_proj','v_proj','o_proj'): d[n]=tensor(prefix+n+'.weight')
        d['qn']=f32(prefix+'q_norm.weight'); d['kn']=f32(prefix+'k_norm.weight')
        d['keys']=np.zeros((2,args.max_context,256),np.float32)
        d['values']=np.zeros((2,args.max_context,256),np.float32)
        d['att_arrays']=[d[n] for n in ('q_proj','k_proj','v_proj','o_proj','qn','kn')]
        d['att_wp']=(P*6)(*(a.ctypes.data for a in d['att_arrays']))
        d['att_scratch']=np.empty(7424,np.float32)
        d['scores']=np.empty(args.max_context,np.float32)
    variant=d['wp'] if i%4 != 3 else d['att_wp']
    d['descriptor']=(P*7)(d['pre'].ctypes.data,d['post'].ctypes.data,ctypes.addressof(variant),
                            *(w.ctypes.data for w in d['mlp']), 0 if i%4 != 3 else 1)
    layers.append(d)
mlp_scratch=np.empty(2*3584,np.float32)
decoder_work=np.empty(2048,np.float32)
layer_ptrs=(P*24)(*(ctypes.addressof(d['descriptor']) for d in layers))
scratch_ptrs=(P*24)(*(d['scratch' if i%4 != 3 else 'att_scratch'].ctypes.data for i,d in enumerate(layers)))
model_x=np.empty(1024,np.float32)
model_logits=np.empty(248320,np.float32)
inv_freq=(1.0/(10000000.0**(np.arange(0,64,2,dtype=np.float32)/64))).astype(np.float32)

def forward(token, pos, need_logits=True):
    cs=np.cos(pos*inv_freq).astype(np.float32); sn=np.sin(pos*inv_freq).astype(np.float32)
    states=[]
    for i,d in enumerate(layers):
        if i%4 != 3:
            states.append(d['sp'])
        else:
            states.append((P*8)(d['keys'][0].ctypes.data,d['keys'][1].ctypes.data,
                                d['values'][0].ctypes.data,d['values'][1].ctypes.data,
                                cs.ctypes.data,sn.ctypes.data,pos+1,d['scores'].ctypes.data))
    state_ptrs=(P*24)(*(ctypes.addressof(st) for st in states))
    cfg=(P*9)(emb.ctypes.data,final_norm.ctypes.data,ctypes.addressof(layer_ptrs),
              ctypes.addressof(state_ptrs),ctypes.addressof(scratch_ptrs),mlp_scratch.ctypes.data,
              decoder_work.ctypes.data,model_x.ctypes.data,model_logits.ctypes.data)
    return int(lib.model_step(token,ctypes.addressof(cfg),int(need_logits)))

start=time.perf_counter()
for pos,token in enumerate(ids):
    logits=forward(token,pos,need_logits=(pos == len(ids)-1))
    print(f'prefill {pos+1}/{len(ids)}: {time.perf_counter()-start:.1f}s',file=sys.stderr,flush=True)
out=[]
for step in range(args.tokens):
    token=logits; out.append(token)
    print(f'token {step+1}: {token} {tok.decode([token])!r}',file=sys.stderr,flush=True)
    if token in (248044,248046): break
    if step+1<args.tokens: logits=forward(token,len(ids)+step)
print(json.dumps({'input_ids':ids,'generated_ids':out,'text':tok.decode(out),
                  'seconds':time.perf_counter()-start},ensure_ascii=False,indent=2))
