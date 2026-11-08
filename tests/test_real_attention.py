"""Real Qwen3.5 attention layer vs Transformers with sequential KV cache."""
import ctypes
import json
import struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
path=root/'model/model.safetensors-00001-of-00001.safetensors'
with path.open('rb') as f:
    hn=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(hn))
    def weight(name):
        info=hdr['model.language_model.layers.3.self_attn.'+name]
        f.seek(8+hn+info['data_offsets'][0])
        raw=np.fromfile(f,'<u2',count=int(np.prod(info['shape']))).reshape(info['shape'])
        return raw,(raw.astype(np.uint32)<<16).view(np.float32)
    names=['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','q_norm.weight','k_norm.weight']
    weights={n:weight(n) for n in names}
config=AutoConfig.from_pretrained(root/'model',local_files_only=True).text_config
config._attn_implementation='eager'
ref=Qwen3_5Attention(config,layer_idx=3).eval()
ref.load_state_dict({n:torch.from_numpy(w[1]) for n,w in weights.items()})
rng=np.random.default_rng(43)
x=rng.standard_normal((1,4,1024),dtype=np.float32)*.2
with torch.inference_mode():
    pos=torch.arange(4).reshape(1,4)
    cos,sin=Qwen3_5TextRotaryEmbedding(config)(torch.from_numpy(x),pos)
    mask=torch.full((1,1,4,4),float('-inf')).triu(diagonal=1)
    expected=ref(torch.from_numpy(x),(cos,sin),mask)[0].numpy().reshape(4,1024)
lib=ctypes.CDLL(str(root/'build/libkernels.so'))
P=ctypes.c_void_p
lib.full_attention_layer.argtypes=[P]*5
arrays=[weights[n][1 if 'norm' in n else 0] for n in names]
wptrs=(P*6)(*(a.ctypes.data for a in arrays))
keys=np.zeros((2,4,256),np.float32); values=np.zeros_like(keys)
scratch=np.empty(7424,np.float32); scores=np.empty(4,np.float32)
out=[]
for i in range(4):
    c=np.ascontiguousarray(cos.numpy()[0,i,:32]); s=np.ascontiguousarray(sin.numpy()[0,i,:32])
    state=(P*8)(keys[0].ctypes.data,keys[1].ctypes.data,values[0].ctypes.data,values[1].ctypes.data,
                c.ctypes.data,s.ctypes.data,i+1,scores.ctypes.data)
    y=np.empty(1024,np.float32)
    lib.full_attention_layer(x[0,i].ctypes.data,y.ctypes.data,ctypes.addressof(wptrs),ctypes.addressof(state),scratch.ctypes.data)
    out.append(y)
out=np.stack(out)
np.testing.assert_allclose(out,expected,rtol=5e-3,atol=5e-6)
print(f'Entire full-attention layer in assembly, four tokens: passed; max abs error {np.max(np.abs(out-expected)):.3g}')
