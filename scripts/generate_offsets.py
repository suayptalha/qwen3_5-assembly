#!/usr/bin/env python3
"""Generate fixed checkpoint tensor offsets for the freestanding assembly runner."""
import json
import hashlib
import struct
from pathlib import Path
root=Path(__file__).resolve().parents[1]
path=root/'model/model.safetensors-00001-of-00001.safetensors'
with path.open('rb') as f:
    digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest != '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696':
        raise SystemExit('Checkpoint SHA-256 mismatch; expected Qwen/Qwen3.5-0.8B')
    f.seek(0)
    n=struct.unpack('<Q',f.read(8))[0]
    data=json.loads(f.read(n))
base='model.language_model.'
def off(name): return 8+n+data[base+name]['data_offsets'][0]
lines=['.section .rodata', '.align 8', '.globl model_metadata', 'model_metadata:',
       f' .quad {off("embed_tokens.weight")}',f' .quad {off("norm.weight")}']
for i in range(24):
    p=f'layers.{i}.'
    pre=[p+'input_layernorm.weight',p+'post_attention_layernorm.weight']
    if i%4!=3:
        lp=p+'linear_attn.'
        names=[lp+x for x in ('in_proj_qkv.weight','in_proj_z.weight','in_proj_b.weight','in_proj_a.weight',
                               'conv1d.weight','A_log','dt_bias','norm.weight','out_proj.weight')]
    else:
        lp=p+'self_attn.'
        names=[lp+x for x in ('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight',
                               'q_norm.weight','k_norm.weight')]+[None]*3
    mlp=[p+'mlp.'+x+'_proj.weight' for x in ('gate','up','down')]
    lines += [' .quad '+str(off(n) if n else 0) for n in pre+names+mlp]
lines+=['.section .note.GNU-stack,"",@progbits']
(root/'build/model_offsets.S').write_text('\n'.join(lines)+'\n')
