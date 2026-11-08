"""Compare a real Gated DeltaNet layer: assembly kernels + Python glue vs Transformers.

This is an integration test, not a complete assembly runtime. The kernel calls
and the individual layer weights are real, while Python connects the stages.
"""
import ctypes
import json
import struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
path = root / 'model/model.safetensors-00001-of-00001.safetensors'
if not path.exists():
    path = Path(str(path) + '.part')
with path.open('rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    def weight(name):
        info = header['model.language_model.layers.0.linear_attn.' + name]
        assert 8 + header_size + info['data_offsets'][1] <= path.stat().st_size, name
        f.seek(8 + header_size + info['data_offsets'][0])
        if info['dtype'] == 'BF16':
            raw = np.fromfile(f, '<u2', count=int(np.prod(info['shape']))).reshape(info['shape'])
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            raw = np.fromfile(f, '<f4', count=int(np.prod(info['shape']))).reshape(info['shape'])
            f32 = raw
        return raw, f32
    names = ['dt_bias', 'A_log', 'conv1d.weight', 'norm.weight', 'out_proj.weight',
             'in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_b.weight', 'in_proj_a.weight']
    weights = {n: weight(n) for n in names}

config = AutoConfig.from_pretrained(root / 'model', local_files_only=True).text_config
ref = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
ref.load_state_dict({n: torch.from_numpy(data[1]) for n, data in weights.items()})
rng = np.random.default_rng(7)
xseq = rng.standard_normal((3, 1024), dtype=np.float32) * .2
with torch.inference_mode():
    expected = ref(torch.from_numpy(xseq.reshape(1,3,-1))).numpy().reshape(3, -1)

lib = ctypes.CDLL(str(root / 'build/libkernels.so'))
P, N, F = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float
lib.matvec_bf16_f32.argtypes = [P,P,P,N,N]
lib.causal_conv4_bf16.argtypes = [P,P,P,P,N]
lib.silu_f32.argtypes = [P,P,N]
lib.sigmoid_f32.argtypes = [P,P,N]
lib.softplus_f32.argtypes = [P,P,N]
lib.exp_f32.argtypes = [F]
lib.exp_f32.restype = F
lib.l2norm_f32.argtypes = [P,P,N,F,F]
lib.rmsnorm_f32.argtypes = [P,P,P,N,F,F]
lib.delta_step.argtypes = [P]*5 + [N,N,F,F]

def proj(name, x):
    w = weights[name + '.weight'][0]
    y = np.empty(w.shape[0], np.float32)
    lib.matvec_bf16_f32(w.ctypes.data, x.ctypes.data, y.ctypes.data, w.shape[0], w.shape[1])
    return y

def activation(fn, x):
    y = np.empty_like(x)
    fn(x.ctypes.data, y.ctypes.data, x.size)
    return y

convstate = np.zeros((6144,4), np.float32)
states = np.zeros((16,128,128), np.float32)
outseq = []
for x in xseq:
    qkv = proj('in_proj_qkv', x)
    z = proj('in_proj_z', x)
    b = activation(lib.sigmoid_f32, proj('in_proj_b', x))
    a = proj('in_proj_a', x)
    plus = activation(lib.softplus_f32, a + weights['dt_bias'][1])
    g = np.array([-lib.exp_f32(float(log_a)) * sp for log_a, sp in zip(weights['A_log'][1], plus)], dtype=np.float32)
    conv = np.empty(6144, np.float32)
    conv_w = weights['conv1d.weight'][0]
    lib.causal_conv4_bf16(convstate.ctypes.data, conv_w.ctypes.data, qkv.ctypes.data, conv.ctypes.data, 6144)
    conv = activation(lib.silu_f32, conv)
    q, k, v = np.split(conv, 3)
    core = np.empty(2048, np.float32)
    for h in range(16):
        sl = slice(h*128, (h+1)*128)
        qn, kn = np.empty(128,np.float32), np.empty(128,np.float32)
        lib.l2norm_f32(q[sl].ctypes.data, qn.ctypes.data, 128, 1e-6, 1/np.sqrt(128))
        lib.l2norm_f32(k[sl].ctypes.data, kn.ctypes.data, 128, 1e-6, 1.)
        lib.delta_step(states[h].ctypes.data, qn.ctypes.data, kn.ctypes.data, v[sl].ctypes.data,
                       core[sl].ctypes.data, 128, 128, lib.exp_f32(float(g[h])), b[h])
        norm = np.empty(128,np.float32)
        nw = weights['norm.weight'][1]
        lib.rmsnorm_f32(core[sl].ctypes.data, nw.ctypes.data, norm.ctypes.data, 128, 1e-6, 0.)
        core[sl] = norm * activation(lib.silu_f32, z[sl])
    outseq.append(proj('out_proj', core))
outseq = np.stack(outseq)
np.testing.assert_allclose(outseq, expected, rtol=5e-3, atol=5e-6)
print(f'Real layer-0 Gated DeltaNet, three tokens: passed; max abs error {np.max(np.abs(outseq-expected)):.3g}')

# Exercise the same entire layer in a single assembly call per token.
keys = ['in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_b.weight', 'in_proj_a.weight',
        'conv1d.weight', 'A_log', 'dt_bias', 'norm.weight', 'out_proj.weight']
arrays = [weights[n][1 if n in ('A_log','dt_bias','norm.weight') else 0] for n in keys]
wptrs = (P * len(arrays))(*(a.ctypes.data for a in arrays))
convstate = np.zeros((6144,4), np.float32)
states = np.zeros((16,128,128), np.float32)
sptrs = (P * 2)(convstate.ctypes.data, states.ctypes.data)
scratch = np.empty(16800, np.float32)
lib.delta_layer.argtypes = [P,P,P,P,P]
output = []
for x in xseq:
    y = np.empty(1024, np.float32)
    lib.delta_layer(x.ctypes.data, y.ctypes.data, ctypes.addressof(wptrs),
                    ctypes.addressof(sptrs), scratch.ctypes.data)
    output.append(y)
output = np.stack(output)
np.testing.assert_allclose(output, expected, rtol=5e-3, atol=5e-6)
print(f'Entire DeltaNet layer in assembly, three tokens: passed; max abs error {np.max(np.abs(output-expected)):.3g}')
