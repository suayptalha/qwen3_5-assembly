"""Check assembly matrix-vector multiplication against real checkpoint BF16 rows."""
import ctypes
import json
import struct
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
model = root / 'model/model.safetensors-00001-of-00001.safetensors'
if not model.exists():
    model = root / 'model/model.safetensors-00001-of-00001.safetensors.part'
with model.open('rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    info = header['model.language_model.embed_tokens.weight']
    assert info['dtype'] == 'BF16' and info['shape'][1] == 1024
    f.seek(8 + header_size + info['data_offsets'][0])
    w = np.fromfile(f, dtype='<u2', count=128 * 1024).reshape(128, 1024)
rng = np.random.default_rng(221)
x = rng.standard_normal(1024, dtype=np.float32)
out = np.empty(128, dtype=np.float32)
lib = ctypes.CDLL(str(root / 'build/libkernels.so'))
lib.matvec_bf16_f32.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 2
lib.matvec_bf16_f32(w.ctypes.data, x.ctypes.data, out.ctypes.data, 128, 1024)
converted = (w.astype(np.uint32) << 16).view(np.float32)
expected = converted.astype(np.float64) @ x.astype(np.float64)
np.testing.assert_allclose(out, expected, rtol=3e-4, atol=1e-5)
print(f'Real embedding weights: 128 rows passed, max absolute error {np.max(np.abs(out - expected)):.3g}')

# A complete actual QKV projection, once the partial download has reached it.
info = header['model.language_model.layers.0.linear_attn.in_proj_qkv.weight']
end = 8 + header_size + info['data_offsets'][1]
if model.stat().st_size >= end:
    with model.open('rb') as f:
        f.seek(8 + header_size + info['data_offsets'][0])
        qkv = np.fromfile(f, dtype='<u2', count=6144 * 1024).reshape(6144, 1024)
    x = rng.standard_normal(1024, dtype=np.float32)
    out = np.empty(6144, dtype=np.float32)
    lib.matvec_bf16_f32(qkv.ctypes.data, x.ctypes.data, out.ctypes.data, 6144, 1024)
    expected = (qkv.astype(np.uint32) << 16).view(np.float32).astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=2e-5)
    print(f'Real layer-0 QKV projection: 6144 rows passed, max absolute error {np.max(np.abs(out - expected)):.3g}')
else:
    print('Layer-0 QKV projection skipped: weights not yet downloaded')

names = [f'model.language_model.layers.0.mlp.{p}_proj.weight' for p in ('gate', 'up', 'down')]
if model.stat().st_size >= max(8 + header_size + header[n]['data_offsets'][1] for n in names):
    weights = []
    with model.open('rb') as f:
        for name in names:
            info = header[name]
            f.seek(8 + header_size + info['data_offsets'][0])
            weights.append(np.fromfile(f, dtype='<u2', count=np.prod(info['shape'])).reshape(info['shape']))
    x = rng.standard_normal(1024, dtype=np.float32) * .1
    out = np.empty(1024, dtype=np.float32)
    scratch = np.empty(2 * 3584, dtype=np.float32)
    lib.mlp_bf16.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p]
    lib.mlp_bf16(*(w.ctypes.data for w in weights), x.ctypes.data, out.ctypes.data, 1024, 3584, scratch.ctypes.data)
    decoded = [(w.astype(np.uint32) << 16).view(np.float32) for w in weights]
    gate = decoded[0] @ x
    expected = decoded[2] @ ((gate / (1 + np.exp(-gate))) * (decoded[1] @ x))
    np.testing.assert_allclose(out, expected, rtol=3e-3, atol=1e-6)
    print(f'Real layer-0 SwiGLU MLP: passed, max absolute error {np.max(np.abs(out - expected)):.3g}')
else:
    print('Layer-0 MLP skipped: weights not yet downloaded')
