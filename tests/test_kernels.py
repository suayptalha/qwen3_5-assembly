import ctypes
from pathlib import Path
import numpy as np

lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'build/libkernels.so'))
ptr = ctypes.c_void_p
for fn in (lib.dot_f32, lib.dot_bf16_f32):
    fn.argtypes = [ptr, ptr, ctypes.c_size_t]
    fn.restype = ctypes.c_float
rng = np.random.default_rng(42)
for n in (0, 1, 7, 8, 9, 31, 128, 1024, 3584):
    for _ in range(20):
        a = rng.standard_normal(n, dtype=np.float32)
        b = rng.standard_normal(n, dtype=np.float32)
        got = lib.dot_f32(a.ctypes.data, b.ctypes.data, n)
        np.testing.assert_allclose(got, np.dot(a.astype(np.float64), b.astype(np.float64)), rtol=2e-4, atol=2e-4)
        bf = (a.view(np.uint32) >> 16).astype(np.uint16)
        restored = (bf.astype(np.uint32) << 16).view(np.float32)
        got = lib.dot_bf16_f32(bf.ctypes.data, b.ctypes.data, n)
        np.testing.assert_allclose(got, np.dot(restored.astype(np.float64), b.astype(np.float64)), rtol=2e-4, atol=2e-4)
print('AVX2 FP32 and BF16 dot products: all tests passed')

lib.matvec_bf16_f32.argtypes = [ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
for rows, cols in [(0, 9), (3, 0), (1, 7), (7, 9), (20, 1024)]:
    matrix = rng.standard_normal((rows, cols), dtype=np.float32)
    weights = (matrix.view(np.uint32) >> 16).astype(np.uint16)
    restored = (weights.astype(np.uint32) << 16).view(np.float32)
    x = rng.standard_normal(cols, dtype=np.float32)
    out = np.full(rows, np.nan, dtype=np.float32)
    lib.matvec_bf16_f32(weights.ctypes.data, x.ctypes.data, out.ctypes.data, rows, cols)
    np.testing.assert_allclose(out, restored.astype(np.float64) @ x.astype(np.float64), rtol=2e-4, atol=2e-4)
print('AVX2 BF16 matrix-vector multiplication: all tests passed')

lib.delta_step.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
for kd, vd in [(1, 1), (3, 7), (128, 128)]:
    state = rng.standard_normal((kd, vd), dtype=np.float32) * .1
    q = rng.standard_normal(kd, dtype=np.float32) * .1
    k = rng.standard_normal(kd, dtype=np.float32) * .1
    v = rng.standard_normal(vd, dtype=np.float32)
    out = np.empty(vd, dtype=np.float32)
    for _ in range(5):
        decay, beta = np.float32(.91), np.float32(.37)
        expected_state = state * decay
        expected_state = expected_state + k[:, None] * ((v - k @ expected_state) * beta)[None, :]
        expected_out = q @ expected_state
        lib.delta_step(state.ctypes.data, q.ctypes.data, k.ctypes.data, v.ctypes.data, out.ctypes.data, kd, vd, decay, beta)
        np.testing.assert_allclose(state, expected_state, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, expected_out, rtol=1e-5, atol=1e-6)
print('Gated DeltaNet recurrent update: all tests passed')

lib.rmsnorm_f32.argtypes = [ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
for n in (0, 1, 128, 1024, 3584):
    x = rng.standard_normal(n, dtype=np.float32)
    weight = rng.standard_normal(n, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for offset in (0.0, 1.0):
        lib.rmsnorm_f32(x.ctypes.data, weight.ctypes.data, out.ctypes.data, n, 1e-6, offset)
        if n:
            ref = x / np.sqrt(np.mean(x * x) + 1e-6) * (weight + offset)
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
print('RMS normalization: all tests passed')

lib.rope_f32.argtypes = [ptr, ptr, ptr, ctypes.c_size_t]
for half in (0, 1, 32, 128):
    x = rng.standard_normal(half * 2 + 5, dtype=np.float32)
    angle = rng.standard_normal(half, dtype=np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    original = x.copy()
    lib.rope_f32(x.ctypes.data, cos.ctypes.data, sin.ctypes.data, half)
    ref = original.copy()
    ref[:half] = original[:half] * cos - original[half:2*half] * sin
    ref[half:2*half] = original[half:2*half] * cos + original[:half] * sin
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-7)
print('Rotary embeddings: all tests passed')

lib.exp_f32.argtypes = [ctypes.c_float]
lib.exp_f32.restype = ctypes.c_float
for value in np.linspace(-80, 80, 10001, dtype=np.float32):
    got = lib.exp_f32(float(value))
    np.testing.assert_allclose(got, np.exp(np.float64(value)), rtol=4e-6, atol=0)
print('Scalar exponential approximation over [-80, 80]: all tests passed')

lib.softmax_f32.argtypes = [ptr, ptr, ctypes.c_size_t]
for n in (0, 1, 2, 7, 128, 2048):
    for scale in (1, 10, 100):
        x = rng.standard_normal(n, dtype=np.float32) * scale
        out = np.empty(n, dtype=np.float32)
        lib.softmax_f32(x.ctypes.data, out.ctypes.data, n)
        if n:
            t = np.exp(x.astype(np.float64) - np.max(x))
            ref = t / t.sum()
            np.testing.assert_allclose(out, ref, rtol=3e-5, atol=1e-7)
            inplace = x.copy()
            lib.softmax_f32(inplace.ctypes.data, inplace.ctypes.data, n)
            np.testing.assert_allclose(inplace, ref, rtol=3e-5, atol=1e-7)
print('Softmax (finite inputs, including in-place): all tests passed')

lib.causal_conv4_bf16.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_size_t]
for channels in (0, 1, 17, 6144):
    state = rng.standard_normal((channels, 4), dtype=np.float32)
    matrix = rng.standard_normal((channels, 4), dtype=np.float32)
    weights = (matrix.view(np.uint32) >> 16).astype(np.uint16)
    wf = (weights.astype(np.uint32) << 16).view(np.float32)
    output = np.empty(channels, dtype=np.float32)
    for _ in range(4):
        x = rng.standard_normal(channels, dtype=np.float32)
        refstate = np.concatenate((state[:, 1:], x[:, None]), axis=1)
        ref = (refstate * wf).sum(axis=1)
        lib.causal_conv4_bf16(state.ctypes.data, weights.ctypes.data, x.ctypes.data, output.ctypes.data, channels)
        np.testing.assert_array_equal(state, refstate)
        np.testing.assert_allclose(output, ref, rtol=1e-5, atol=1e-6)
print('Streaming depthwise causal convolution, BF16 weights: all tests passed')

lib.silu_f32.argtypes = [ptr, ptr, ctypes.c_size_t]
x = np.linspace(-100, 100, 10001, dtype=np.float32)
out = np.empty_like(x)
lib.silu_f32(x.ctypes.data, out.ctypes.data, x.size)
ref = x.astype(np.float64) / (1 + np.exp(-x.astype(np.float64)))
np.testing.assert_allclose(out, ref, rtol=5e-6, atol=1e-7)
lib.silu_f32(x.ctypes.data, x.ctypes.data, x.size)
np.testing.assert_allclose(x, ref, rtol=5e-6, atol=1e-7)
print('SiLU activation including in-place and extreme finite inputs: all tests passed')

lib.mlp_bf16.argtypes = [ptr]*5 + [ctypes.c_size_t]*2 + [ptr]
for hidden, intermediate in ((1,1), (7,9), (32,50)):
    mats = [rng.standard_normal((r,c), dtype=np.float32)*.1 for r,c in ((intermediate,hidden), (intermediate,hidden), (hidden,intermediate))]
    weights = [(m.view(np.uint32)>>16).astype(np.uint16) for m in mats]
    decoded = [(w.astype(np.uint32)<<16).view(np.float32) for w in weights]
    x = rng.standard_normal(hidden, dtype=np.float32)
    out = np.empty(hidden, dtype=np.float32)
    scratch = np.empty(intermediate*2, dtype=np.float32)
    lib.mlp_bf16(*(w.ctypes.data for w in weights), x.ctypes.data, out.ctypes.data, hidden, intermediate, scratch.ctypes.data)
    gate = decoded[0] @ x
    ref = decoded[2] @ ((gate / (1 + np.exp(-gate))) * (decoded[1] @ x))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-6)
print('Complete BF16 SwiGLU MLP: all tests passed')

lib.bf16_to_f32.argtypes = [ptr, ptr, ctypes.c_size_t]
for n in (0, 1, 7, 8, 9, 1024):
    b = rng.integers(0, 65536, n, dtype=np.uint16)
    out = np.empty(n, dtype=np.float32)
    lib.bf16_to_f32(b.ctypes.data, out.ctypes.data, n)
    np.testing.assert_array_equal(out.view(np.uint32), b.astype(np.uint32) << 16)
print('BF16 to FP32 bit-exact conversion: all tests passed')

lib.attention_f32.argtypes = [ptr]*4 + [ctypes.c_size_t]*2 + [ptr]
for tokens, dim in ((0,8), (1,1), (4,7), (17,128), (31,256), (1025,256)):
    q = rng.standard_normal(dim, dtype=np.float32)
    k = rng.standard_normal((tokens,dim), dtype=np.float32)
    v = rng.standard_normal((tokens,dim), dtype=np.float32)
    out = np.empty(dim, dtype=np.float32)
    scratch = np.empty(tokens, dtype=np.float32)
    lib.attention_f32(q.ctypes.data, k.ctypes.data, v.ctypes.data, out.ctypes.data, tokens, dim, scratch.ctypes.data)
    if tokens:
        score = k @ q / np.sqrt(dim)
        prob = np.exp(score - score.max())
        ref = (prob / prob.sum()) @ v
    else:
        ref = np.zeros(dim)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=3e-6)
print('Scaled dot-product attention: all tests passed')

lib.l2norm_f32.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
for n in (1, 7, 128, 256):
    for scale in (1.0, 1/np.sqrt(128)):
        x = rng.standard_normal(n, dtype=np.float32)
        out = np.empty_like(x)
        lib.l2norm_f32(x.ctypes.data, out.ctypes.data, n, 1e-6, scale)
        ref = x / np.sqrt(np.sum(x*x) + 1e-6) * scale
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)
print('L2 normalization: all tests passed')

for name in ('sigmoid_f32', 'softplus_f32'):
    fn = getattr(lib, name)
    fn.argtypes = [ptr, ptr, ctypes.c_size_t]
    x = np.linspace(-100, 100, 10001, dtype=np.float32)
    out = np.empty_like(x)
    fn(x.ctypes.data, out.ctypes.data, x.size)
    if name.startswith('sigmoid'):
        ref = 1. / (1. + np.exp(-x.astype(np.float64)))
    else:
        ref = np.logaddexp(0., x.astype(np.float64))
    np.testing.assert_allclose(out, ref, rtol=4e-4, atol=1e-7)
print('Sigmoid and softplus gating: all tests passed')
