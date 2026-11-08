"""Compare our recurrent kernel against the Transformers Qwen3.5 reference."""
import ctypes
from pathlib import Path
import numpy as np
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

torch.set_num_threads(1)
rng = np.random.default_rng(82)
steps, heads, dim = 4, 3, 128
q, k, v = [rng.standard_normal((1, steps, heads, dim), dtype=np.float32) for _ in range(3)]
g = -rng.random((1, steps, heads), dtype=np.float32)
beta = rng.random((1, steps, heads), dtype=np.float32)
with torch.inference_mode():
    reference, final = torch_recurrent_gated_delta_rule(
        *(torch.from_numpy(x) for x in (q, k, v, g, beta)),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'build/libkernels.so'))
ptr = ctypes.c_void_p
lib.delta_step.argtypes = [ptr]*5 + [ctypes.c_size_t]*2 + [ctypes.c_float]*2
lib.l2norm_f32.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
state = np.zeros((heads, dim, dim), dtype=np.float32)
out = np.empty((1, steps, heads, dim), dtype=np.float32)
for t in range(steps):
    for h in range(heads):
        qn, kn = np.empty(dim, np.float32), np.empty(dim, np.float32)
        lib.l2norm_f32(q[0,t,h].ctypes.data, qn.ctypes.data, dim, 1e-6, 1 / np.sqrt(dim))
        lib.l2norm_f32(k[0,t,h].ctypes.data, kn.ctypes.data, dim, 1e-6, 1.)
        lib.delta_step(state[h].ctypes.data, qn.ctypes.data, kn.ctypes.data, v[0,t,h].ctypes.data,
                       out[0,t,h].ctypes.data, dim, dim, np.exp(g[0,t,h]), beta[0,t,h])
np.testing.assert_allclose(out, reference.numpy(), atol=5e-6, rtol=3e-4)
np.testing.assert_allclose(state, final.numpy()[0], atol=5e-6, rtol=3e-4)
print('Gated DeltaNet: multi-head, multi-token outputs and state match Transformers')
