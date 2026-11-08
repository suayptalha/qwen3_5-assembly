import ctypes
from pathlib import Path
import numpy as np
lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/libkernels.so'))
lib.sample_topk.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_float]
lib.sample_topk.restype=ctypes.c_uint32
rng=np.random.default_rng(41)
for n,k,temp in [(1,1,1.0),(17,1,.7),(17,4,.7),(1000,40,.7),(500,100,2.0)]:
    logits=rng.normal(size=n).astype(np.float32)
    ids=np.argsort(-logits,kind='stable')[:k]
    probs=np.exp((logits[ids].astype(np.float64)-logits[ids[0]])/temp)
    cumulative=np.cumsum(probs)/probs.sum()
    for rand in rng.integers(0,2**32,size=200,dtype=np.uint32):
        fraction=(int(rand)>>8)/2**24
        expected=int(ids[min(np.searchsorted(cumulative,fraction,side='right'),k-1)])
        actual=lib.sample_topk(logits.ctypes.data,n,k,int(rand),temp)
        assert actual==expected,(n,k,temp,int(rand),actual,expected)
print('Temperature and top-k sampling: deterministic distribution tests passed')
