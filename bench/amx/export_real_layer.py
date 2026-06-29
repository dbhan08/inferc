#!/usr/bin/env python3
# Export a REAL OPT-125M layer, NF4-quantized per-channel, to /tmp/real_*.bin for
# amx_real_kernel_test.cc (which runs it through the actual AMX kernel and checks it
# reproduces PyTorch's A.dequant(W)). pip install torch transformers; then run this.
import torch, numpy as np
from transformers import AutoModelForCausalLM
NF4=np.array([-1.,-0.6961928,-0.52507305,-0.39491749,-0.28444138,-0.18477343,-0.09105004,0.,
 0.0795803,0.1609302,0.2461123,0.33791524,0.44070983,0.562617,0.72295684,1.],np.float32)
m=AutoModelForCausalLM.from_pretrained("facebook/opt-125m",torch_dtype=torch.float32).eval()
W=m.model.decoder.layers[0].fc1.weight.data.numpy().astype(np.float32)   # [N=3072,K=768]=[out,in]
N,K=W.shape; M=16
scale=np.abs(W).max(1); scale[scale<1e-8]=1e-8
idx=np.abs((W/scale[:,None])[:,:,None]-NF4[None,None,:]).argmin(2).astype(np.uint8)
Wdq=(scale[:,None]*NF4[idx]).astype(np.float32)
rng=np.random.default_rng(0); A=rng.standard_normal((M,K)).astype(np.float32)
Cref=(A@Wdq.T).astype(np.float32); At=np.ascontiguousarray(A.T)
NT=N//16; ir=idx.reshape(NT,16,K)
packed=(ir[:,0::2,:]|(ir[:,1::2,:]<<4)).astype(np.uint8)
idxp=np.ascontiguousarray(packed.transpose(0,2,1)).reshape(-1)
for nm,a in [("idxp",idxp),("scale",scale.astype(np.float32)),("At",At),("Cref",Cref)]: a.tofile(f"/tmp/real_{nm}.bin")
print(f"exported N={N} K={K} M={M}, weight rel-err {np.linalg.norm(Wdq-W)/np.linalg.norm(W):.4f}")
