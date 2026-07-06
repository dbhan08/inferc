#!/usr/bin/env python3
# Export REAL NF4-quantized linear layers (unpacked) for amx_e2e_codebook.cc.
# For each target writes /tmp/<tag>_{idx,scale,At16,Cref16,dims}.bin:
#   idx[N][K] uint8 (0..15), scale[N] f32 (per-out-channel absmax), dims=[N,K] int32,
#   At16[K][16] f32 (real random activation, transposed), Cref16[16][N] f32 = PyTorch A@dequant(W).T
# Usage: export_real_layers.py opt | tinyllama
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM
NF4=np.array([-1.,-0.6961928,-0.52507305,-0.39491749,-0.28444138,-0.18477343,-0.09105004,0.,
 0.0795803,0.1609302,0.2461123,0.33791524,0.44070983,0.562617,0.72295684,1.],np.float32)

def export(tag, W):
    W=W.astype(np.float32); N,K=W.shape
    scale=np.abs(W).max(1); scale[scale<1e-8]=1e-8
    idx=np.abs((W/scale[:,None])[:,:,None]-NF4[None,None,:]).argmin(2).astype(np.uint8)
    Wdq=(scale[:,None]*NF4[idx]).astype(np.float32)
    rng=np.random.default_rng(0); A=rng.standard_normal((16,K)).astype(np.float32)
    Cref=(A@Wdq.T).astype(np.float32); At=np.ascontiguousarray(A.T)   # [K][16]
    np.ascontiguousarray(idx).tofile(f"/tmp/{tag}_idx.bin")
    scale.astype(np.float32).tofile(f"/tmp/{tag}_scale.bin")
    At.tofile(f"/tmp/{tag}_At16.bin"); Cref.tofile(f"/tmp/{tag}_Cref16.bin")
    np.array([N,K],np.int32).tofile(f"/tmp/{tag}_dims.bin")
    print(f"  {tag}: N={N} K={K}  weight rel-err {np.linalg.norm(Wdq-W)/np.linalg.norm(W):.4f}")

which=sys.argv[1] if len(sys.argv)>1 else "opt"
if which=="opt":
    m=AutoModelForCausalLM.from_pretrained("facebook/opt-125m",torch_dtype=torch.float32).eval()
    export("opt_fc1", m.model.decoder.layers[0].fc1.weight.data.numpy())   # N=3072 K=768
elif which=="tinyllama":
    m=AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",torch_dtype=torch.float32).eval()
    L=m.model.layers[0]
    export("tl_gate", L.mlp.gate_proj.weight.data.numpy())   # N=5632 K=2048 (FFN up, N-large)
    export("tl_down", L.mlp.down_proj.weight.data.numpy())   # N=2048 K=5632 (FFN down, K-large)
    export("tl_q",    L.self_attn.q_proj.weight.data.numpy())# N=2048 K=2048 (square attn)
else:
    print("unknown target",which); sys.exit(1)
print("done")
