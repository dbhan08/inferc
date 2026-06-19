#!/usr/bin/env python3
"""Generate fp32 CoreML matmul graphs (constant weight baked in) for the 12 LLM
prefill shapes, for the BNNS Graph head-to-head (amx_bnns_graph.cc).

Each model is y = x @ W with x:[M,K] the per-token activations and W:[K,N] the
CONSTANT weight -- the exact GEMM the paper's pre-packed kernel targets. fp32 is
forced (compute_precision=FLOAT32) so the comparison is bit-exact-precision with
the paper's fp32 kernel, not BNNS Graph's default fp16 downcast.

Compile happens once at model load (BNNS Graph repacks the constant weight then);
the harness times only graph execution, matching the paper's amortized pre-pack.
"""
import os, glob, struct
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

M = 128  # S, prefill batch (paper headline)

# (tag, N, K) -- C[M,N] = A[M,K] @ W[K,N]; shapes listed (N,K) as in Table 3.
SHAPES = [
    ("gpt2_qkv",   2048,  2048), ("gpt2_ffn1",  8192,  2048),
    ("gpt2_ffn2",  2048,  8192), ("gpt2_lmh",   60000, 2048),
    ("tiny_qkv",   2048,  2048), ("tiny_ffn1",  5632,  2048),
    ("tiny_ffn2",  2048,  5632), ("tiny_lmh",   32000, 2048),
    ("llama_qkv",  4096,  4096), ("llama_ffn1", 11008, 4096),
    ("llama_ffn2", 4096,  11008),("llama_lmh",  32000, 4096),
]

OUT = os.environ.get("BNNSG_DIR", "/tmp/amxbench/bnns_graph_models")
os.makedirs(OUT, exist_ok=True)

def build(tag, N, K, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, N), dtype=np.float64).astype(np.float32)
    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K))])
    def prog(x):
        return mb.matmul(x=x, y=W, name="y")
    m = ct.convert(prog, convert_to="mlprogram",
                   compute_units=ct.ComputeUnit.CPU_ONLY,
                   compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=ct.target.macOS15)
    pkg = os.path.join(OUT, tag + ".mlpackage")
    m.save(pkg)
    # also dump W so the harness can verify bit-exactness against its own sgemm
    W.tofile(os.path.join(OUT, tag + "_W.f32"))
    wb = glob.glob(pkg + "/Data/com.apple.CoreML/weights/*.bin")[0]
    fp32 = os.path.getsize(wb) >= K * N * 4
    print(f"  {tag:12} (N={N:5} K={K:5})  weight {os.path.getsize(wb):>10}B  fp32={fp32}")
    assert fp32, f"{tag}: weight not fp32 -- BNNS Graph would run reduced precision"

if __name__ == "__main__":
    print(f"coremltools {ct.__version__} -> {OUT}")
    for i, (tag, N, K) in enumerate(SHAPES):
        build(tag, N, K, seed=1000 + i)
    print(f"done: {len(SHAPES)} fp32 matmul graphs")
