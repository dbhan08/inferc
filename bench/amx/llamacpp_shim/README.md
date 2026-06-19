# llama.cpp in-engine AMX shim — measured full-forward prefill speedup

This is the integration artifact behind §4.7 of the paper: the measured,
in-engine end-to-end prefill speedup of the pre-packed direct-AMX kernel,
obtained by intercepting llama.cpp's BLAS prefill matmuls.

llama.cpp's BLAS backend (`ggml-blas`) routes every prefill weight matmul
through a single `cblas_sgemm(NoTrans, Trans)` call whose `src0` is the
constant fp32 weight `W[N, K]`. The shim intercepts exactly that call for the
constant-weight fp32 case, pre-packs each weight once at first use, and runs
the multi-threaded `Nc=64`, `Kc=2048` kernel (same code as `amx_prepack.cc`).
Everything else (attention scores, small shapes, non-fp32) falls back to
`cblas_sgemm`. A single env var toggles the routine, so the **same binary**
runs both arms over the same fp32 GGUF — the GEMM routine is the only
deliberate difference.

## Files

- `amx_shim.h` — self-contained shim: vendored AMX encodings, pack-once weight
  cache (keyed on weight pointer), per-call activation transpose +
  multi-threaded dispatch. Copy into `ggml/src/ggml-blas/`.
- `ggml-blas.cpp.patch` — the two-hunk patch to `ggml/src/ggml-blas/ggml-blas.cpp`
  (include the shim + intercept the prefill `cblas_sgemm` + optional first-call
  bit-exact self-check). Generated against llama.cpp `f58bad4`.

## Recipe (reproduce the §4.7 number)

```sh
# 1. Clone + build llama.cpp with the Apple Accelerate BLAS backend.
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
      -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release

# 2. Drop in the shim and apply the patch.
cp /path/to/inferc/bench/amx/llamacpp_shim/amx_shim.h ggml/src/ggml-blas/
git apply /path/to/inferc/bench/amx/llamacpp_shim/ggml-blas.cpp.patch
cmake --build build --target llama-bench -j

# 3. Get an fp32 GGUF (fp32 so BOTH arms see identical weights — a quantized
#    GGUF would add a per-call dequant the shim does not target).
#    e.g. convert TinyLlama-1.1B:
#    PYTHONPATH=gguf-py python3 convert_hf_to_gguf.py /path/to/tinyllama-hf \
#        --outtype f32 --outfile tinyllama-f32.gguf

# 4. Measure both arms (same binary; env toggles only the GEMM routine).
#    Drop the first (cold) iteration: it carries the one-time weight pack +
#    GCD thread-pool spin-up, both load-time costs in deployment.
GGML_AMX_SHIM=0 ./build/bin/llama-bench -m tinyllama-f32.gguf -p 128 -n 0 -r 20 -t 4   # baseline (cblas)
GGML_AMX_SHIM=1 ./build/bin/llama-bench -m tinyllama-f32.gguf -p 128 -n 0 -r 20 -t 4   # ours
```

## Env toggles

| var | effect |
|---|---|
| `GGML_AMX_SHIM=1`   | route qualifying prefill matmuls to the AMX kernel (default off → cblas) |
| `GGML_AMX_SHIM_CHECK=1` | on the first intercepted call, recompute via cblas and print max-abs-diff (correctness) |
| `GGML_AMX_SHIM_DIAG=1`  | print `calls / packs / kernel-ms / pack-ms` at exit |

## Result (M1, macOS Tahoe 26.5.1, TinyLlama-1.1B fp32, 4 threads)

Steady-state throughput, dropping the cold first iteration, medians of 20:

| prefill | ours (tok/s) | cblas (tok/s) | ratio |
|---|---|---|---|
| 128 (the paper's `S=128`) | 420 | 291 | **1.44×** |
| 256 | 403 | 366 | 1.10× |
| 512 | 342 | 419 | 0.82× |

- **Bit-exact**: `GGML_AMX_SHIM_CHECK=1` reports max-abs-diff `0.000e+00`.
- `GGML_AMX_SHIM_DIAG=1` confirms `packs=107` (one per distinct weight, not
  per-call) — the pack-once cache works.
- The advantage is specific to the short-prefill regime (`M ≈ 128`): it narrows
  at 256 and reverses by 512, where the per-call activation transpose (weights
  are pre-packed, activations are not, and the transpose grows with `M`) and
  Accelerate's stronger large-`M` scaling overtake the gain.

The shim is a research instrument for an apples-to-apples measurement, not a
production patch: it only targets the fp32 BLAS path (not the quantized ggml
path, where dequant is fused into the matmul), and the activation transpose is
paid per call rather than hoisted.
