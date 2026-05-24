# inferc — Reproducible demo

The full path from a fresh clone to the headline 20.5% speedup, ~5 minutes on an M1.

## 0. Setup (one-time)

```bash
brew install cmake protobuf poetry
poetry env use /opt/homebrew/bin/python3.13
poetry install --extras dev
cmake -B build && cmake --build build
```

Smoke test:

```bash
$ ./build/inferc --version
0.1.0
$ cd build && ctest --output-on-failure
...
100% tests passed, 0 tests failed out of 48
```

## 1. Fetch DistilBERT + canonical inputs

```bash
poetry run python scripts/fetch_distilbert.py    # downloads ~268 MB DistilBERT-SST2
poetry run python scripts/make_inputs.py         # writes models/input_ids.bin,
                                                 #        models/attention_mask.bin,
                                                 #        models/golden_logits.bin
```

Fixed prompt: `"The food at this restaurant was incredible."`
Expected ORT prediction: `POSITIVE`, logits ≈ `[-4.336, 4.662]`.

## 2. Inspect the model

```bash
$ ./build/inferc inspect models/distilbert.onnx
Model: torch-jit-export
  IR version:        7
  Producer:          pytorch 1.10
  Opset:             ai.onnx v13

Inputs (2):
  input_ids           int64     [batch, sequence]
  attention_mask      int64     [batch, sequence]

Outputs (1):
  logits              float32   [batch, 2]

Nodes: 555
  Op type counts (descending):
    Constant                    115
    Add                          81
    MatMul                       48
    ...
```

## 3. Run end-to-end (unoptimized) — correctness check

```bash
$ ./build/inferc run models/distilbert.onnx \
      --input-ids models/input_ids.bin \
      --attention-mask models/attention_mask.bin \
      --output /tmp/inferc_logits.bin

inferc run: logits float32[1, 2] -> /tmp/inferc_logits.bin
  values: -4.33637 4.6618
```

Matches ORT golden at float32 epsilon. The end-to-end correctness test (`tests/correctness_test.cc`) asserts `max_abs_diff ≤ 1e-3`; actual measured `≤ 5e-7`.

## 4. Optimize — apply the fusion pass

```bash
$ ./build/inferc optimize models/distilbert.onnx --out models/distilbert.opt.onnx
inferc optimize:
  recognize-GELU folded: 6 patterns
  MatMul+Add+GELU fused: 6 patterns
  nodes: 555 -> 501 (54 removed)
  wrote models/distilbert.opt.onnx
```

6 GELU patterns = 6 transformer blocks × 1 FFN-GELU each. Each fusion replaces a `MatMul → Add → 7-node Erf-decomposed GELU` chain with one `FusedMatMulAddGELU` op.

## 5. Run the optimized plan

```bash
$ ./build/inferc run models/distilbert.opt.onnx \
      --input-ids models/input_ids.bin \
      --attention-mask models/attention_mask.bin \
      --output /tmp/inferc_opt_logits.bin

inferc run: logits float32[1, 2] -> /tmp/inferc_opt_logits.bin
  values: -4.33637 4.6618
```

Bit-identical to the unoptimized run modulo fp32 rounding. The optimized-correctness test gates this.

## 6. Bench — the headline number

```bash
$ ./build/inferc bench --model models/distilbert.opt.onnx \
                       --ort-model models/distilbert.onnx \
                       -n 30 --warmup 5

[1/3] Profiling inferc (30 iters + 5 warmup)...
inferc run: profile -> bench_out/inferc_baseline.json (30 iters, mean=3922.26ms p50=3912.44ms p95=3974.73ms)

[2/3] Profiling ort-cpu (30 iters + 5 warmup)...
  pass-1 (totals): mean=124.30ms p50=124.22ms p95=124.89ms (n=30)
  pass-2 (top ops): MatMul=234.26ms, Gelu=5.70ms, LayerNormalization=3.74ms, Add=2.37ms, Softmax=1.23ms
  wrote bench_out/baseline_ort.json

[3/3] Comparing...

=== inferc compare ===
  model: models/distilbert.opt.onnx

backend                iters   mean(ms)    p50(ms)    p95(ms)    min(ms)    max(ms)   RSS(MB)
---------------------------------------------------------------------------------------------
inferc-optimized          30    3922.26    3912.44    3974.73    3877.55    3980.29     932.0
ort-cpu                   30     124.30     124.22     124.89     123.62     126.27     792.8

  inferc-optimized is 31.55x slower than ort-cpu (mean total)

Top ops by mean total time per iter (ratio = ort-cpu/inferc-optimized):
op_type              calls/iter inferc-optimize     ort-cpu(ms)     ratio
-------------------------------------------------------------------------
ReduceMean                   26        1382.041           0.000       —
Transpose                    24         753.204           0.952     0.001
Add                          69         538.041           2.369     0.004
Div                          19         243.931           0.126     0.001
MatMul                       42          33.510         234.255      6.99
...
```

## 7. Compare baseline vs optimized — the fusion win

```bash
$ ./build/inferc run models/distilbert.onnx \
      --input-ids models/input_ids.bin \
      --attention-mask models/attention_mask.bin \
      --profile bench_out/inferc_baseline.json -n 30 --warmup 5
inferc run: profile -> bench_out/inferc_baseline.json (30 iters, mean=4935.11ms p50=4932.73ms p95=4964.62ms)

$ ./build/inferc compare bench_out/inferc_baseline.json bench_out/inferc_optimized.json
=== inferc compare ===
  model: models/distilbert.onnx

backend                iters   mean(ms)    p50(ms)    p95(ms)    min(ms)    max(ms)   RSS(MB)
---------------------------------------------------------------------------------------------
inferc-baseline           30    4935.11    4932.73    4964.62    4907.63    4982.94    1091.4
inferc-optimized          30    3915.37    3900.60    4018.79    3881.52    4071.54    1013.0

  inferc-baseline is 1.26x slower than inferc-optimized (mean total)

Top ops by mean total time per iter (ratio = inferc-optimized/inferc-baseline):
op_type              calls/iter inferc-baseline inferc-optimize     ratio
-------------------------------------------------------------------------
Add                          81        1075.155         537.204     0.500
Mul                          25         429.112         161.496     0.376
Div                          25         503.466         243.177     0.483
MatMul                       48          49.942          33.384     0.668
FusedMatMulAddGELU            6           0.000          53.279       —
...
```

**That's the headline: ~1020 ms / iter eliminated by collapsing 12 separate pointwise ops + 6 FFN MatMuls into 6 fused-kernel calls.** The fusion replaces 5+ buffer passes per FFN with a single per-element sweep that does bias-add and exact GELU in one go.

## What's NOT in this demo

- Multi-threaded ORT (intentionally compared single-thread for reproducibility — flip `intra_op_num_threads` in `bench/bench_ort.py` to lift this).
- Quantization, GPU/Metal, fp16 — explicitly out of scope for v1 (see [`PROJECT.md`](PROJECT.md)).
- Vectorized pointwise ops — the 31x absolute gap to ORT is the unvectorized LayerNorm/Transpose pipeline. That's v2 work.
