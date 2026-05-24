# inferc v2 — Spec

> v1 shipped: 8 sessions, github.com/dbhan08/inferc, 48 tests, DistilBERT-SST2 within fp32 epsilon of ORT, 6.99× faster than ORT on MatMul, 20.5% over un-fused baseline. v2 is the next ~12 sessions and the path to an arxiv paper.

## Pitch

v2 extends inferc from encoder (DistilBERT) to autoregressive decoder (GPT-2 small) with **native KV cache, AMX-aware kernel dispatch, vDSP-vectorized pointwise pipeline, fused LayerNorm**, and an interactive `inferc chat` REPL. Targets *per-token GPT-2 decode latency lower than ONNX Runtime CPU EP on Apple M1*, attributed to specific shape thresholds in AMX engagement that nobody has empirically characterized in published work.

Outputs: working CPU inference engine extension, AMX microbenchmark artifact, and a tech-report-style paper submitted to arxiv (and to NeurIPS ENLSP 2026 as Plan B for peer review).

## Goals (v2)

**Engineering:**
- Load GPT-2 small (124M params) from ONNX (HuggingFace export)
- Add any IR / executor ops not yet in inferc (likely: a small handful — to be enumerated in session 9)
- **Native KV cache** as runtime executor state (not graph I/O). Two execution modes:
  - **Prefill**: forward over the prompt, populate per-layer K/V buffers
  - **Decode**: single-token forward, read cached K/V, append new K/V
- **AMX engagement microbench suite** — `inferc amx-probe` measures `cblas_sgemm` and `cblas_sgemv` GFLOPs across an M/N/K shape sweep, outputs the threshold curve as a JSON + heatmap
- **AMX-aware kernel dispatch** — decode-step matmul routes through `cblas_sgemv` (or scalar fallback) based on session-12 thresholds
- **vDSP-vectorized pointwise pipeline** — `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Erf`, `Tanh` route through `vDSP_v*` and `vv*` calls
- **Fused LayerNorm pattern matcher + kernel** — recognize the 8-op decomposition, replace with one-pass fused kernel
- **Constant folding + dead-code elimination** pass (small but ubiquitous win)
- **`inferc chat <model>`** — interactive REPL: greedy / temperature decode, prints token stream

**Paper-grade:**
- Per-token decode latency curves (positions 1, 8, 32, 128, 256) for inferc vs ORT-CPU vs llama.cpp vs CTranslate2 vs PyTorch-CPU
- Ablation table: independent contribution of each optimization
- Hardware-counter attribution via Instruments — *why* we win, not just *that* we win
- Open-source artifact: github.com/dbhan08/inferc + AMX microbench suite + one-command reproduction script
- arxiv submission as cs.PF / cs.LG tech report

## Non-goals (deferred)

- M2 / M3 / M4 (paper is M1-only; cross-chip generalization in v3)
- INT8 weight quantization (separate axis of optimization; significant scope; defer to v3)
- Speculative decoding (requires running 2 models; out of scope)
- Multi-batch decode (batch=1 generation is the canonical case; batched in v3)
- Neural Engine / GPU / Metal (explicitly CPU-focused paper)
- CLIP / multimodal (out of scope for this paper; v3 generality demo)
- GPT-2-medium / large / Llama (same architecture, scale up later)

## System dependencies (additions to v1)

| Tool | Why |
|---|---|
| `optimum-cli` (Python, via Poetry) | Export GPT-2 to ONNX with `past_key_values` as graph I/O for the apples-to-apples ORT bench |
| `tiktoken` (Python, optional) | If GPT-2's BPE is faster via tiktoken than HF `transformers` |
| `Instruments.app` (Xcode) | Hardware-counter attribution; no additional install needed |

No new C++ deps. Same Accelerate, Protobuf, GoogleTest, nlohmann/json.

## KV cache design sketch

The mechanism that turns "compute O(n²) attention per token" into "compute O(n) per token." Two pieces of state per transformer layer:

```cpp
// runtime/executor.h (concept)
struct LayerKVCache {
  rt::Tensor k;  // shape [num_heads, max_seq_len, head_dim], pre-allocated
  rt::Tensor v;  // shape [num_heads, max_seq_len, head_dim], pre-allocated
};

class Executor {
  ...
  std::vector<LayerKVCache> kv_;  // one per transformer block; lazily allocated
  int64_t cached_len_ = 0;        // current length filled in the cache
  ...
};
```

**Prefill flow** (`inferc decode --prompt "..."` first call):
1. Tokenize prompt → input IDs `[1, N]`.
2. Run forward pass through all transformer blocks. For each attention op, project Q/K/V from the full prompt.
3. **Write** K and V to `kv_[layer].k[:, :N, :]` and `kv_[layer].v[:, :N, :]`.
4. Set `cached_len_ = N`.
5. Output logits for position `N-1`. Sample next token.

**Decode flow** (subsequent steps):
1. Input is a single token `[1, 1]`.
2. Run forward through transformer blocks. For each attention op:
   - Compute Q, K, V for just this one token (1-vector projections, GEMV-shaped — AMX engagement matters).
   - **Append** new K to `kv_[layer].k[:, cached_len_, :]`; same for V.
   - Attention reads the full `kv_[layer].k[:, :cached_len_+1, :]` (all cached + new).
3. `cached_len_ += 1`. Output logits, sample next token, repeat until EOS or max tokens.

**Why this is fast:**
- Without cache: each new token re-projects K and V for all N previous tokens (work = O(N) per token, total O(N²) over N tokens).
- With cache: each new token projects K and V only for itself (work = O(1) per token, total O(N)).
- For N=256 tokens, that's a ~50x reduction in attention compute relative to naive recompute. Every modern LLM runtime does this; we're implementing it directly.

**The decode-step matmul shapes are the AMX-engagement question.** Q/K/V projections at decode time are `[1, hidden_dim] × [hidden_dim, hidden_dim]` — that's a 1×768 row times a 768×768 matrix = a single-row matmul, which BLAS calls **GEMV**. Whether Accelerate dispatches this to AMX depends on shape thresholds that aren't documented. **Session 12 measures the threshold; sessions 13+ exploit it.**

## Sessions (v2 — 12 sessions, ~3 months)

| # | Subject | Done when | Paper section |
|---|---|---|---|
| 9 | GPT-2 scaffold | `scripts/fetch_gpt2.py` works; `inferc inspect models/gpt2.onnx` runs; missing ops enumerated in a session note | §3.1 |
| 10 | Fill missing ops + GPT-2 forward pass | Position-0 logits match HF `transformers` reference within 1e-3 (no cache yet, full forward pass) | §3.1 (correctness) |
| 11 | **KV cache executor state** | `inferc decode` greedy 32-token generation matches HF token-for-token | §3.6 |
| 12 | **AMX microbench suite** | `inferc amx-probe` produces M/N/K × GFLOPs CSV; threshold curve visible in plot | **Figure 1** (novel contribution) |
| 13 | **AMX-aware decode kernel** | Per-token decode latency drops measurably from session 11 baseline; correctness still gates | §3.7 + Table 2 row |
| 14 | vDSP pointwise pipeline | DistilBERT and GPT-2 both faster; existing 48 tests still pass | Ablation row |
| 15 | Fused LayerNorm pass + kernel | LayerNorm replaces 8-op chain with 1 op; ~1s saved per inference on DistilBERT | Ablation row |
| 16 | **`inferc chat` REPL** | Interactive token stream from a prompt; `--temperature`, `--max-tokens`, `--top-k` | Demo for paper, screenshot |
| 17 | Multi-baseline bench harness | inferc / ORT / llama.cpp / CTranslate2 / PyTorch all benchable from one command | Table 1 data |
| 18 | Hardware-counter attribution | Instruments traces show where time goes for inferc vs ORT on GPT-2 decode | §6, Figure 2 |
| 19 | Paper draft | LaTeX outline locked, all figures + tables in, first full draft of all sections | Paper |
| 20 | Polish + arxiv submission | Endorsement secured (university affiliation), reviewer reads pass, final pass, posted | arxiv URL live |

## Paper plan

### Target: arxiv first, ENLSP as Plan B

**Plan A: arxiv (cs.PF primary, cs.LG and cs.AR cross-listed).** University affiliation handles endorsement (auto-eligible). Submit Week ~14-15. URL live within 1-2 days of submission.

**Plan B: NeurIPS ENLSP 2026.** Workshop submission usually due late August / early September. Same paper, polished further. Held in December.

### What the paper is

**Working title:** *Empirical AMX Engagement Characterization for Transformer Inference on Apple Silicon, with Application to Per-Token Decode*

**Two contributions:**

1. **Measurement (novel):** systematic GFLOPs-vs-shape characterization of Apple Accelerate's sgemm and sgemv kernels on M1, identifying the shape thresholds at which AMX dispatch engages. Reproducible from `inferc amx-probe`.

2. **Application:** an AMX-aware decode-step kernel that exploits the characterization to beat ORT-CPU on GPT-2-small per-token decode latency on M1.

### Paper outline (target ~6 pages content + refs)

```
1. Introduction (1 pg)
   - Hook: ORT-CPU under-optimizes batch-1 decode shapes on Apple Silicon
   - Two contributions
   - Headline number
2. Background and related work (1 pg)
   - AMX prior art (corsix, dougallj, Zhou MIT thesis)
   - CPU transformer inference (Fast DistilBERT, CTranslate2, llama.cpp)
   - KV cache: standard practice, prior layout/compression work
3. System (1 pg)
   - ONNX→IR→executor pipeline (terse; details on GitHub)
   - Pattern library (Erf-GELU from v1; fused LayerNorm new in v2)
   - KV cache as executor state
4. AMX engagement characterization (1 pg — novel core)
   - Methodology: shape sweep design, n=30 trials, error bars
   - Figure 1: GFLOPs heatmap with threshold curve overlaid
   - Findings: where AMX engages vs falls back to NEON
5. AMX-aware decode kernel (1 pg)
   - Algorithm: shape-aware dispatch (AMX / NEON / scalar)
   - Microbenchmark on isolated decode shapes
6. Evaluation (1.5 pg)
   - Setup: M1, GPT-2-small, fp32, single-thread, batch=1
   - Table 1: per-token latency at positions 1, 8, 32, 128, 256, across 5 baselines
   - Figure 2: latency curves
   - Table 2: ablation (which optimization buys what)
   - Correctness verification
7. Discussion + limitations (0.5 pg)
   - Why we win: GEMV path under-optimized in MLAS
   - M1-only scope; cross-chip future work
   - fp32 only; INT8 future work
8. Conclusion + open-source artifact (0.25 pg)
```

### "Ready to submit" criteria

Don't submit unless:
1. **AMX figure (Figure 1) shows a clean threshold** — engagement curve is interpretable, not noise.
2. **Headline table shows inferc beats ORT-CPU on at least one (position) configuration** for GPT-2 decode on M1.
3. **Correctness gate holds** — token-for-token match with HuggingFace through position 256.
4. **Ablation table is real** — each row turns one optimization on/off and measures.
5. **Repo is reproducible** — `git clone && one_command.sh` produces the paper's headline number.

If 1-2 don't materialize → re-scope paper as "characterization only" (just contribution 1) or shelve. Decision point at end of session 18.

## Risks / unknowns

- **AMX threshold may not be clean.** Apple's heuristics could be non-monotonic. **Mitigation:** report honestly; "noisy but reproducible" is still a contribution. Don't fabricate a clean curve.
- **GPT-2 may need ops we don't have.** Likely: more involved Gather patterns, possibly `OneHot`, `MaskedFill` (probably maps to existing `Where`). **Mitigation:** session 9 enumerates the gap; session 10 fills it. If the gap is huge (>10 new ops), re-scope.
- **ORT's KV cache export may not be apples-to-apples.** ORT requires re-exporting with `past_kv` as graph I/O. **Mitigation:** verify via op-count profiling that ORT actually uses its cache (attention ops should not grow O(n²) per step).
- **vDSP overhead at small sizes.** Single-token decode produces tiny tensors where vDSP setup overhead may dominate. **Mitigation:** branch on size; fall back to scalar for tiny N.
- **Paper feedback bandwidth.** Reading drafts is slow. **Mitigation:** find one reader (advisor / friend / labmate) by week 13 so they have warning before draft is ready.
- **Time slippage.** ~12 sessions × 1 weekend each = ~12 weeks; with life happening, realistically 16-18 weeks. **Mitigation:** buffer week in the schedule; cut sessions 14b/15 first if behind.

## Target resume bullet (v2)

> Built a from-scratch C++17 CPU inference engine for transformers on Apple Silicon. Implemented native KV-cache autoregressive decoding routed through Apple Accelerate's AMX-tuned sgemv path, IR pattern-matching fusion passes (MatMul+Add+GELU, fused LayerNorm), and a vDSP-vectorized pointwise pipeline. **Per-token GPT-2-small decode latency [X] ms on M1, vs ONNX Runtime CPU EP's [Y] ms — a [Y/X]x improvement on the autoregressive decode path that ORT's GEMM-shaped kernels under-optimize for batch-1 GEMV shapes.** Open-source: github.com/dbhan08/inferc. arxiv tech report: arxiv.org/abs/[XXXX].

X and Y filled in at submission, not made up.

## Timeline summary

- **Today + 13 weeks → end of session 18:** all engineering and evaluation done.
- **Week 14-15:** writing.
- **Week 16:** arxiv submission, URL live.
- **Week 17+:** ENLSP submission (if deadline still open), polish, share on HN / Twitter / LinkedIn.
- **December 2026:** if ENLSP accepts, present at workshop.

## Next concrete action

**Session 9: GPT-2 scaffold.** ~1-2 hours. Three deliverables:
1. `scripts/fetch_gpt2.py` — pulls GPT-2-small from HuggingFace, exports to ONNX.
2. `scripts/make_gpt2_inputs.py` — tokenizes a fixed prompt, generates HF reference logits at position 0.
3. A session note in `SESSIONS.md` listing: opset version, total node count, op-type counts, **and the list of op types inferc doesn't yet handle.** That last list determines session 10's scope.

If session 9 reveals GPT-2 needs ≤5 new ops, sessions 9-13 are realistic as scoped. If it needs >10 new ops, we re-plan.
