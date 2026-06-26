#!/usr/bin/env python3
"""Real-model perplexity for the AMX codebook-GEMM mechanism.

The AMX kernel computes A.dequant(W) BIT-EXACTLY (verified in amx_codebook_gemm.cc),
so a model running through the kernel produces IDENTICAL outputs to the same model with
its weights replaced by their dequantized values. Therefore the real-model accuracy can
be measured exactly here in PyTorch -- no AMX needed for the accuracy number; the kernel
supplies only the speed (4-5x over Accelerate, see the C++ benches).

This quantizes every Linear weight to 4-bit with a per-group (64-weight) 16-entry codebook
-- non-uniform k-means (what the indexed-MATFP gather uniquely enables free) and, for
contrast, uniform int4 -- then reports wikitext-2 perplexity vs the fp32 baseline.

Run (you supply the model download):
    pip install torch transformers datasets
    python bench/amx/codebook_perplexity.py --model gpt2
    python bench/amx/codebook_perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
import argparse, numpy as np, torch
# transformers/datasets imported lazily in main() so the quant functions below are
# importable/testable without those deps (only numpy+torch needed for quant).

GROUP = 64          # weights per quant group (NF4-style)
LEVELS = 16         # 4-bit -> 16 codebook entries

def _pad_groups(w):
    flat = w.reshape(-1).astype(np.float32)
    pad = (-len(flat)) % GROUP
    if pad: flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    return flat.reshape(-1, GROUP), len(w.reshape(-1)), pad

def quant_kmeans(w, iters=12):
    """Per-group 1D k-means (vectorized over groups). Returns dequantized weight, same shape."""
    g, n, pad = _pad_groups(w.cpu().numpy())               # [G,64]
    s = np.sort(g, axis=1)
    qi = ((np.arange(LEVELS) + 0.5) * GROUP / LEVELS).astype(int)  # integer quantile indices
    c = s[:, qi].astype(np.float32)                        # [G,16] quantile init
    for _ in range(iters):
        d = np.abs(g[:, :, None] - c[:, None, :])           # [G,64,16]
        a = d.argmin(2)                                      # [G,64] assignments
        for e in range(LEVELS):                             # recompute centroids
            m = (a == e)
            cnt = m.sum(1)
            nz = cnt > 0
            c[nz, e] = (g * m).sum(1)[nz] / cnt[nz]
    a = np.abs(g[:, :, None] - c[:, None, :]).argmin(2)
    dq = np.take_along_axis(c, a, 1).reshape(-1)[:n]
    return torch.from_numpy(dq.reshape(w.shape)).to(w.dtype)

def quant_uniform(w):
    g, n, pad = _pad_groups(w.cpu().numpy())
    lo = g.min(1, keepdims=True); hi = g.max(1, keepdims=True)
    st = (hi - lo) / (LEVELS - 1) + 1e-12
    q = np.clip(np.round((g - lo) / st), 0, LEVELS - 1)
    dq = (lo + q * st).reshape(-1)[:n]
    return torch.from_numpy(dq.reshape(w.shape)).to(w.dtype)

def quantize_model(model, fn):
    """Quantize transformer Linear/Conv1D weights ONLY. Skips embeddings and the output
    head (standard practice -- they're sensitive and often tied), and dedups tied tensors."""
    try:
        from transformers.pytorch_utils import Conv1D            # gpt2-family uses Conv1D, not Linear
        LinearTypes = (torch.nn.Linear, Conv1D)
    except Exception:
        LinearTypes = (torch.nn.Linear,)
    SKIP = ("lm_head", "wte", "wpe", "embed", "shared", "embed_tokens")
    n, seen = 0, set()
    for name, mod in model.named_modules():
        if any(s in name for s in SKIP): continue
        if not isinstance(mod, LinearTypes): continue
        w = getattr(mod, "weight", None)
        if w is None or w.ndim != 2 or min(w.shape) < GROUP: continue
        if id(w) in seen: continue                                # don't double-quantize tied weights
        seen.add(id(w))
        with torch.no_grad():
            w.copy_(fn(w.data))
        n += 1
    return n

@torch.no_grad()
def perplexity(model, enc, ctx=512, stride=512):
    ids = enc.input_ids
    nll, ntok = 0.0, 0
    for i in range(0, ids.size(1) - 1, stride):
        a, b = i, min(i + ctx, ids.size(1) - 1)
        inp = ids[:, a:b]; tgt = ids[:, a+1:b+1]
        logits = model(inp).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        nll += loss.item(); ntok += tgt.numel()
    return float(np.exp(nll / ntok))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--maxtok", type=int, default=60000, help="cap eval tokens for tractable CPU run (0=all)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.model)
    txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    enc = tok(txt, return_tensors="pt")
    if args.maxtok and enc.input_ids.size(1) > args.maxtok:
        enc.input_ids = enc.input_ids[:, :args.maxtok]
        print(f"(eval capped to {args.maxtok} tokens for speed)")

    print(f"model={args.model}  group={GROUP}  4-bit codebook  (kernel is bit-exact to this)")
    rows = []
    for name, fn in [("fp32 baseline", None), ("k-means 4-bit (non-uniform)", quant_kmeans),
                     ("uniform int4", quant_uniform)]:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
        nq = quantize_model(model, fn) if fn else 0
        ppl = perplexity(model, enc, args.ctx)
        rows.append((name, ppl, nq)); del model
        print(f"  {name:30s} ppl={ppl:8.3f}  ({nq} linears quantized)")
    base = rows[0][1]
    print("\nperplexity delta vs fp32:")
    for name, ppl, _ in rows[1:]:
        print(f"  {name:30s} +{ppl-base:6.3f}  ({100*(ppl-base)/base:+.2f}%)")
    print("\n(non-uniform k-means should beat uniform int4 most where weights have outliers.)")

if __name__ == "__main__":
    main()
