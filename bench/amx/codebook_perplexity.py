#!/usr/bin/env python3
"""Real-model perplexity for the AMX codebook-GEMM mechanism -- accurate AND fast.

The AMX kernel computes A.dequant(W) BIT-EXACTLY (verified in amx_codebook_gemm.cc), so
a model run through the kernel == the model with dequantized weights. Accuracy is therefore
measurable exactly in PyTorch; the kernel supplies the speed (4-5x over Accelerate). Every
method below produces per-group scalar codebooks the kernel runs bit-exactly.

Methods (all 4-bit, per-64-group, scalar -- the kernel's regime):
  uniform int4   : affine min/max grid (cheap, no codebook needed)
  k-means        : learned non-uniform codebook (weight-MSE)
  NF4            : QLoRA's fixed non-uniform levels + per-block absmax  (path B: standard SOTA scalar)
  NF4 + GPTQ     : NF4 grid with GPTQ error-feedback using a calib Hessian (path A: our own SOTA)
The point: error feedback (GPTQ) -- not a cleverer codebook -- is what reaches SOTA scalar 4-bit.

Run (you supply the model download):
    pip install torch transformers datasets
    python bench/amx/codebook_perplexity.py --model gpt2
"""
import argparse, numpy as np, torch

GROUP = 64
LEVELS = 16
# QLoRA NF4 levels (quantiles of a standard normal, normalized to [-1,1]).
NF4_LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

def _pad_groups(w):
    flat = w.reshape(-1).astype(np.float32)
    pad = (-len(flat)) % GROUP
    if pad: flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    return flat.reshape(-1, GROUP), len(w.reshape(-1)), pad

def quant_uniform(w):
    g, n, _ = _pad_groups(w.cpu().numpy())
    lo = g.min(1, keepdims=True); hi = g.max(1, keepdims=True); st = (hi-lo)/(LEVELS-1)+1e-12
    dq = (lo + np.clip(np.round((g-lo)/st), 0, LEVELS-1)*st).reshape(-1)[:n]
    return torch.from_numpy(dq.reshape(w.shape)).to(w.dtype)

def quant_kmeans(w, iters=12):
    g, n, _ = _pad_groups(w.cpu().numpy())
    qi = ((np.arange(LEVELS)+0.5)*GROUP/LEVELS).astype(int)
    c = np.sort(g, axis=1)[:, qi].astype(np.float32)
    for _ in range(iters):
        a = np.abs(g[:, :, None]-c[:, None, :]).argmin(2)
        for e in range(LEVELS):
            m = (a == e); cnt = m.sum(1); nz = cnt > 0
            c[nz, e] = (g*m).sum(1)[nz]/cnt[nz]
    a = np.abs(g[:, :, None]-c[:, None, :]).argmin(2)
    dq = np.take_along_axis(c, a, 1).reshape(-1)[:n]
    return torch.from_numpy(dq.reshape(w.shape)).to(w.dtype)

def quant_nf4(w):
    g, n, _ = _pad_groups(w.cpu().numpy())
    s = np.abs(g).max(1, keepdims=True); s[s < 1e-8] = 1e-8
    nf4 = np.array(NF4_LEVELS, np.float32)
    qi = np.abs((g/s)[:, :, None] - nf4[None, None, :]).argmin(2)
    dq = (nf4[qi]*s).reshape(-1)[:n]
    return torch.from_numpy(dq.reshape(w.shape)).to(w.dtype)

def gptq_nf4(W, H, group=GROUP, percdamp=0.01):
    """GPTQ error-feedback with a per-group NF4 grid. W:[out,in], H:[in,in] calib Hessian.
    Quantizes columns left-to-right, pushing each column's rounding error into the remaining
    columns via H^-1 -> minimizes ||X(W - Q)||, the actual layer-output error."""
    nf4 = torch.tensor(NF4_LEVELS, dtype=torch.float32)
    W = W.detach().float().clone(); H = H.detach().float().clone()
    out, inn = W.shape
    diag = torch.arange(inn)
    dead = torch.diag(H) == 0
    if dead.any(): H[dead, dead] = 1.0; W[:, dead] = 0.0
    H[diag, diag] += percdamp*torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for c0 in range(0, inn, group):
        c1 = min(c0+group, inn)
        s = W[:, c0:c1].abs().amax(1, keepdim=True).clamp_min(1e-8)   # per-row scale [out,1]
        grid = s*nf4[None, :]                                         # [out,16]
        for j in range(c0, c1):
            w = W[:, j]
            qi = (w[:, None]-grid).abs().argmin(1)
            q = torch.gather(grid, 1, qi[:, None]).squeeze(1)
            Q[:, j] = q
            err = (w-q)/Hinv[j, j]
            if j+1 < inn:
                W[:, j+1:] -= err[:, None]*Hinv[j, j+1:][None, :]
    return Q.to(W.dtype)

# ---- model plumbing ----
def _targets(model):
    from transformers.pytorch_utils import Conv1D
    Lin = (torch.nn.Linear, Conv1D)
    SKIP = ("lm_head", "wte", "wpe", "embed", "shared", "embed_tokens")
    for name, m in model.named_modules():
        if any(s in name for s in SKIP): continue
        if isinstance(m, Lin):
            w = getattr(m, "weight", None)
            if w is not None and w.ndim == 2 and min(w.shape) >= GROUP:
                yield name, m, isinstance(m, Conv1D)

def quantize_model(model, fn):
    n, seen = 0, set()
    for _, mod, _ in _targets(model):
        if id(mod.weight) in seen: continue
        seen.add(id(mod.weight))
        with torch.no_grad(): mod.weight.copy_(fn(mod.weight.data))
        n += 1
    return n

def collect_hessians(model, calib, ctx, nseq=8):
    H, cnt, hooks = {}, {}, []
    mods = [(m, conv) for _, m, conv in _targets(model)]
    def mk(m):
        def h(mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            H[m] = (x.t() @ x) + (H[m] if m in H else 0)
            cnt[m] = x.shape[0] + cnt.get(m, 0)
        return m.register_forward_pre_hook(h)
    for m, _ in mods: hooks.append(mk(m))
    with torch.no_grad():
        for i in range(nseq):
            a, b = i*ctx, (i+1)*ctx
            if b > calib.size(1): break
            model(calib[:, a:b])
    for h in hooks: h.remove()
    return {m: H[m]/max(cnt[m], 1) for m in H}

def quantize_model_gptq(model, hess):
    n, seen = 0, set()
    for _, mod, conv in _targets(model):
        if id(mod.weight) in seen or mod not in hess: continue
        seen.add(id(mod.weight))
        with torch.no_grad():
            W = mod.weight.data.t().contiguous() if conv else mod.weight.data
            Q = gptq_nf4(W, hess[mod])
            mod.weight.copy_(Q.t().contiguous() if conv else Q)
        n += 1
    return n

@torch.no_grad()
def perplexity(model, ids, ctx=512):
    nll, ntok = 0.0, 0
    for i in range(0, ids.size(1)-1, ctx):
        a, b = i, min(i+ctx, ids.size(1)-1)
        logits = model(ids[:, a:b]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, a+1:b+1].reshape(-1), reduction="sum")
        nll += loss.item(); ntok += b-a
    return float(np.exp(nll/ntok))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--maxtok", type=int, default=40000)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.model)
    test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(test, return_tensors="pt").input_ids
    if args.maxtok: ids = ids[:, :args.maxtok]
    calib = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:3000]),
                return_tensors="pt").input_ids[:, :8*args.ctx]

    print(f"model={args.model}  4-bit per-{GROUP}-group  (kernel runs all of these bit-exactly)")
    rows = []
    def load(): return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    for name, fn in [("fp32 baseline", None), ("uniform int4", quant_uniform),
                     ("k-means", quant_kmeans), ("NF4 (path B: standard)", quant_nf4)]:
        m = load(); nq = quantize_model(m, fn) if fn else 0
        p = perplexity(m, ids, args.ctx); rows.append((name, p)); del m
        print(f"  {name:26s} ppl={p:8.3f}  ({nq} quantized)")
    m = load(); hess = collect_hessians(m, calib, args.ctx); nq = quantize_model_gptq(m, hess)
    p = perplexity(m, ids, args.ctx); rows.append(("NF4 + GPTQ (path A: ours)", p)); del m
    print(f"  {'NF4 + GPTQ (path A: ours)':26s} ppl={p:8.3f}  ({nq} quantized)")

    base = rows[0][1]
    print("\nperplexity delta vs fp32:")
    for name, p in rows[1:]:
        print(f"  {name:26s} +{p-base:6.3f}  ({100*(p-base)/base:+.2f}%)")

if __name__ == "__main__":
    main()
