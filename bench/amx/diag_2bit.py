#!/usr/bin/env python3
"""PROFILE the 2-bit codebook. End-to-end perplexity can't diagnose this (GPT-2 is
destroyed at 2-bit for every method), so we measure per-group WEIGHT reconstruction
error on REAL weight matrices -- that's where the codebook design shows itself.

Compares four 2-bit (4-level) codebooks, per-group, across group sizes:
  uniform     : affine min/max grid (adapts range per group)
  nf-zero     : normal-quantile, asymmetric-with-zero  [-1, 0, +.44, +1]   (current)
  nf-nozero   : normal-quantile, symmetric no-zero      [-1, -a, +a, +1]    (old)
  kmeans      : per-group data-fit 4 levels (MSE-optimal for that group)

Hypotheses: (H1) nf-zero is lopsided -> poor; (H2) kmeans wins; (H3) uniform beats NF
because it adapts range; (H4) finer groups help a lot at 2-bit.
Metric: relative Frobenius error ||W-Q||/||W|| (lower better), averaged over layers.
Also prints the actual level placements so we can SEE why.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

BITS = 2; N = 2 ** BITS

def levels_nf_zero(device):
    n = N; offset = 1 - 0.5 * (1/(2*n) + 1/(2*(n-1))); half = n // 2
    pos = torch.special.ndtri(torch.linspace(offset, 0.5, half+1, dtype=torch.float64)[:-1])
    neg = -torch.special.ndtri(torch.linspace(offset, 0.5, half, dtype=torch.float64)[:-1])
    lv = torch.cat([neg, torch.zeros(1, dtype=torch.float64), pos]).sort().values
    return (lv/lv.abs().max()).float().to(device)

def levels_nf_nozero(device):
    p = (torch.arange(N, dtype=torch.float64) + 0.5) / N
    lv = torch.special.ndtri(p); return (lv/lv.abs().max()).float().to(device)

@torch.no_grad()
def q_uniform(W, group):
    out, inn = W.shape; Q = W.clone()
    for c0 in range(0, inn, group):
        g = W[:, c0:c0+group]; lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True)
        st = (hi-lo)/(N-1) + 1e-12
        Q[:, c0:c0+group] = lo + torch.clamp(torch.round((g-lo)/st), 0, N-1)*st
    return Q

@torch.no_grad()
def q_codebook(W, group, levels):
    out, inn = W.shape; Q = W.clone()
    for c0 in range(0, inn, group):
        g = W[:, c0:c0+group]; s = g.abs().amax(1, keepdim=True).clamp_min(1e-8)
        idx = (g.unsqueeze(-1)/s.unsqueeze(-1) - levels).abs().argmin(-1)
        Q[:, c0:c0+group] = levels[idx]*s
    return Q

@torch.no_grad()
def q_kmeans(W, group, iters=12):
    """Per-group Lloyd on the scaled group (absmax scale), 4 centroids."""
    out, inn = W.shape; Q = W.clone()
    for c0 in range(0, inn, group):
        g = W[:, c0:c0+group]; s = g.abs().amax(1, keepdim=True).clamp_min(1e-8)
        x = (g/s)                                                 # [out, gw] in ~[-1,1]
        # init centroids at sorted quantiles per row
        gw = x.shape[1]; qi = ((torch.arange(N)+0.5)*gw/N).long().clamp(max=gw-1)
        c = torch.sort(x, 1).values[:, qi]                        # [out, N]
        for _ in range(iters):
            a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)   # [out, gw]
            for e in range(N):
                m = (a == e); cnt = m.sum(1); nz = cnt > 0
                num = (x*m).sum(1)
                c[nz, e] = num[nz]/cnt[nz]
        a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)
        Q[:, c0:c0+group] = torch.gather(c, 1, a) * s
    return Q

def relerr(W, Q): return (torch.linalg.norm(W-Q)/torch.linalg.norm(W)).item()

def layers(model, k=6):
    SKIP = ("lm_head","embed","wte","wpe","shared","embed_tokens")
    out = []
    for n, m in model.named_modules():
        if any(s in n for s in SKIP): continue
        if isinstance(m, (torch.nn.Linear, Conv1D)) and m.weight.ndim == 2 and min(m.weight.shape) >= 128:
            W = m.weight.data.t().contiguous() if isinstance(m, Conv1D) else m.weight.data
            out.append((n, W.float()))
            if len(out) >= k: break
    return out

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    dev = "cpu"
    print(f"model={model_name}  2-bit (4 levels)  metric=||W-Q||/||W|| (lower better)\n", flush=True)
    nf0 = levels_nf_zero(dev); nfn = levels_nf_nozero(dev)
    print("level placements (x per-group absmax scale):")
    print(f"  nf-zero   : {[round(x,3) for x in nf0.tolist()]}")
    print(f"  nf-nozero : {[round(x,3) for x in nfn.tolist()]}")
    print("  uniform   : per-group min..max (adapts)")
    print("  kmeans    : per-group data-fit\n", flush=True)

    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    L = layers(m, 6)
    for group in (128, 64, 32):
        errs = {k: [] for k in ("uniform","nf-zero","nf-nozero","kmeans")}
        for name, W in L:
            errs["uniform"].append(relerr(W, q_uniform(W, group)))
            errs["nf-zero"].append(relerr(W, q_codebook(W, group, nf0)))
            errs["nf-nozero"].append(relerr(W, q_codebook(W, group, nfn)))
            errs["kmeans"].append(relerr(W, q_kmeans(W, group)))
        print(f"group={group:3d}  " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in errs.items()), flush=True)
    print("\n(lower = better weight reconstruction; kmeans is the MSE-optimal 4-level ceiling)", flush=True)

if __name__ == "__main__":
    main()
