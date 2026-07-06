#!/usr/bin/env python3
"""SOTA-aligned 4/3/2-bit codebook-vs-uniform evaluation harness.

Produces the numbers the paper needs, on the protocol the quantization field uses
(GPTQ/AWQ/SqueezeLLM/QuIP#):
  * models     : any HF causal LM (default TinyLlama; use Llama-2-7B / Mistral-7B on a GPU)
  * calibration: C4 (allenai/c4), 128 x 2048 = 262k tokens by default  -> full-rank Hessian,
                 no WikiText leakage (the two fixes from the calibration-bug investigation)
  * eval       : WikiText-2 AND C4 perplexity at seq-len 2048
  * methods    : uniform / NF-codebook / NF+GPTQ  at bit-widths 4,3,2
  * grouping   : per-group (default 128; 64 also), the standard knob
  * GPTQ       : blocked + activation-order + Cholesky-inverse error feedback (the real algo)
  * device     : CUDA if available else CPU (bit-exact-portable; accuracy is device-independent)

Because the AMX kernel is bit-exact vs A.dequant(W), perplexity computed here in PyTorch
equals what the kernel produces -- so this file IS the accuracy table.

Examples
  # local smoke test (small model, tiny calib, wikitext calib to skip the C4 download):
  python3 -u bench/amx/eval_sota.py --model gpt2 --bits 4 --groups 128 \
      --calib-source wikitext --calib-tokens 8192 --eval-tokens 8000

  # the real run (rent a 24GB+ GPU):
  python3 -u bench/amx/eval_sota.py --model meta-llama/Llama-2-7b-hf \
      --bits 4 3 2 --groups 128 64 --calib-tokens 262144 --out results_llama2_7b.csv
"""
import argparse, os, sys, csv, math
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# QLoRA NF4 (exact) -- used verbatim at 4-bit; other bit-widths use normal quantiles below.
NF4 = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
       0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0]

def nf_levels(bits, device, dtype=torch.float32):
    """Non-uniform 'NormalFloat' codebook, 2^bits normal-quantile levels in [-1,1].
    Exactly QLoRA NF4 at bits==4; for other bit-widths, the QLoRA construction generalized:
    two asymmetric halves that SHARE an exact zero level, so near-zero weights (which
    dominate) stay representable -- critical at 2/3-bit, where a zero-less symmetric grid
    loses to uniform."""
    n = 2 ** bits
    if bits == 4:
        lv = torch.tensor(NF4, dtype=torch.float64)
    else:
        # offset keeps the extreme level off the (infinite) normal tail; the formula
        # reduces to QLoRA's 0.9677083 at bits==4.
        offset = 1 - 0.5 * (1 / (2 * n) + 1 / (2 * (n - 1)))
        half = n // 2
        pos = torch.special.ndtri(torch.linspace(offset, 0.5, half + 1, dtype=torch.float64)[:-1])
        neg = -torch.special.ndtri(torch.linspace(offset, 0.5, half, dtype=torch.float64)[:-1])
        lv = torch.cat([neg, torch.zeros(1, dtype=torch.float64), pos]).sort().values
    lv = (lv / lv.abs().max()).to(dtype)
    return lv.to(device)

# ---------------- quantizers (per-group along the input dimension, per output row) ----------
@torch.no_grad()
def quant_uniform(W, bits, group):
    """Affine min/max grid, 2^bits levels, per (row, contiguous input group)."""
    n = 2 ** bits
    out, inn = W.shape
    Q = W.clone()
    for c0 in range(0, inn, group):
        c1 = min(c0 + group, inn)
        g = W[:, c0:c1]
        lo = g.amin(1, keepdim=True); hi = g.amax(1, keepdim=True)
        st = (hi - lo) / (n - 1) + 1e-12
        Q[:, c0:c1] = lo + torch.clamp(torch.round((g - lo) / st), 0, n - 1) * st
    return Q

@torch.no_grad()
def quant_codebook(W, bits, group, levels):
    """Nearest non-uniform level, per (row, input group) absmax scale."""
    out, inn = W.shape
    Q = W.clone()
    for c0 in range(0, inn, group):
        c1 = min(c0 + group, inn)
        g = W[:, c0:c1]
        s = g.abs().amax(1, keepdim=True).clamp_min(1e-8)
        idx = (g.unsqueeze(-1) / s.unsqueeze(-1) - levels).abs().argmin(-1)
        Q[:, c0:c1] = levels[idx] * s
    return Q

@torch.no_grad()
def quant_kmeans(W, bits, group, iters=12):
    """Per-group data-fit codebook (Lloyd's algorithm on the absmax-scaled group).
    The MSE-optimal 2^bits levels for each group -- needed at low bit-widths, where a
    FIXED normal-quantile grid can't adapt (profiled: fixed NF loses to uniform at 2-bit,
    k-means beats it ~40%). The AMX kernel supports per-group codebooks via register reload."""
    n = 2 ** bits; out, inn = W.shape; Q = W.clone()
    for c0 in range(0, inn, group):
        c1 = min(c0 + group, inn)
        g = W[:, c0:c1]; s = g.abs().amax(1, keepdim=True).clamp_min(1e-8)
        x = g / s; gw = x.shape[1]
        qi = ((torch.arange(n, device=W.device) + 0.5) * gw / n).long().clamp(max=gw - 1)
        c = torch.sort(x, 1).values[:, qi]                       # quantile init [out, n]
        for _ in range(iters):
            a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)
            for e in range(n):
                m = (a == e); cnt = m.sum(1); nz = cnt > 0
                c[nz, e] = (x * m).sum(1)[nz] / cnt[nz]
        a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)
        Q[:, c0:c1] = torch.gather(c, 1, a) * s
    return Q

def _fit_grid(cols, bits, levels, codebook, dev, iters=8):
    """Build a per-row codebook grid [out, 2^bits] for one group of columns.
    codebook='nf'     -> fixed normal-quantile levels x per-row absmax scale (good >=3-bit).
    codebook='kmeans' -> per-row Lloyd fit to the group (needed at 2-bit; adapts range+shape)."""
    n = 2 ** bits
    s = cols.abs().amax(1, keepdim=True).clamp_min(1e-8)          # [out,1]
    if codebook == "nf":
        return s * levels[None, :]                               # [out, n]
    x = cols / s; gw = x.shape[1]
    qi = ((torch.arange(n, device=dev) + 0.5) * gw / n).long().clamp(max=gw - 1)
    c = torch.sort(x, 1).values[:, qi]                           # [out, n] quantile init
    for _ in range(iters):
        a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)
        for e in range(n):
            m = (a == e); cnt = m.sum(1); nz = cnt > 0
            c[nz, e] = (x * m).sum(1)[nz] / cnt[nz]
    return c * s                                                 # [out, n]

@torch.no_grad()
def gptq(W, H, bits, group, levels, codebook="nf", actorder=True, blocksize=128, percdamp=0.01):
    """Blocked + act-order GPTQ error feedback. The per-group grid is either the fixed NF
    codebook (codebook='nf') or a per-group k-means codebook (codebook='kmeans') -- the
    latter is the low-bit recipe (profiled: fixed grids fail at 2-bit)."""
    W = W.clone().float(); H = H.clone().float(); out, inn = W.shape
    dev = W.device; levels = levels.to(dev).float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]; inv = torch.argsort(perm)
    i = torch.arange(inn, device=dev); H[i, i] += percdamp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); grid = None
    for a in range(0, inn, blocksize):
        b = min(a + blocksize, inn)
        W1 = W[:, a:b].clone(); Q1 = torch.zeros_like(W1); E = torch.zeros_like(W1)
        Hi = Hinv[a:b, a:b]
        for j in range(b - a):
            col = a + j
            if (col % group) == 0:                    # rebuild the per-row grid at group start
                grid = _fit_grid(W[:, col:min(col + group, inn)], bits, levels, codebook, dev)
            w = W1[:, j]
            qi = (w[:, None] - grid).abs().argmin(1)  # nearest grid level per row
            q = torch.gather(grid, 1, qi[:, None]).squeeze(1)
            Q1[:, j] = q; err = (w - q) / Hi[j, j]; E[:, j] = err
            if j + 1 < b - a:
                W1[:, j + 1:] -= err[:, None] * Hi[j, j + 1:][None, :]
        Q[:, a:b] = Q1
        if b < inn:
            W[:, b:] -= E @ Hinv[a:b, b:]
    return (Q[:, inv] if actorder else Q).to(W.dtype)

# ---------------- model plumbing ----------------
def targets(model):
    from transformers.pytorch_utils import Conv1D
    SKIP = ("lm_head", "embed", "wte", "wpe", "shared", "embed_tokens")
    for n, m in model.named_modules():
        if any(s in n for s in SKIP): continue
        if isinstance(m, (torch.nn.Linear, Conv1D)) and m.weight.ndim == 2 and min(m.weight.shape) >= 64:
            yield n, m, isinstance(m, Conv1D)

@torch.no_grad()
def quantize_rtn(model, method, bits, group, dev):
    """uniform / codebook (fixed NF) / kmeans (data-fit): no calibration, quantize in place."""
    levels = nf_levels(bits, dev) if method == "codebook" else None
    seen, nq = set(), 0
    for _, m, conv in targets(model):
        if id(m.weight) in seen: continue
        seen.add(id(m.weight))
        W = (m.weight.data.t().contiguous() if conv else m.weight.data).to(dev).float()
        if method == "uniform":    Q = quant_uniform(W, bits, group)
        elif method == "kmeans":   Q = quant_kmeans(W, bits, group)
        else:                      Q = quant_codebook(W, bits, group, levels)
        m.weight.copy_((Q.t().contiguous() if conv else Q).to(m.weight.dtype))
        nq += 1
    return nq

@torch.no_grad()
def collect_hessians(model, calib, ctx, dev):
    # keyed by layer NAME (stable across model reloads), not module identity
    H, cnt, hk = {}, {}, []
    def mk(name, m):
        def h(mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            H[name] = (x.t() @ x) + H.get(name, 0); cnt[name] = x.shape[0] + cnt.get(name, 0)
        return m.register_forward_pre_hook(h)
    for name, m, _ in targets(model): hk.append(mk(name, m))
    n = calib.size(1) // ctx
    for i in range(n):
        model(calib[:, i * ctx:(i + 1) * ctx].to(dev))
    for h in hk: h.remove()
    return {name: H[name] / max(cnt[name], 1) for name in H}

@torch.no_grad()
def quantize_gptq(model, hess, bits, group, dev, codebook="nf"):
    levels = nf_levels(bits, dev)
    seen, nq = set(), 0
    for name, m, conv in targets(model):
        if id(m.weight) in seen or name not in hess: continue
        seen.add(id(m.weight))
        W = (m.weight.data.t().contiguous() if conv else m.weight.data).to(dev)
        Q = gptq(W, hess[name].to(dev), bits, group, levels, codebook=codebook)
        m.weight.copy_((Q.t().contiguous() if conv else Q).to(m.weight.dtype))
        nq += 1
    return nq

@torch.no_grad()
def perplexity(model, ids, ctx, dev):
    nll, ntok = 0.0, 0
    for i in range(0, ids.size(1) - 1, ctx):
        b = min(i + ctx, ids.size(1) - 1)
        logits = model(ids[:, i:b].to(dev)).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, i + 1:b + 1].reshape(-1).to(dev), reduction="sum")
        nll += loss.item(); ntok += b - i
    return float(np.exp(nll / ntok))

# ---------------- data ----------------
def wikitext2_test(tok, ctx, maxtok):
    txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(txt, return_tensors="pt").input_ids
    return ids[:, :maxtok] if maxtok else ids

def c4_stream_tokens(tok, ntokens, split, ctx):
    """Grab ~ntokens tokens from C4 (streaming, no full download)."""
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    chunks, have = [], 0
    for ex in ds:
        e = tok(ex["text"], return_tensors="pt").input_ids
        chunks.append(e); have += e.size(1)
        if have >= ntokens + ctx: break
    ids = torch.cat(chunks, 1)
    return ids[:, :((ntokens // ctx) * ctx if ntokens else ids.size(1))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--bits", type=int, nargs="+", default=[4])
    ap.add_argument("--groups", type=int, nargs="+", default=[128])
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--calib-source", choices=["c4", "wikitext"], default="c4")
    ap.add_argument("--calib-tokens", type=int, default=262144)   # 128 x 2048 (SOTA)
    ap.add_argument("--eval-tokens", type=int, default=0)         # 0 = full wikitext-2 test
    ap.add_argument("--eval-c4", action="store_true", help="also report C4 perplexity")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model)
    maxpos = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
    if maxpos and a.ctx > maxpos:
        print(f"  clamping ctx {a.ctx} -> {maxpos} (model max position)", flush=True)
        a.ctx = maxpos
    print(f"device={dev}  model={a.model}  bits={a.bits}  groups={a.groups}  "
          f"calib={a.calib_tokens}tok/{a.calib_source}  ctx={a.ctx}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    def load():
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).eval()
        return m.to(dev)

    print("loading eval data...", flush=True)
    wt = wikitext2_test(tok, a.ctx, a.eval_tokens)
    c4_eval = c4_stream_tokens(tok, a.eval_tokens or 262144, "validation", a.ctx) if a.eval_c4 else None
    if a.calib_source == "c4":
        calib = c4_stream_tokens(tok, a.calib_tokens, "train", a.ctx)
    else:
        txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:8000])
        calib = tok(txt, return_tensors="pt").input_ids[:, :a.calib_tokens]
    print(f"  wikitext2 eval tokens={wt.size(1)}  calib tokens={calib.size(1)}", flush=True)

    rows = []
    def rec(method, bits, group, ppl_wt, ppl_c4, base_wt):
        d = 100 * (ppl_wt - base_wt) / base_wt
        rows.append(dict(model=a.model, method=method, bits=bits, group=group,
                         wikitext2=round(ppl_wt, 4), wikitext2_delta_pct=round(d, 2),
                         c4=round(ppl_c4, 4) if ppl_c4 else ""))
        c4s = f"  c4={ppl_c4:.3f}" if ppl_c4 else ""
        print(f"  {method:9s} {bits}b g{group:<4d}  wikitext2={ppl_wt:8.3f} ({d:+.2f}%){c4s}", flush=True)
        if a.out:
            with open(a.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # fp32/fp16 reference
    m = load(); base_wt = perplexity(m, wt, a.ctx, dev)
    base_c4 = perplexity(m, c4_eval, a.ctx, dev) if c4_eval is not None else None
    del m; rec("reference", "-", 0, base_wt, base_c4, base_wt)

    for bits in a.bits:
        for group in a.groups:
            # RTN methods (no calibration)
            for method in ("uniform", "codebook", "kmeans"):
                m = load(); quantize_rtn(m, method, bits, group, dev)
                pw = perplexity(m, wt, a.ctx, dev)
                pc = perplexity(m, c4_eval, a.ctx, dev) if c4_eval is not None else None
                del m; rec(method, bits, group, pw, pc, base_wt)
            # GPTQ (calibrated): NF grid at 4-bit, k-means grid at <=3-bit (per the profile).
            # Collect the Hessian once, reuse it for both GPTQ codebook variants.
            m = load()
            print(f"  [collecting Hessians for {bits}b g{group}...]", flush=True)
            hess = collect_hessians(m, calib, a.ctx, dev)
            hcpu = {k: v.cpu() for k, v in hess.items()}; del hess
            for cb, tag in (("nf", "nf+gptq"), ("kmeans", "km+gptq")):
                mm = load(); h = {k: v.to(dev) for k, v in hcpu.items()}
                quantize_gptq(mm, h, bits, group, dev, codebook=cb)
                pw = perplexity(mm, wt, a.ctx, dev)
                pc = perplexity(mm, c4_eval, a.ctx, dev) if c4_eval is not None else None
                del mm, h; rec(tag, bits, group, pw, pc, base_wt)
            del hcpu

    print(f"\nwrote {a.out}" if a.out else "\ndone", flush=True)

if __name__ == "__main__":
    main()
