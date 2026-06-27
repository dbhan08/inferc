#!/usr/bin/env python3
"""Apples-to-apples 4-bit eval: proper blocked+act-order GPTQ on the FULL wikitext-2 test,
all on M1 CPU (no CUDA). Compares to the published per-channel table (GANQ arXiv 2501.12956):
OPT-125M FP16 27.65 | RTN 37.11 | GPTQ 31.08 (+12.4%) | OmniQuant 30.98 | GANQ 28.58 (+3.4%).
Our quantizer is hardware-independent (quality is portable); the kernel runs its output at
2.8-4.7x on M1. Goal: see how close our GPTQ gets, on a comparable eval (baseline ~27.65 => standard)."""
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

NF4 = torch.tensor([-1.,-0.6961928,-0.52507305,-0.39491749,-0.28444138,-0.18477343,-0.09105004,0.,
 0.0795803,0.1609302,0.2461123,0.33791524,0.44070983,0.562617,0.72295684,1.], dtype=torch.float32)

def _q(w, s):                                              # per-element NF4 quant, scale s broadcast
    return s * NF4[((w/s).unsqueeze(-1) - NF4).abs().argmin(-1)]

def gptq(W, H, group, actorder, blocksize=128, percdamp=0.01):
    W = W.clone().float(); H = H.clone().float(); out, inn = W.shape
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    perm = inv = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]; inv = torch.argsort(perm)
    i = torch.arange(inn); H[i, i] += percdamp*torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    chan = (group >= inn)
    scale_pc = W.abs().amax(1, keepdim=True).clamp_min(1e-8) if chan else None   # per-channel scale (once)
    Q = torch.zeros_like(W)
    for a in range(0, inn, blocksize):
        b = min(a+blocksize, inn); W1 = W[:, a:b].clone(); Q1 = torch.zeros_like(W1); E = torch.zeros_like(W1); Hi = Hinv[a:b, a:b]
        for j in range(b-a):
            col = a+j
            if chan: s = scale_pc[:, 0]
            else:
                if (col % group) == 0:
                    gg = W1[:, j:min(j+group, b-a)] if (col//group)==(a//group) else W[:, col:col+group]
                    s_cur = gg.abs().amax(1).clamp_min(1e-8)
                s = s_cur
            w = W1[:, j]; q = _q(w, s); Q1[:, j] = q; err = (w-q)/Hi[j, j]; E[:, j] = err
            if j+1 < b-a: W1[:, j+1:] -= err[:, None]*Hi[j, j+1:][None, :]
        Q[:, a:b] = Q1
        if b < inn: W[:, b:] -= E @ Hinv[a:b, b:]
    return Q[:, inv] if actorder else Q

def targets(model):
    from transformers.pytorch_utils import Conv1D
    SKIP = ("lm_head","embed","wte","wpe","shared","embed_tokens")
    for n, m in model.named_modules():
        if any(s in n for s in SKIP): continue
        if isinstance(m, (torch.nn.Linear, Conv1D)) and m.weight.ndim==2 and min(m.weight.shape)>=64:
            yield m, isinstance(m, Conv1D)

def hessians(model, calib, ctx, nseq=16):
    H, cnt, hk = {}, {}, []
    def mk(m):
        def h(mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1]); H[m] = (x.t()@x)+(H.get(m,0)); cnt[m]=x.shape[0]+cnt.get(m,0)
        return m.register_forward_pre_hook(h)
    ms = [m for m,_ in targets(model)]
    for m in ms: hk.append(mk(m))
    with torch.no_grad():
        for i in range(nseq):
            if (i+1)*ctx > calib.size(1): break
            model(calib[:, i*ctx:(i+1)*ctx])
    for h in hk: h.remove()
    return {m: H[m]/max(cnt[m],1) for m in H}

def quantize(model, hess, group, actorder):
    seen=set()
    for m, conv in targets(model):
        if id(m.weight) in seen or m not in hess: continue
        seen.add(id(m.weight))
        with torch.no_grad():
            W = m.weight.data.t().contiguous() if conv else m.weight.data
            Q = gptq(W, hess[m], group, actorder)
            m.weight.copy_(Q.t().contiguous() if conv else Q)

@torch.no_grad()
def ppl(model, ids, ctx):
    nll=ntok=0
    for i in range(0, ids.size(1)-1, ctx):
        b=min(i+ctx, ids.size(1)-1); lo=model(ids[:, i:b]).logits
        nll += torch.nn.functional.cross_entropy(lo.reshape(-1,lo.size(-1)), ids[:,i+1:b+1].reshape(-1), reduction="sum").item(); ntok += b-i
    return float(np.exp(nll/ntok))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="facebook/opt-125m"); ap.add_argument("--ctx",type=int,default=1024); ap.add_argument("--maxtok",type=int,default=0)
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model)
    ids=tok("\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"]),return_tensors="pt").input_ids
    if a.maxtok: ids=ids[:,:a.maxtok]
    calib=tok("\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"][:4000]),return_tensors="pt").input_ids[:,:16*a.ctx]
    ld=lambda: AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).eval()
    fp=ppl(ld(),ids,a.ctx)
    print(f"\nMODEL={a.model} ctx={a.ctx} tokens={ids.size(1)}  [GANQ tbl: FP16 27.65 | GPTQ 31.08/+12.4% | GANQ 28.58/+3.4%]",flush=True)
    print(f"  fp32 baseline                 ppl={fp:.3f}   (standard if ~27.65)",flush=True)
    for label, group, ao in [("GPTQ per-channel +act-order", 1<<30, True),
                             ("GPTQ per-channel  no-act",     1<<30, False)]:
        m=ld(); quantize(m, hessians(m,calib,a.ctx), group, ao); p=ppl(m,ids,a.ctx); del m
        print(f"  {label:30s} ppl={p:.3f}  (+{100*(p-fp)/fp:.2f}%)",flush=True)

if __name__=="__main__": main()
