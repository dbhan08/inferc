#!/usr/bin/env python3
"""LEAN full-rank fix-confirmation. Only the 9 sampled layers are hooked, so only
their Hessians are allocated (<1GB) -- no all-154-layer 6GB blowup, no swap thrash.
Runs ONLY the full-rank (32k-token) calib; the paper-calib config is already recorded.

Confirms: with a full-rank Hessian, does GPTQ's held-out error ratio on down_proj
drop from ~1.7-2.0 (harmful, paper calib) to <1.0 (helpful)?
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NSEQ, CTX = 16, 2048                      # 32k tokens > 5632 => down_proj full-rank
SAMPLE = {f"model.layers.{L}.{n}"
          for L in (0, 10, 21)
          for n in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")}

tok = AutoTokenizer.from_pretrained(MODEL)
train_txt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
def stream(lo, hi):
    return tok("\n\n".join(train_txt[lo:hi]), return_tensors="pt").input_ids[:, :NSEQ*CTX]

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
mods = {n: mod for n, mod, conv in cp._targets(m) if n in SAMPLE}   # all Linear here (conv=False)

def collect(ids):
    """Hessian X^T X, ONLY for the sampled layers -> tiny memory."""
    H, cnt, hk = {}, {}, []
    def mk(name, mod):
        def h(_, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            H[name] = x.t() @ x + H.get(name, 0); cnt[name] = x.shape[0] + cnt.get(name, 0)
        return mod.register_forward_pre_hook(h)
    for n, mod in mods.items(): hk.append(mk(n, mod))
    with torch.no_grad():
        for i in range(NSEQ):
            if (i+1)*CTX > ids.size(1): break
            m(ids[:, i*CTX:(i+1)*CTX])
    for h in hk: h.remove()
    return {n: H[n]/max(cnt[n], 1) for n in H}

print(f"collecting full-rank Hessians ({NSEQ}x{CTX} = {NSEQ*CTX} tokens), 9 layers only...", flush=True)
H_tr = collect(stream(0, 3000))
H_ho = collect(stream(3000, 6000))
def err(D, H): return float((D @ H * D).sum())

print(f"\n=== full-rank calib ({NSEQ*CTX} tok) -- GPTQ vs plain NF4 ===", flush=True)
for n in sorted(mods):
    W = mods[n].weight.data.float()
    Htr, Hho = H_tr[n], H_ho[n]
    Q_nf4, Q_gptq = cp.quant_nf4(W).float(), cp.gptq_nf4(W, Htr).float()
    r_tr = err(W-Q_gptq, Htr)/max(err(W-Q_nf4, Htr), 1e-30)
    r_ho = err(W-Q_gptq, Hho)/max(err(W-Q_nf4, Hho), 1e-30)
    flag = "  <-- GPTQ HURTS" if r_ho > 1 else "  <-- GPTQ HELPS"
    print(f"  {n:34s} in={W.shape[1]:5d}  train={r_tr:5.3f}  heldout={r_ho:5.3f}{flag}", flush=True)
