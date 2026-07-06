#!/usr/bin/env python3
"""FAST fix-confirmation: does full-rank calibration flip GPTQ from harmful->helpful
on TinyLlama's MLP? Same held-out-Hessian metric as the diagnostic, but sampled
layers only (fast GPTQ solve) and a calibration-size sweep.

E(Q)=tr(D H D^T), D=W-Q.  ratio = E_gptq/E_heldout_nf4  (<1 GPTQ helps, >1 it hurts).
Paper calib (4096 tok) made down_proj heldout ratio ~1.7-2.0. If full-rank calib
drops it below 1, the weak 1.1B result was a calibration artifact -- confirmed.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# one attn + the two MLP layer types that failed, at early/mid/late depth
SAMPLE = {f"model.layers.{L}.{n}"
          for L in (0, 10, 21)
          for n in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")}

tok = AutoTokenizer.from_pretrained(MODEL)
train_txt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
def stream(lo, hi, ntok):
    return tok("\n\n".join(train_txt[lo:hi]), return_tensors="pt").input_ids[:, :ntok]
def load(): return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

def err(D, H): return float((D @ H * D).sum())

m = load()
nf4 = torch.tensor(cp.NF4_LEVELS, dtype=torch.float32)
mods = {n: (mod, conv) for n, mod, conv in cp._targets(m) if n in SAMPLE}

# sweep calibration size. paper=4096; full-rank for down_proj (in=5632) needs >5632.
for nseq, ctx, tag in [(8, 512, "paper 4096 tok (down_proj RANK-DEFICIENT)"),
                       (16, 2048, "32k tok (full-rank)")]:
    ntok = nseq*ctx
    H_tr = cp.collect_hessians(m, stream(0, 3000, ntok), ctx, nseq=nseq)
    H_ho = cp.collect_hessians(m, stream(3000, 6000, ntok), ctx, nseq=nseq)
    print(f"\n=== calib {tag} ===", flush=True)
    for n in sorted(mods):
        mod, conv = mods[n]
        if mod not in H_tr: continue
        W = (mod.weight.data.t().contiguous() if conv else mod.weight.data).float()
        Htr, Hho = H_tr[mod].float(), H_ho[mod].float()
        Q_nf4, Q_gptq = cp.quant_nf4(W).float(), cp.gptq_nf4(W, Htr).float()
        r_tr = err(W-Q_gptq, Htr)/max(err(W-Q_nf4, Htr), 1e-30)
        r_ho = err(W-Q_gptq, Hho)/max(err(W-Q_nf4, Hho), 1e-30)
        flag = "  <-- GPTQ HURTS" if r_ho > 1 else ""
        print(f"  {n:34s} in={W.shape[1]:5d}  train={r_tr:5.3f}  heldout={r_ho:5.3f}{flag}", flush=True)
