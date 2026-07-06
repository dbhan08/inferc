#!/usr/bin/env python3
"""DIAGNOSTIC: why does NF4+GPTQ barely beat plain NF4 on TinyLlama-1.1B
(+5.1% vs +5.4%) while on GPT-2 it beats it hugely (+2.9% vs +11.2%)?

Hypothesis: the paper's GPTQ calibration (nseq=8 x ctx=512 = 4096 tokens) is too
small for a 1.1B model. GPTQ's objective is to minimize the layer-output error
E(Q) = tr(D H D^T), D = W-Q, where H = X^T X is the calibration Hessian. If H is
undersampled/rank-deficient, GPTQ overfits it: it reduces error on the TRAIN
Hessian but not on a HELD-OUT Hessian -> no real perplexity gain.

We measure, per layer, the error-reduction ratio  E(Q_gptq)/E(Q_nf4)  on both a
TRAIN Hessian and an independent HELD-OUT Hessian. This needs only Hessian
collection + the GPTQ solve -- no full-model perplexity runs.

  train ratio << 1 AND heldout ratio << 1   -> GPTQ genuinely helps (good calib)
  train ratio << 1 BUT heldout ratio ~ 1    -> OVERFITTING (undersampled Hessian)  <-- the bug
  train ratio ~ 1                            -> numerical collapse (damping-dominated)
"""
import sys, os, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ap.add_argument("--ctx", type=int, default=512)      # paper setting
ap.add_argument("--nseq", type=int, default=8)       # paper setting -> 8*512 = 4096 tok
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
def load(): return AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()

# two DISJOINT calibration streams from wikitext train
train_txt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
def stream(lo, hi):
    return tok("\n\n".join(train_txt[lo:hi]), return_tensors="pt").input_ids[:, :a.nseq*a.ctx]
calib_tr = stream(0, 3000)
calib_ho = stream(3000, 6000)

def err(D, H):  # tr(D H D^T), the exact calibration layer-output MSE (x #samples)
    return float((D @ H * D).sum())

print(f"model={a.model}  calib = {a.nseq}x{a.ctx} = {a.nseq*a.ctx} tokens (paper setting)\n", flush=True)

# collect train + heldout Hessians on the SAME model instance
m = load()
print("collecting TRAIN Hessians...", flush=True)
H_tr = cp.collect_hessians(m, calib_tr, a.ctx, nseq=a.nseq)
print("collecting HELD-OUT Hessians...", flush=True)
H_ho = cp.collect_hessians(m, calib_ho, a.ctx, nseq=a.nseq)

nf4 = torch.tensor(cp.NF4_LEVELS, dtype=torch.float32)
rows = []
mods = [(n, mod, conv) for n, mod, conv in cp._targets(m)]
seen = set()
for name, mod, conv in mods:
    if id(mod.weight) in seen or mod not in H_tr: continue
    seen.add(id(mod.weight))
    W = (mod.weight.data.t().contiguous() if conv else mod.weight.data).float()
    Htr, Hho = H_tr[mod].float(), H_ho[mod].float()
    inn = W.shape[1]

    # rank / conditioning of the TRAIN Hessian
    ev = torch.linalg.eigvalsh(Htr)
    ev = ev.clamp_min(0)
    rank_frac = float((ev > 1e-6*ev.max()).sum()) / inn
    # NF4 (no error feedback) vs GPTQ
    Q_nf4 = cp.quant_nf4(W)          # per-64-group NF4, same grid GPTQ uses
    Q_gptq = cp.gptq_nf4(W, Htr)
    Dn, Dg = W - Q_nf4.float(), W - Q_gptq.float()
    r_tr = err(Dg, Htr) / max(err(Dn, Htr), 1e-30)
    r_ho = err(Dg, Hho) / max(err(Dn, Hho), 1e-30)
    rows.append((name, inn, rank_frac, r_tr, r_ho))
    print(f"  {name:34s} in={inn:5d} rank={rank_frac:5.2f} "
          f"train E_gptq/E_nf4={r_tr:5.3f}  heldout={r_ho:5.3f}", flush=True)

print("\nSUMMARY (geo-mean of error ratios, GPTQ vs NF4; <1 = GPTQ helps):", flush=True)
rt = np.exp(np.mean([np.log(r[3]) for r in rows]))
rh = np.exp(np.mean([np.log(r[4]) for r in rows]))
print(f"  train Hessian:    {rt:.3f}", flush=True)
print(f"  held-out Hessian: {rh:.3f}", flush=True)
print(f"  overfit gap (heldout/train): {rh/rt:.2f}x", flush=True)
dp = [r for r in rows if "down_proj" in r[0]]
if dp:
    print(f"\n  down_proj only (rank-deficient layers):", flush=True)
    print(f"    mean rank_frac={np.mean([r[2] for r in dp]):.2f}  "
          f"train={np.exp(np.mean([np.log(r[3]) for r in dp])):.3f}  "
          f"heldout={np.exp(np.mean([np.log(r[4]) for r in dp])):.3f}", flush=True)
