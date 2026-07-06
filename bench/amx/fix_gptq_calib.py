#!/usr/bin/env python3
"""TARGETED FIX TEST: does giving GPTQ enough calibration close the TinyLlama gap?

Paper calib = 8 x 512 = 4096 tokens -> down_proj Hessian (5632-dim) is rank-deficient.
Here we sweep calibration size and re-measure real WikiText-2 perplexity for NF4+GPTQ.
If GPTQ improves toward its GPT-2-style margin as calibration grows, the weak 1.1B
result was an undersampling artifact, not a property of the model or the method.

  python3 -u bench/amx/fix_gptq_calib.py                # TinyLlama, sweep calib
"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ap.add_argument("--maxtok", type=int, default=12000)   # eval tokens (perplexity)
ap.add_argument("--ctx", type=int, default=512)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(test, return_tensors="pt").input_ids[:, :a.maxtok]
train_txt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
def load(): return AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()

# reference points: fp32, uniform, plain NF4 (no calibration)
m = load(); fp = cp.perplexity(m, ids, a.ctx); del m
print(f"model={a.model} eval_tok={a.maxtok}", flush=True)
print(f"  fp32                     ppl={fp:8.3f}", flush=True)
m = load(); cp.quantize_model(m, cp.quant_uniform); pu = cp.perplexity(m, ids, a.ctx); del m
print(f"  uniform int4             ppl={pu:8.3f}  ({100*(pu-fp)/fp:+.2f}%)", flush=True)
m = load(); cp.quantize_model(m, cp.quant_nf4); pn = cp.perplexity(m, ids, a.ctx); del m
print(f"  NF4 (no calib)           ppl={pn:8.3f}  ({100*(pn-fp)/fp:+.2f}%)", flush=True)

# GPTQ at increasing calibration.  (nseq, ctx) -> tokens.  ctx=2048 = TinyLlama native.
configs = [(8, 512, "paper: 4096 tok"), (32, 2048, "64k tok, native ctx")]
for nseq, cctx, tag in configs:
    calib = tok("\n\n".join(train_txt[:4000]), return_tensors="pt").input_ids[:, :nseq*cctx]
    m = load()
    hess = cp.collect_hessians(m, calib, cctx, nseq=nseq)
    cp.quantize_model_gptq(m, hess)
    p = cp.perplexity(m, ids, a.ctx); del m, hess
    print(f"  NF4+GPTQ [{tag:22s}] ppl={p:8.3f}  ({100*(p-fp)/fp:+.2f}%)", flush=True)
