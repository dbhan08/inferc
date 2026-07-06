#!/usr/bin/env python3
# Lean, UNBUFFERED WikiText-2 perplexity for the codebook methods that matter:
# fp32, uniform int4, NF4 codebook, NF4+GPTQ. Skips k-means (slow, not headline).
# Prints each result as it completes (run with python3 -u). Usage:
#   python3 -u bench/amx/ppl_fast.py <model> [maxtok]
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
maxtok = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
ctx = 512
tok = AutoTokenizer.from_pretrained(model)
ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]),
          return_tensors="pt").input_ids[:, :maxtok]
def load(): return AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()

print(f"model={model} maxtok={maxtok} (per-64-group 4-bit, bit-exact on kernel)", flush=True)
rows = []
for name, fn in [("fp32", None), ("uniform int4", cp.quant_uniform), ("NF4 codebook", cp.quant_nf4)]:
    m = load(); nq = cp.quantize_model(m, fn) if fn else 0
    p = cp.perplexity(m, ids, ctx); rows.append((name, p)); del m
    print(f"  {name:14s} ppl={p:8.3f}  ({nq} layers)", flush=True)

calib = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:2000]),
            return_tensors="pt").input_ids[:, :8*ctx]
m = load(); print("  collecting GPTQ Hessians...", flush=True)
hess = cp.collect_hessians(m, calib, ctx); print("  running GPTQ...", flush=True)
nq = cp.quantize_model_gptq(m, hess)
p = cp.perplexity(m, ids, ctx); rows.append(("NF4+GPTQ", p))
print(f"  {'NF4+GPTQ':14s} ppl={p:8.3f}  ({nq} layers)", flush=True)

base = rows[0][1]
print("delta vs fp32:", flush=True)
for name, p in rows[1:]:
    print(f"  {name:14s} {p-base:+7.3f}  ({100*(p-base)/base:+.2f}%)", flush=True)
