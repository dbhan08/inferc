#!/usr/bin/env python3
# Perplexity at a chosen bitwidth: uniform grid vs a data-driven non-uniform (k-means) codebook,
# per-64-group. Tests the hypothesis that the codebook advantage over uniform GROWS as bits drop
# (2-bit is where non-uniform wins big). The AMX gather runs both 2-bit and 4-bit indices for free.
#   python3 -u bench/amx/ppl_bits.py <model> <bits> [maxtok]
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
bits = int(sys.argv[2]) if len(sys.argv) > 2 else 2
maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
ctx = 512
cp.LEVELS = 2 ** bits                      # quant_uniform / quant_kmeans read module-global LEVELS
tok = AutoTokenizer.from_pretrained(model)
ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]),
          return_tensors="pt").input_ids[:, :maxtok]
def load(): return AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()

print(f"model={model} bits={bits} (levels={cp.LEVELS}) maxtok={maxtok}", flush=True)
rows = []
for name, fn in [("fp32", None), (f"uniform {bits}b", cp.quant_uniform),
                 (f"codebook {bits}b (k-means)", cp.quant_kmeans)]:
    m = load(); nq = cp.quantize_model(m, fn) if fn else 0
    p = cp.perplexity(m, ids, ctx); rows.append((name, p)); del m
    print(f"  {name:26s} ppl={p:9.3f}  ({nq} layers)", flush=True)
base = rows[0][1]
print("delta vs fp32:", flush=True)
for name, p in rows[1:]:
    print(f"  {name:26s} {p-base:+9.3f}  ({100*(p-base)/base:+.1f}%)", flush=True)
