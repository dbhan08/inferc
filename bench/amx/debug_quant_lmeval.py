#!/usr/bin/env python3
# Debug: does the lm-eval path quantize the SAME as the perplexity path, and does HFLM keep the
# quantized weights? If direct perplexity here matches ppl_fast (uniform~12.36, nf4~11.92) AND
# HFLM preserves the weights, the downstream null is genuinely sample-size. If not, it's a bug.
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"; ctx = 512
tok = AutoTokenizer.from_pretrained(model)
ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]),
          return_tensors="pt").input_ids[:, :4000]
def load(): return AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()

# baseline weight fingerprint (first quantizable layer)
base = load()
fp_layer_name = next(n for n, m, _ in cp._targets(base))
def wfp(model):
    for n, m, _ in cp._targets(model):
        return float(m.weight.float().sum().item()), tuple(m.weight.shape)
print(f"fingerprint layer: {fp_layer_name}", flush=True)
print(f"fp32 weight-sum: {wfp(base)}", flush=True)

for name, fn in [("uniform", cp.quant_uniform), ("nf4", cp.quant_nf4)]:
    m = load(); nq = cp.quantize_model(m, fn)
    ws, shp = wfp(m)
    p_direct = cp.perplexity(m, ids, ctx)
    # now wrap in HFLM exactly as the eval does, and re-fingerprint + re-perplex through lm.model
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=m, tokenizer=tok, batch_size=1)
    ws_h, _ = wfp(lm.model)
    p_hflm = cp.perplexity(lm.model, ids, ctx)
    dtype = next(lm.model.parameters()).dtype
    same_obj = lm.model is m
    print(f"[{name}] layers={nq} wsum={ws:.4f}  direct_ppl={p_direct:.3f}  "
          f"HFLM.wsum={ws_h:.4f} HFLM.ppl={p_hflm:.3f} dtype={dtype} same_obj={same_obj}", flush=True)
    del m, lm
