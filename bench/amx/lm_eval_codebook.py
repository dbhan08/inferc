#!/usr/bin/env python3
"""Downstream zero-shot accuracy for the codebook-GEMM mechanism, via lm-evaluation-harness.

The AMX kernel is BIT-EXACT against A.dequant(W), so a model with its weights quantized-then-
dequantized in PyTorch == the model run through the kernel. We therefore quantize each Linear in
PyTorch with the chosen 4-bit method (reusing codebook_perplexity.py) and evaluate the resulting
model on standard zero-shot tasks. The comparison that matters: uniform 4-bit vs the non-uniform
NF4+GPTQ codebook (both run at the same kernel speed/energy) vs fp32.

Usage:
    pip install lm-eval
    python bench/amx/lm_eval_codebook.py --model gpt2 --method nf4_gptq --tasks lambada_openai,arc_easy,piqa --limit 200
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codebook_perplexity as cp
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_model(model_id, method, ctx=512):
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    if method == "fp32":
        return m, 0
    if method == "uniform":
        return m, cp.quantize_model(m, cp.quant_uniform)
    if method == "nf4":
        return m, cp.quantize_model(m, cp.quant_nf4)
    if method == "nf4_gptq":
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained(model_id)
        txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:3000])
        calib = tok(txt, return_tensors="pt").input_ids[:, :8*ctx]
        hess = cp.collect_hessians(m, calib, ctx)
        return m, cp.quantize_model_gptq(m, hess)
    raise ValueError(method)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--method", required=True, choices=["fp32", "uniform", "nf4", "nf4_gptq"])
    ap.add_argument("--tasks", default="lambada_openai,arc_easy,piqa")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    print(f"building {args.model} method={args.method} ...", flush=True)
    m, nq = build_model(args.model, args.method)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = HFLM(pretrained=m, tokenizer=tok, batch_size=1)
    res = simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit,
                          bootstrap_iters=0, verbosity="ERROR")
    out = {}
    for task, d in res["results"].items():
        metric = d.get("acc_norm,none", d.get("acc,none", d.get("perplexity,none")))
        out[task] = metric
    print(f"\n=== RESULT {args.model} method={args.method} ({nq} layers quantized) limit={args.limit} ===")
    for task, v in out.items():
        print(f"  {task:22s} {v:.4f}" if isinstance(v, float) else f"  {task:22s} {v}")
    print("JSON " + json.dumps({"model": args.model, "method": args.method, "limit": args.limit, "results": out}))

if __name__ == "__main__":
    main()
