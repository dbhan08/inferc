"""Tokenize a fixed prompt for GPT-2 and generate golden logits via ORT.

Writes:
  models/gpt2_input_ids.bin       — int64, shape [1, N] (N depends on prompt)
  models/gpt2_golden_logits.bin   — float32, shape [1, N, 50257] (next-token logits at every position)
  models/gpt2_inputs_meta.txt     — human-readable summary

The fixed prompt is chosen to be:
  - short (low cost to iterate),
  - in-distribution for GPT-2 (English natural language),
  - with a "predictable" next-token that we can sanity-check.

inferc's correctness gate in session 10 will compare its position-N-1 logits
against `gpt2_golden_logits.bin`'s position-N-1 slice.

Later (session 11), the KV-cache greedy decode test will sample tokens from
the golden logits in a loop and verify inferc generates the same token IDs.

Run via: poetry run python scripts/make_gpt2_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

REPO_ID = "gpt2"
PROMPT = "The quick brown fox jumps over the lazy"  # next token should be " dog"
GREEDY_TOKENS = 32   # how many tokens to greedy-decode for the golden sequence
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ONNX_PATH = MODELS / "gpt2.onnx"
ONNX_PAST_PATH = MODELS / "gpt2_with_past.onnx"


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"error: {ONNX_PATH} not found. Run scripts/fetch_gpt2.py first.",
              file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    encoded = tokenizer(PROMPT, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)  # shape [1, N]
    N = input_ids.shape[1]

    print(f"Prompt:    {PROMPT!r}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Decoded:   {tokenizer.decode(input_ids[0])!r}")
    print(f"Shape:     {input_ids.shape}, N={N}")

    input_ids.tofile(MODELS / "gpt2_input_ids.bin")
    # Always write an all-ones attention mask matching input_ids; needed by
    # the Xenova/gpt2 ONNX which lists attention_mask as a graph input even
    # in the no-past-cache flavor.
    attention_mask = np.ones_like(input_ids).astype(np.int64)
    attention_mask.tofile(MODELS / "gpt2_attention_mask.bin")

    # Load the no-cache ONNX (full forward over the prompt).
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print("\nORT session loaded.")
    print("  inputs:  ", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("  outputs: ", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    # The Xenova/gpt2 decoder_model.onnx expects at minimum `input_ids`. It may
    # also expect `attention_mask` and/or `position_ids` — feed what it asks.
    feed = {"input_ids": input_ids}
    input_names = {i.name for i in sess.get_inputs()}
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.ones_like(input_ids).astype(np.int64)
    if "position_ids" in input_names:
        feed["position_ids"] = np.arange(N, dtype=np.int64).reshape(1, N)

    outs = sess.run(None, feed)
    logits = outs[0].astype(np.float32)  # shape [1, N, 50257]
    print(f"\nLogits: shape {logits.shape}, dtype {logits.dtype}")

    # Argmax-sample the next token (predict what comes after position N-1).
    next_token_id = int(np.argmax(logits[0, -1]))
    next_token_str = tokenizer.decode([next_token_id])
    print(f"Predicted next token id: {next_token_id}  decoded: {next_token_str!r}")

    logits.tofile(MODELS / "gpt2_golden_logits.bin")

    # --- Greedy-decode N tokens using the with-past model, save as golden ---
    if ONNX_PAST_PATH.exists():
        print(f"\nGreedy-decoding {GREEDY_TOKENS} tokens via gpt2_with_past.onnx ...")
        sess_past = ort.InferenceSession(str(ONNX_PAST_PATH),
                                         providers=["CPUExecutionProvider"])
        past_input_names = [i.name for i in sess_past.get_inputs()]
        n_layers = sum(1 for n in past_input_names
                       if n.endswith(".key") and n.startswith("past_key_values."))
        print(f"  with-past inputs: {len(past_input_names)} ({n_layers} KV layers)")

        # The prefill already gave us logits + all `present.*` cache tensors.
        # Build the initial cache (present.* from the prefill outputs).
        out_names = [o.name for o in sess.get_outputs()]
        present_map = {}
        for name, out in zip(out_names, outs):
            if name.startswith("present."):
                present_map[name] = out
        # The keys we need are `past_key_values.N.{key,value}`; rename.
        cache = {}
        for k, v in present_map.items():
            past_k = k.replace("present.", "past_key_values.")
            cache[past_k] = v

        generated = []
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
        cur_seq_len = N  # already saw N tokens during prefill

        for step in range(GREEDY_TOKENS - 1):
            cur_seq_len += 1
            feed_past = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((1, cur_seq_len), dtype=np.int64),
                **cache,
            }
            out_past = sess_past.run(None, feed_past)
            # Outputs: logits + present.*.
            out_names_past = [o.name for o in sess_past.get_outputs()]
            step_logits = out_past[0]   # shape [1, 1, 50257]
            next_token = int(np.argmax(step_logits[0, 0]))
            generated.append(next_token)
            # Rebuild cache from the present_* outputs.
            cache = {}
            for name, arr in zip(out_names_past, out_past):
                if name.startswith("present."):
                    cache[name.replace("present.", "past_key_values.")] = arr

        full_continuation = input_ids[0].tolist() + generated
        print(f"  decoded: {tokenizer.decode(full_continuation)!r}")
        np.array(generated, dtype=np.int64).tofile(MODELS / "gpt2_golden_tokens.bin")
        print(f"  saved gpt2_golden_tokens.bin ({len(generated)} tokens)")
    else:
        print("(skipping greedy-decode golden — gpt2_with_past.onnx not found)")
        generated = []

    meta = MODELS / "gpt2_inputs_meta.txt"
    meta.write_text(
        f"prompt: {PROMPT!r}\n"
        f"token_ids: {input_ids[0].tolist()}\n"
        f"decoded: {tokenizer.decode(input_ids[0])!r}\n"
        f"N: {N}\n"
        f"input_ids: int64 [1, {N}]\n"
        f"golden_logits: float32 [1, {N}, {logits.shape[-1]}]\n"
        f"predicted_next_token: {next_token_id} ({next_token_str!r})\n"
        f"greedy_tokens_generated: {len(generated)}\n"
        f"greedy_tokens: {generated}\n"
        f"model: {ONNX_PATH.name}\n"
        f"ort_inputs: {sorted(input_names)}\n"
    )
    print(f"\nSaved fixtures to {MODELS}/")
    print(f"  gpt2_input_ids.bin       ({input_ids.nbytes} bytes)")
    print(f"  gpt2_golden_logits.bin   ({logits.nbytes} bytes)")
    print(f"  gpt2_inputs_meta.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
