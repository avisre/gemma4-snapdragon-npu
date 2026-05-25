"""CPU-only reference inference for Gemma 4 E2B.

Generates ground-truth artifacts so we can validate the NPU implementation:
  - tokenized input_ids
  - per_layer_inputs captured from `embed_tokens_per_layer` (shape: 1, seq, 35, 256)
  - logits for the first generated token
  - the full generated text

Run:
  /home/hardoker77/gemma4_e2b_v69/venv/bin/python3 \
      /home/hardoker77/gemma4_e2b_v69/runtime/cpu_reference.py
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it"
OUT_DIR = Path("/home/hardoker77/gemma4_e2b_v69/runtime/ref_outputs")
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32


def peak_rss_mb() -> float:
    """Peak resident set size in MiB for this process."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print(f"[cpu_reference] loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"[cpu_reference] loading model (bf16, cpu, low_cpu_mem_usage=True)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        max_memory={"cpu": "10GiB"},
    )
    model.eval()
    print(f"[cpu_reference] load took {time.time() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MiB")

    # --- Locate the text model so we can hook embed_tokens_per_layer ----------
    # Gemma4ForConditionalGeneration wraps a Gemma4TextModel via
    # `model.language_model.model` (or `model.model` for the text-only variant).
    text_model = None
    for path in (
        "model.language_model",        # Gemma4ForConditionalGeneration -> model -> language_model
        "language_model.model",
        "language_model",
        "model",
    ):
        node = model
        ok = True
        for part in path.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok and hasattr(node, "embed_tokens_per_layer"):
            text_model = node
            print(f"[cpu_reference] found embed_tokens_per_layer at model.{path}")
            break
    if text_model is None:
        raise RuntimeError("could not locate embed_tokens_per_layer on the model")

    captured = {}

    def hook(_module, _inputs, output):
        # embed_tokens_per_layer is a ScaledWordEmbedding -> shape (B, seq, V_pl)
        # The model later .reshape(B, seq, num_layers, hidden_per_layer).
        # We store both the raw output and the reshaped (1, seq, 35, 256) view.
        captured["raw"] = output.detach().to(torch.float32).cpu().numpy()

    handle = text_model.embed_tokens_per_layer.register_forward_hook(hook)

    # --- Tokenize -------------------------------------------------------------
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"[cpu_reference] prompt: {PROMPT!r}")
    print(f"[cpu_reference] input_ids shape: {tuple(input_ids.shape)}")
    print(f"[cpu_reference] input_ids: {input_ids[0].tolist()}")

    # --- Single forward pass to capture per_layer_inputs + logits -------------
    print("[cpu_reference] forward pass (prefill)")
    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
    print(f"[cpu_reference] forward took {time.time() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MiB")

    logits = out.logits  # (1, seq, vocab)
    # Logits at the last prompt position correspond to the first generated token.
    logits_first = logits[:, -1, :].to(torch.float32).cpu().numpy()
    print(f"[cpu_reference] logits shape: {tuple(logits.shape)}, first-token slice: {logits_first.shape}")

    # Reshape captured per_layer raw output -> (B, seq, num_layers, hidden_per_layer)
    raw = captured["raw"]
    text_cfg = model.config.get_text_config()
    n_layers = text_cfg.num_hidden_layers
    hidden_per_layer = text_cfg.hidden_size_per_layer_input
    per_layer_inputs = raw.reshape(raw.shape[0], raw.shape[1], n_layers, hidden_per_layer)
    print(f"[cpu_reference] per_layer_inputs shape: {per_layer_inputs.shape}")
    print(f"[cpu_reference] per_layer_inputs[0,0,0,:8]: {per_layer_inputs[0,0,0,:8].tolist()}")

    handle.remove()

    # --- Save artifacts -------------------------------------------------------
    np.save(OUT_DIR / "input_ids.npy", input_ids.cpu().numpy().astype(np.int64))
    np.save(OUT_DIR / "per_layer_inputs.npy", per_layer_inputs.astype(np.float32))
    np.save(OUT_DIR / "logits_first_token.npy", logits_first.astype(np.float32))
    print(f"[cpu_reference] wrote npy artifacts to {OUT_DIR}")

    # --- Greedy generation ----------------------------------------------------
    print(f"[cpu_reference] greedy generate(max_new_tokens={MAX_NEW_TOKENS})")
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    print(f"[cpu_reference] generate took {time.time() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MiB")

    full_text = tokenizer.decode(gen[0], skip_special_tokens=False)
    print(f"[cpu_reference] generated tokens: {gen[0].tolist()}")
    print(f"[cpu_reference] generated text:\n---\n{full_text}\n---")

    (OUT_DIR / "generated_text.txt").write_text(full_text)
    print(f"[cpu_reference] wrote {OUT_DIR / 'generated_text.txt'}")

    print(f"[cpu_reference] DONE. final peak RSS {peak_rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
