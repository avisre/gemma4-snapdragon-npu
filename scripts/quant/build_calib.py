#!/usr/bin/env python3
"""Build calibration data for AI Hub submit_quantize_job.

Two inputs to calibrate:
  input_ids        (1, 32) int64
  per_layer_inputs (1, 32, 35, 256) float16

We pull the per_layer_inputs from the model's own embed_tokens_per_layer
(Gemma-3 architecture) to get realistic activations.
"""
import os
import json
import pickle
import numpy as np
import torch

os.environ['TRANSFORMERS_OFFLINE'] = '1'

CKPT = "/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it"
OUT_PKL = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/calib_data.pkl"
N_PROMPTS = 16
SEQ_LEN = 32

# Diverse short prompts covering common LLM domains
PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank tonight.",
    "Hello world, today I would like to talk about machine learning and how",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci",
    "The mitochondria is the powerhouse of the cell. It produces ATP through",
    "To be or not to be, that is the question whether tis nobler in the",
    "Climate change is a global issue that requires immediate action from all",
    "Once upon a time in a faraway kingdom there lived a brave young princess",
    "The recipe calls for two cups of flour, one cup of sugar, three eggs",
    "Python is a high-level programming language widely used for data science",
    "The mountain rose majestically against the morning sky, snow glistening",
    "Quantum computing leverages superposition and entanglement to perform",
    "She opened the old wooden door and stepped into a room filled with",
    "import numpy as np\nimport torch\nfrom transformers import AutoModel",
    "The economic implications of artificial intelligence on labor markets are",
    "Dear customer, thank you for your recent purchase of our premium product",
]

print("Loading tokenizer...", flush=True)
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(CKPT)
cfg = AutoConfig.from_pretrained(CKPT)
text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
n_layers = text_cfg.num_hidden_layers
ple_dim = text_cfg.hidden_size_per_layer_input
print(f"n_layers={n_layers} ple_dim={ple_dim}", flush=True)

# Encode prompts
input_ids_list = []
for p in PROMPTS[:N_PROMPTS]:
    ids = tok.encode(p, add_special_tokens=True)
    if len(ids) < SEQ_LEN:
        ids = ids + [tok.pad_token_id or 0] * (SEQ_LEN - len(ids))
    else:
        ids = ids[:SEQ_LEN]
    input_ids_list.append(np.asarray(ids, dtype=np.int64).reshape(1, SEQ_LEN))

# Build per_layer_inputs using embed_tokens_per_layer when available; else random
print("Loading model for embed_tokens_per_layer...", flush=True)
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16, device_map="cpu")
    text_model = model.model if hasattr(model, "model") else model
    # Gemma-4 places embed_tokens_per_layer on the text/language model
    lm = getattr(text_model, "language_model", text_model)
    embed_per_layer = getattr(lm, "embed_tokens_per_layer", None)
    if embed_per_layer is None:
        # Search submodules
        for n, m in lm.named_modules():
            if "per_layer" in n and hasattr(m, "weight"):
                embed_per_layer = m
                print(f"  found per-layer module at {n}", flush=True)
                break
    if embed_per_layer is None:
        raise RuntimeError("Cannot find embed_tokens_per_layer; falling back to random.")
    print(f"embed_per_layer weight shape={embed_per_layer.weight.shape}", flush=True)
    use_random_ple = False
except Exception as e:
    print(f"Falling back to random per_layer_inputs: {e}", flush=True)
    use_random_ple = True
    embed_per_layer = None

ple_list = []
for ids_np in input_ids_list:
    if not use_random_ple:
        with torch.no_grad():
            ids_t = torch.from_numpy(ids_np).to(torch.long)
            ple = embed_per_layer(ids_t)  # (1, SEQ, n_layers*ple_dim) or (1, SEQ, n_layers, ple_dim)
            ple = ple.reshape(1, SEQ_LEN, n_layers, ple_dim).to(torch.float16).cpu().numpy()
    else:
        # Random fp16, small magnitude similar to embeddings
        rng = np.random.default_rng(42)
        ple = (rng.standard_normal((1, SEQ_LEN, n_layers, ple_dim)) * 0.02).astype(np.float16)
    ple_list.append(ple)

# AI Hub DatasetEntries format: dict[input_name] -> list[np.ndarray]
calib_data = {
    "input_ids": input_ids_list,
    "per_layer_inputs": ple_list,
}

print(f"Calib entries: input_ids={len(calib_data['input_ids'])} ple={len(calib_data['per_layer_inputs'])}", flush=True)
print(f"  ids[0] dtype={calib_data['input_ids'][0].dtype} shape={calib_data['input_ids'][0].shape}", flush=True)
print(f"  ple[0] dtype={calib_data['per_layer_inputs'][0].dtype} shape={calib_data['per_layer_inputs'][0].shape}", flush=True)

with open(OUT_PKL, "wb") as f:
    pickle.dump(calib_data, f)
print(f"Saved -> {OUT_PKL} ({os.path.getsize(OUT_PKL)/1e6:.1f} MB)", flush=True)
