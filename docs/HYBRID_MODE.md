# Hybrid Mode (Prefill + Decode with KV Cache)

The production-grade architecture. Replaces single-mode SHA (which re-prefills 32 tokens for every output token) with proper LLM serving: one big prefill at start, fast per-token decode after.

## Why this matters

| Mode | Per-token cost | tok/s estimate |
|---|---|---|
| **Single-mode SHA** (current shipped) | full 32-token prefill | ~0.1 tok/s |
| **Hybrid mode** (this) | 1-token decode w/ O(seq) attention | **3-15 tok/s** |

30-50× speedup. The model goes from "research artifact" to "interactive chat-capable".

## Architecture

Two ONNX graphs exported from the same SHA-patched model:

### Prefill graph
- **Input:** `input_ids[1, 128]`, `per_layer_inputs[1, 128, 35, 256]`
- **Output:** `logits[1, 128, 262144]` + initial KV cache (`k_l{0..14}`, `v_l{0..14}`)
- Runs once at the start of generation, chunked over the prompt
- Uses the full SHA per-head attention (as in single-mode)

### Decode graph
- **Input:** `input_ids[1, 1]`, `per_layer_inputs[1, 1, 35, 256]`, `past_k_l{0..14}[1, 1, 2047, head_dim]`, `past_v_l{0..14}[...]`
- **Output:** `logits[1, 1, 262144]`, `new_k_l{0..14}[1, 1, 2048, head_dim]`, `new_v_l{0..14}[...]`
- Runs once per output token
- Sliding-window concat of past + new KV (cap at 2047 past + 1 new = 2048 max context)

### Gemma 4-specific KV sharing
Gemma 4 has 35 layers but only **15 own KV** — the other 20 share KV from earlier layers (sliding window from layer 13, global from layer 14). This is exposed as explicit ONNX tensor references rather than duplicate cache slots.

- KV scratch buffers: 15 layers × 2 (K+V) = 30 tensors
- Per-layer size: `(1, 1, 2047, head_dim)` fp16, where head_dim is 256 (sliding) or 512 (global)
- Total persistent KV state: ~30 MB

## Output sizes (all 10 parts, validated under v69's 2 GB context limit)

| Part | Prefill .bin | Decode .bin |
|---|---|---|
| 0 | 1522 MB | (similar) |
| 1 | 987 MB | (similar) |
| 2 | 990 MB | (similar) |
| 3 | 880 MB | (similar) |
| 4 | 771 MB | (similar) |

Each part fits in v69 PD memory individually. Two parts cannot co-reside (v69 PD budget = ~1.5 GB), so the runner still does swap-load — but with hybrid, each token only triggers one decode chain instead of a full prefill, so swap-load is amortized over many fewer ops per generated token.

## Files

- **Export:** [`src/export/export_hybrid.py`](../src/export/export_hybrid.py) — produces both ONNX graphs
- **Split:** [`src/split/split_hybrid_5way.py`](../src/split/split_hybrid_5way.py) — splits each into 5 parts under 1.6 GB
- **Submit:** [`scripts/submit_hybrid.py`](../scripts/submit_hybrid.py) — uploads all 10 parts to AI Hub, polls, downloads
- **Deploy:** [`scripts/auto_push_test.sh`](../scripts/auto_push_test.sh) — auto-push each .bin to phone as it lands

## Runner integration (TODO)

The current `gemma4_runner.cpp` is single-mode. For hybrid:

1. **Load both 5-part sets** (10 binaries). Add `--decode_model_paths` flag mirroring `--model_paths` (which becomes prefill).

2. **Allocate KV scratch buffers** (~30 MB total persistent):
   ```cpp
   // Per OWN_KV layer:
   uint16_t k_scratch[15][1*1*2047*head_dim_for_layer];
   uint16_t v_scratch[15][1*1*2047*head_dim_for_layer];
   ```

3. **2-phase Generate()**:
   ```cpp
   // Phase A: prefill once over the prompt
   for chunk in chunks_of_128(input_ids):
       hidden, k_init, v_init = RunPrefillChain(chunk, per_layer_inputs_slice)
       update_kv_scratch(k_init, v_init)
   
   // Phase B: decode per output token
   for step in range(max_tokens):
       logits, k_new, v_new = RunDecodeChain(next_token_id, per_layer_inputs_slice,
                                              k_scratch, v_scratch)
       update_kv_scratch_slide(k_new, v_new)  # keep last 2047
       next_token_id = argmax(logits)
       emit(next_token_id)
   ```

4. **Drop RoPE host injection** for decode — decode graph folds rotary into per-layer constants, so the dead-code-elimination bug from single-mode doesn't apply.

5. **Drop SHA inter-part scratch buffers** (`buf_attn_*`) — hybrid graph contains full attention internally, no inter-part scratch needed beyond hidden states + KV cache.

Estimated work: 1-2 days of C++ runtime engineering. The hard work (export + compile) is done.

## All 10 job IDs (for reference)

See [`hybrid_job_ids.json`](hybrid_job_ids.json) for the full record. All compiled SUCCESS on Samsung Galaxy S22 5G OS 13 (Hexagon v69 reference device).

## Open issue: KV cache stride layout

The decode graph expects `past_k_l{i}` of shape `(1, 1, 2047, head_dim)` and outputs `new_k_l{i}` of shape `(1, 1, 2048, head_dim)`. The runner must:
- On first decode step: scratch is empty → pad with zeros for first 2046 positions, real token at position 2047
- On subsequent steps: shift scratch left by 1, drop position 0, append new K/V at position 2047

This is standard sliding-window KV cache management. Make sure the head_dim is per-layer (256 for sliding, 512 for full attention layers 4/9/14/19/24/29/34).
