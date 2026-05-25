# Quantization scripts

Variants to try (in order of recommended ship priority):

| Variant | Type | Expected size | Expected accuracy | Fits v69? |
|---|---|---|---|---|
| **A** | PTQ W8A16 with calibration | ~3 GB | < 5% perplexity hit | 2 parts |
| **D** | W8A16 (no calib) | ~3 GB | 5-10% hit | 2 parts |
| **B** | INT8 full (8-bit activations) | ~2.5 GB | 10-30% hit | 2 parts |
| **C** | INT4 / W4A16 | ~1.4 GB | 5-15% hit | **1 part!** |

## How to use

```bash
export QAI_HUB_API_TOKEN=your_token
export TMPDIR=/path/with/space

# 1. Build calibration data (one-time, ~30 sec)
python3 build_calib.py

# 2. Upload the 5 GB ONNX ONCE (10-30 min depending on network)
python3 upload_once.py
# This writes uploaded_model.json with the cached model_id

# 3. Submit all 4 variants from the cached model (instant)
python3 submit_all_variants.py
# Or submit individually:
#   python3 submit_quant_BC.py    # Variants B + C
#   python3 submit_variant_A.py   # Variant A (with calibration)

# 4. Poll until all done (~20-30 min each compile)
python3 poll_jobs.py
# Downloads successes to ./downloads/

# 5. Push each variant to phone and test loadability
for variant in A B C D; do
    ./test_on_phone.sh ./downloads/${variant}_*.bin ${variant}_v69
done
```

## Why the upload is the bottleneck

The 5 GB ONNX has to upload once per variant unless we reuse the cached `model_id`. Each upload is 20+ min on a ~5 MB/s connection with multipart retry overhead.

The `upload_once.py` + `get_model(id)` pattern uploads once and submits all 4 variants in seconds.

## Calibration data

`build_calib.py` produces `calib_data.pkl` (~9 MB) containing:
- 16 diverse prompts tokenized to seq=32
- Real `per_layer_inputs` computed from the model's own `embed_tokens_per_layer` table (NOT random noise)

This is critical — random calibration data degrades W8A16 perplexity by 10-30%; real activations keep it under 5%.

## When upload completes

The phone runner (`src/runtime/gemma4_runner.cpp`) auto-detects total .bin size at startup and picks residency strategy:
- INT8 W8A16 (~3 GB across 2 parts): both parts kept resident, no swap → big speedup
- INT4 W4A16 (~1.4 GB single .bin): single context resident, no chain overhead → maximum speedup

Expected per-token latency: 50-200 ms (vs current 10+ s for FP16 swap-only).
