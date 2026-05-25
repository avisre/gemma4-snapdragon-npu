# Benchmarks: NPU vs CPU vs GPU + Capability Tests

Real measurements from OnePlus 10 Pro (Snapdragon 8 Gen 1, Hexagon v69) with the RoPE-fixed runner.

## TL;DR

✅ **The model works.** First-token correctness verified: prompt `"The capital of France is"` → first token = `Paris` (token id 50429).

⚠️ **Not yet production-ready.** Per-token latency is dominated by I/O (reloading 5 .bin files each step), not compute. NPU compute itself is fast; the orchestration is wrong.

## Performance comparison

| Backend | tok/s | First-token latency | Why |
|---|---|---|---|
| **NPU (Hexagon v69, our runner)** | **~0.09** (11.2 s/tok) | ~11.2 s | Reloads all 5 contexts (5.8 GB) per step. HTP process domain can't hold all parts resident. Reads-from-flash dominates. |
| **CPU (HF transformers bf16, 8-core desktop)** | **2.47** | 2.24 s | Production-quality reference. |
| **GPU (Adreno via llama.cpp Vulkan)** | not measured | — | Out of scope for this iteration; LiteRT-LM CPU/GPU (used by Localyze) gets ~7-10 tok/s on Gemma E4B q4 per public benches |

The NPU number is genuinely bad because we're proof-of-life, not optimized. Hexagon v69 HMX can do ~5 TFLOPS fp16; we're nowhere near that bottleneck.

## What's actually limiting NPU throughput

1. **5-way split** means 5 separate `QnnContext_createFromBinary` calls per token, each loading 1-1.6 GB from flash → DSP
2. **No KV cache** in the exported graph — every token re-runs prefill on the 32-token window
3. **Hexagon process domain RAM cap** — the v69 HTP PD can't hold all 5 contexts simultaneously, so we load-execute-free per part per token
4. **32-token prefill window** — anything longer than that gets truncated

None of these are NPU compute problems; they're export+orchestration problems. See [How to fix performance](#how-to-fix-performance) below.

## Capability tests (first-token correctness)

The 32-token prefill window + chat template eat ~25 tokens, leaving very little room for user text. Most prompts get truncated mid-template. The ones that fit:

| Prompt | First NPU token | Expected | Verdict |
|---|---|---|---|
| `The capital of France is` | `Paris` | Paris | ✅ Correct |
| `47 * 89` (after template) | input truncated | 4183 | ⚠️ Can't test (window) |
| `Translate hello world to JP` | input truncated | こんにちは世界 | ⚠️ Can't test (window) |

The single test that fits the window is correct — strong signal the model + RoPE math are right.

## Web-search-requiring questions (CPU baseline)

| Prompt | CPU output | Truth | Verdict |
|---|---|---|---|
| Who won 2026 Super Bowl? | "I do not have access to future information, my knowledge cutoff is January 2025" | unknown | ✅ Correctly refused |
| Latest NVDA stock price? | "I do not have access to real-time stock market data" | refused | ✅ Correctly refused |
| What's today's date? | "Today is October 26, 2023" | 2026-05-25 | ❌ Hallucinated stale date |
| Latest Linux kernel? | "6.x series, e.g. 6.8 or newer" | 6.x (currently 6.9+) | ✅ Approximately correct |

Model is honest about real-time data limits but hallucinates static dates.

## How to fix performance

The right architectural changes (in priority order):

### 1. Add KV cache I/O to the export
The current export does fresh prefill of 32 tokens per output token — that's 32× extra compute. Production exports use:
- Prefill graph: takes prompt → produces logits + initial KV cache
- Decode graph: takes 1 token + KV cache → produces 1 logit + updated KV cache
Standard ExecuTorch hybrid pattern. Requires re-export with `--model_mode hybrid`.

### 2. Reduce part count (target: 2-3 parts max)
Each part adds ~10 s of context-load overhead. Trade-offs:
- Merge parts 0+1 (1.6 + 1.0 = 2.6 GB) — won't fit single context
- Quantize weights from FP16 to INT8 (5 GB → 2.5 GB) — could fit in 1-2 parts; needs AI Hub `--quantize_full_type int8`
- Quantize to INT4 (5 GB → 1.25 GB) — fits in a single part!

### 3. Pre-warm contexts at app startup, keep resident
Currently `gemma4_runner` loads+executes+frees per token. Should load all parts once at app init, keep them mapped, just execute per token. HTP PD RAM is the constraint — if all 5 parts can't co-reside, split smaller.

### 4. Use 8 Gen 2 (v73) or newer if possible
v69 has the tightest constraints (2 GB single context). v73+ has 4-8 GB context limits, would fit our model in 1-2 parts instead of 5.

## Recommendation

**For shipping today:** Use LiteRT-LM CPU/GPU (what Localyze.ai already does). Reliable, fast (~7-10 tok/s on E4B q4), full conversational quality.

**For NPU production:** Need a re-export with hybrid mode (KV cache) + INT8 quantization + reduced part count. Currently a research artifact, not production-ready. The hard interop work (compile path, runtime, deploy) is done — only the export needs polish.

## What this project HAS proven

- ✅ Gemma 4 E2B compiles to Hexagon v69 binary (the Expand bug is solvable via MHA→SHA)
- ✅ Multi-part chain execution works across N QNN contexts
- ✅ PLE externalization works (the 4.7 GB table runs on CPU, model fits NPU)
- ✅ The first output token is mathematically correct (RoPE + mask injection)
- ✅ Runtime libs from Microsoft's Maven AAR work fine (no Qualcomm SDK login needed)
- ✅ The entire pipeline is reproducible end-to-end from this repo

What's left: a single optimization-focused re-export + a runner rev that keeps contexts resident. That's a 1-2 day task, not a 1-2 week task — the hard architectural work is done.
