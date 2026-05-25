# Gemma 4 E2B - On-Phone Deployment (Hexagon v69)

End-to-end harness for pushing `gemma4_e2b.pte` to a OnePlus 10 Pro and
validating it against the CPU reference.

## Prerequisites

1. **OnePlus 10 Pro must be plugged in.**
   Models NE2213 / NE2215 / NE2210, `ro.board.platform=taro`,
   Snapdragon 8 Gen 1, Hexagon v69. The script will hard-fail if the
   connected device is anything older (e.g. OnePlus 6 = Hexagon v65).
2. `adb devices` shows the phone in `device` state (accept the auth dialog).
3. Local artifacts:
   - `<project>/executorch/build-x86/gemma4_e2b/hybrid_llama_qnn.pte` -- the
     QNN-delegated ExecuTorch model. (Override with `PTE_PATH=...`.)
   - `<project>/checkpoints/gemma-4-e2b-it/tokenizer.json` -- HuggingFace
     tokenizer (SentencePiece BPE, vocab 262 144).
   - `<project>/aihub/gemma4_v69.ple` -- packed per-layer-embedding table
     (`{magic="PLE1", vocab=262144, num_layers=35, ple_dim=256, dtype=fp16}`).
   - `runtime/build/gemma4_runner` -- arm64-v8a binary built from
     `gemma4_runner.cpp` + `ple_preprocess.cpp`. The stock qnn_llama_runner
     from the Qwen3 zip is **not** sufficient (see "Why a custom runner"
     below).
4. The QNN runtime libs are pulled from
   `/home/hardoker77/v69_llm/qwen3_v69/qnn_llama_runner.zip`.

## Quick start

```bash
cd /home/hardoker77/gemma4_e2b_v69
bash runtime/deploy_gemma4.sh             # checks device, pushes everything, runs one prompt
bash runtime/validate_gemma4.sh           # re-runs + compares to CPU ref + tok/s gate
bash runtime/validate_gemma4.sh --deploy  # redeploy + validate in one step
```

Common overrides:

```bash
PTE_PATH=/tmp/my_other.pte bash runtime/deploy_gemma4.sh
DEVICE_SERIAL=<serial>     bash runtime/deploy_gemma4.sh
PROMPT="Why is the sky blue?" SEQ_LEN=256 bash runtime/validate_gemma4.sh
SKIP_RUN=1 bash runtime/deploy_gemma4.sh  # push artifacts only
```

## On-device layout

`/data/local/tmp/gemma4_v69/`
```
gemma4_runner               # custom binary (arm64-v8a)
gemma4_e2b.pte              # ExecuTorch QNN-delegated model
tokenizer.json              # Gemma 4 HF tokenizer (do NOT use Qwen's)
gemma4_v69.ple              # packed PLE table (mmap'd at runtime)
libQnnHtp.so                # from Qwen3 zip
libQnnSystem.so
libQnnHtpV69Stub.so
libQnnHtpV69Skel.so
libqnn_executorch_backend.so
```

## Why a custom runner (instead of stock qnn_llama_runner)

Inspecting the `qnn_llama_runner` binary from the Qwen3 zip
(`strings | grep ^k...`), the supported `--decoder_model_version` values are:

- `kLlama2`, `kLlama3`, `kQwen2_5`, `kQwen3`, `kGemma3`, `kPhi4`,
  `kSmollm3`, `smollm2_135m`

There is **no `kGemma4`**, and no symbols referencing `per_layer_inputs`,
`altup`, `laurel`, or any Gemma 4 architectural feature. Gemma 4 E2B uses
Per-Layer Embeddings (PLE) feeding 35 separate per-layer activation
tensors -- the model graph expects an `input_ids` tensor *and* a
`per_layer_inputs[35, B, S, 256]` tensor. The stock runner only knows how to
push `input_ids`. We therefore ship a custom `gemma4_runner` that:

1. Loads `gemma4_v69.ple` via `PLEPreprocessor::Load` (mmap, 0-copy).
2. On every prefill/decode step, calls
   `PLEPreprocessor::ComputePerLayerInputs(input_ids, batch, seq, out)` to
   fill the second input tensor.
3. Reuses the QNN `.so` libs from the Qwen3 zip unchanged.

The tokenizer is HuggingFace `tokenizer.json` (SentencePiece BPE, vocab
262 144) which the runner's `tokenizers::HFTokenizer` already handles -- the
same path Qwen3 uses, so no special tokenizer work is needed beyond pushing
the right file.

## Common errors

| Symptom                                              | Cause / fix |
|------------------------------------------------------|-------------|
| `Connected device is platform 'sdm845'`              | OnePlus 6 plugged in. Switch to OnePlus 10 Pro. |
| `no adb devices in 'device' state`                   | Phone locked / debugging dialog not accepted / cable broken. |
| `PTE not found at ...`                               | Re-run the ExecuTorch QNN export, or pass the right path. |
| `RUNNER not found at runtime/build/gemma4_runner`    | Cross-compile via `runtime/Makefile` (NDK r26, arm64-v8a). |
| `libQnnHtpV69Skel.so` load failure on device         | `ADSP_LIBRARY_PATH` not set; the script sets it but check `cd $PHONE_DIR` succeeded. |
| `unsupported llama version`                          | You ran the stock `qnn_llama_runner` against the gemma4 .pte. Use `gemma4_runner` instead. |
| Generated text is `the the the the...`               | Greedy decode collapsed (matches the current CPU reference, which has the same bug). Fix the export, not the runner. |
| `[ple] shape mismatch: got (X,Y,Z), expected ...`    | `.ple` blob was built for a different model variant. Re-pack with `ple_preprocess.py`. |
| `[ple] bad magic`                                    | `.ple` file is corrupt or not a PLE1 blob. |

## Logs

All runs append to `runtime/phone_logs/`:
- `deploy_<ts>.log` -- deploy script
- `run_<ts>.log` / `raw_<ts>.txt` -- on-phone output captured from `gemma4_runner`

The CPU reference lives at `runtime/ref_outputs/generated_text.txt` and is
generated by `runtime/cpu_reference.py`.
