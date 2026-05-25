# Multi-Part Gemma-4 E2B Runtime Invocation

The Gemma-4 E2B decoder (35 layers) does not fit in a single QNN context
binary on Hexagon v69. We split it into **3 parts** (`kNumParts = 3`),
each owning a contiguous layer range:

| Part | Layers (abs)   | Layer count | Bin file                |
| ---- | -------------- | ----------- | ----------------------- |
| 0    | `[ 0 .. 11)`   | 12          | `qnn_context_part0.bin` |
| 1    | `[12 .. 23)`   | 12          | `qnn_context_part1.bin` |
| 2    | `[24 .. 34]`   | 11          | `qnn_context_part2.bin` |

Layer boundaries live in `kPartLayerEnd[]` in `gemma4_runner.h`; if you
re-export with a different split, update that constant.

## Forward pass

```
input_ids  ──► [token_embed inside part 0]
                              │
PLE.Lookup(input_ids) ──► per_layer_inputs    (sliced per part)
                              ▼
   part 0  : h_in = embed,    h_out → buf_a
   part 1  : h_in = buf_a,    h_out → buf_b
   part 2  : h_in = buf_b,    OUT = logits[B,S,vocab]
                              │
                              ▼
                         sample(logits)
```

Hidden state width between parts is `kHiddenDim = 1536` (fp16). Two
host scratch buffers `h_buf_a_` / `h_buf_b_` are ping-ponged so each
part's output cannot clobber its own input.

KV cache slices (sliding-window or global per `kLayerAttnType[]`) are
bound per-part: part `i` only reads / writes layers
`[PartFirstLayer(i), kPartLayerEnd[i])`. `kv_.total_tokens` is bumped
only after the **last** part commits so we don't advance position 3×
per token.

## On-device launch (OnePlus 10 Pro, a5523839)

```sh
adb -s a5523839 shell "cd /data/local/tmp/gemma4_v69 && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./gemma4_runner \
  --model_paths part0.bin,part1.bin,part2.bin \
  --ple_path gemma4_v69.ple \
  --tokenizer_path tokenizer.json \
  --prompt 'Hello' --max_tokens 64"
```

### Expected layout under `/data/local/tmp/gemma4_v69/`

```
gemma4_runner
libQnnHtp.so       libQnnSystem.so
libQnnHtpV69Stub.so  libQnnHtpV69Skel.so
part0.bin  part1.bin  part2.bin
gemma4_v69.ple
tokenizer.json
```

## CLI flags (gemma4_main.cpp)

| Flag                    | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| `--probe`               | dlopen QNN libs, list provider versions.                 |
| `--bin <path>`          | Validate a single .bin (legacy single-context path).     |
| `--model_paths a,b,c`   | Comma-separated list of part .bin files (must be 3).     |
| `--ple_path <path>`     | Packed PLE table.                                        |
| `--tokenizer_path <p>`  | SentencePiece tokenizer JSON.                            |
| `--prompt "..."`        | Prompt to generate from.                                 |
| `--max_tokens <N>`      | Decode cap (default 64).                                 |

## Status

- Skeleton compiles host-side (`/tmp/gemma4_runner.o`).
- Cross-compile for Android is gated on the AI Hub split agents
  landing actual `partN.bin` files so we can wire up the per-graph
  `IoTensors` (input_ids vs hidden_in, hidden_out vs logits, per-part
  KV slice tensor counts).
- Once the `.bin` files arrive: re-enable `gemma4_runner.cpp` in
  `Makefile.android`'s `SRCS`, populate the `TODO` blocks in
  `CreateContextFromBinary` and `RetrieveGraphHandlesForPart` from
  `binInfo->contextBinaryInfoV*.graphs[]`, then `make -f Makefile.android`.
