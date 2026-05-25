# What broke and how we fixed it — every challenge in detail

Each section: **the symptom**, **why it happened**, **what we tried that didn't work**, **what finally worked**, and (where useful) **the exact file/line of the fix**.

If you're chasing a similar bug, the section names are the search terms.

---

## <a name="hetero-layers"></a>1. Heterogeneous layer dimensions

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict for MultiScopeAwareLlamaModel:
  size mismatch for layers.4.attention.wq.weight:
    checkpoint=[4096, 1536] model=[2048, 1536]
  size mismatch for layers.15.feed_forward.w1.weight:
    checkpoint=[12288, 1536] model=[6144, 1536]
```

**Why:** Gemma 4 E2B isn't uniform. Layers 4, 9, 14, 19, 24, 29, 34 use `head_dim=512`; layers 15-34 use `intermediate_size=12288`. ExecuTorch's `MultiScopeAwareLlamaModel` assumed every layer used `config.head_dim` and `config.intermediate_size`.

**Didn't work:** Trying to "downsample" the checkpoint to uniform sizes — corrupts the model.

**What worked:** Extended ExecuTorch wrappers to take per-layer arrays:
- `ModelArgs.global_head_dim_per_layer: list[int]` — 35-element array, value per layer
- `ModelArgs.intermediate_size_per_layer: list[int]` — same
- `LlamaAttention.__init__` reads `head_dim_per_layer[layer_idx]` when set
- `FeedForward.__init__(args, layer_idx=...)` reads `intermediate_size_per_layer[layer_idx]`
- `LlamaDecoderLayer` passes `layer_idx` through

Config additions (in `e2b_qcom_static_config.json`):
```json
{
  "global_head_dim_per_layer": [256,256,256,256,512, 256,256,256,256,512, ...35 values],
  "intermediate_size_per_layer": [6144,6144,...,12288,12288,...]
}
```

After this fix: `load_state_dict()` succeeded with 0 missing, 0 unexpected, 0 mismatches.

---

## <a name="ple"></a>2. Per-Layer Embeddings (PLE) — 4.7 GB table doesn't fit on NPU

**Symptom:** AI Hub compile errors out on any single tensor over ~2 GB. Gemma 4's `embed_tokens_per_layer` is `262144 × 35 × 256 × 2 bytes ≈ 4.7 GB`.

**Why:** Hexagon NPU 32-bit addressing limits single tensors to ~2 GB.

**Didn't work:** Quantizing PLE to int8 — still 2.3 GB, still over.

**What worked:** **Externalize PLE.** Don't include it in the NPU graph at all.

- HF formula: `per_layer_inputs[b,s,l,d] = ple_table[input_ids[b,s], l*256+d] * sqrt(256)`
- CPU does the lookup (instant — it's a table access) and feeds the result `(batch, seq, 35, 256)` as an input to the NPU graph
- See `src/runtime/ple_preprocess.py` for the Python converter (HF safetensors → binary `PLE1` format)
- See `src/runtime/ple_preprocess.cpp` for the on-device mmap'd lookup

Storage format (`PLE1`):
```
[24-byte header: magic="PLE1", vocab=262144, num_layers=35, ple_dim=256, dtype=fp16, scale_baked=1]
[262144 rows × 8960 fp16 = 4.5 GB]
```

The `*sqrt(256)` scale is baked into the stored values so the runtime is just `memcpy`. Peak CPU RAM during lookup: zero — `mmap` + direct copy.

---

## <a name="expand-bug"></a>3. The Grouped Query Attention Expand bug (the big one)

**Symptom:** Every ONNX variant we tried failed at AI Hub compile with:
```
validateQnnOpConfig: Failed QNN validation for /text_model/layers.0/self_attn/Expand
Validating Input[1] of ID 106.
Validating tensor 105 and 106 have the same Datatype.
Tensor 105 and 106 have mismatching datatypes. 0x232 != 0x216.
```

`0x232` = `QNN_DATATYPE_FLOAT_32`. `0x216` = `QNN_DATATYPE_FLOAT_16`.

**Why:** HF transformers' `repeat_kv` (used in any GQA model):
```python
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

That `.expand()` becomes an ONNX `Expand` op. Qualcomm's QAIRT compiler runs internal optimization passes that promote some tensor to FP32, then validates that both `Expand` inputs share dtype. They don't, because the compiler itself broke the invariant. The ONNX is clean — both inputs are FP16 — but QAIRT sees FP32 vs FP16 internally and bails.

**Didn't work (six attempts):**
1. **Pure FP16 conversion** (`onnxconverter_common.float16.convert_float_to_float16`) — same error
2. **Source-level patch** (replace `.expand()` with `.repeat()`) — `.repeat()` becomes `Expand+Tile` in dynamo export, still hit
3. **Constant-fold the dynamic Where→Equal→ConstantOfShape chain** feeding Expand's shape — same error
4. **Tile rewrite** (replace Expand with Tile post-hoc) — different error (no improvement)
5. **Mul-by-ones broadcast** (replace Expand with `data * ones_constant`) — different error
6. **Cast-everywhere** (wrap every Expand in/out with `Cast(fp16)`) — same error

The pattern: ONNX-level rewrites didn't help because QAIRT's internal passes were the problem.

**What worked:** **Don't use Expand at all.** Rewrite Grouped Query Attention as explicit per-head independent attention (true Single Head Attention / SHA):

```python
def sha_eager_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
    B, n_heads, S, head_dim = query.shape
    n_kv = key.shape[1]
    n_rep = n_heads // n_kv
    outputs = []
    for h in range(n_heads):
        kv_h = h // n_rep  # which K/V head this Q head shares
        q_h = query[:, h:h+1]
        k_h = key[:, kv_h:kv_h+1]
        v_h = value[:, kv_h:kv_h+1]
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :1]
        probs = torch.softmax(scores, dim=-1)
        out_h = torch.matmul(probs, v_h)
        outputs.append(out_h)
    return torch.cat(outputs, dim=1), None
```

When `torch.onnx.export` unrolls this loop, the graph has zero `Expand` ops in self-attention — just N independent `MatMul → Softmax → MatMul` chains, one per head. Bigger graph, slower export, but compiles cleanly. **This is the single most important fix in the project.**

File: `src/patches/mha2sha_attention.py`. Apply with:
```python
import source_patches.mha2sha_attention  # auto-patches via module side-effect
```

This is also exactly what ExecuTorch's Qualcomm backend does internally (`use_mha2sha=True`), but doing it at the PyTorch source level makes it work via AI Hub too.

After SHA: AI Hub QAIRT converter logged `INFO_CONVERSION_SUCCESS`. We had cleared the Expand bug.

---

## <a name="context-limit"></a>4. 5.4 GB graph won't fit in 2 GB v69 context

**Symptom:** Compile succeeded through QAIRT conversion but failed at the final binary generation step:
```
ERROR: graph requires estimated allocation of 5404364 KB, limit is 2097152 KB
ERROR: error during serialize: memory usage too large
```

**Why:** The model is 2.6B params × 2 bytes (FP16) = ~5 GB weights, plus activations. The Hexagon v69 single-context binary limit is 2 GB (uses a 32-bit address space). This is **documented** Qualcomm behavior, not a bug.

**What worked:** Split the model into 5 sub-models. Boundaries chosen so each fits under ~1.8 GB:

| Part | Layer range | Weight | Notes |
|---|---|---|---|
| 0 | token_embed + decoder layers 0-9 | 1.6 GB | Heavy embed table |
| 1 | layers 10-19 | 1.0 GB | Mostly normal |
| 2 | layers 20-27 | 1.3 GB | Wide-MLP layers (intermediate=12288) |
| 3 | layers 28-34 | 1.1 GB | Wide-MLP, no lm_head |
| 4 | lm_head only | 0.8 GB | The 262144-vocab output projection |

The lm_head MUST be its own part — at 262144 × 1536 × 2 = 805 MB, it dominates any part it lands in.

We first tried 4-way ({10, 20, 28, 35}). Parts 0-2 compiled and executed. Part 3 (containing lm_head + 7 wide-MLP layers) compiled but failed at runtime: needed 2.13 GB to instantiate, limit is 2.0 GB. Splitting lm_head into its own part (5-way) fixed it.

At runtime the parts chain: hidden_state flows from part N to part N+1. Last part outputs logits. See `src/runtime/gemma4_runner.cpp:RunChainOnce()`.

File: `src/split/split_sha_5way.py`.

---

## <a name="zip-format"></a>5. AI Hub zip rejects external weights

**Symptom:**
```
The uploaded ONNX model is missing its external weights.
Please use Qualcomm AI Hub Workbench's ONNX model directory format to upload this model.
```

**Why:** ONNX files >2 GB use an external data file (`.onnx` + `.onnx.data` sidecar). The ONNX file has an internal reference to the data filename. If you rename or restructure them inside the zip, the references break.

**What worked:** Zip in a specific directory structure that preserves names:
```bash
mkdir gemma4_sha5_part0.onnx                # ← directory matching the .onnx filename
cp part0.onnx gemma4_sha5_part0.onnx/      # original file inside
cp part0.data gemma4_sha5_part0.onnx/      # original sidecar inside
zip -r -0 gemma4_sha5_part0.onnx.zip gemma4_sha5_part0.onnx/
```

The `-0` is important — no compression, AI Hub can stream-read it.

---

## <a name="qnn-version"></a>6. QNN runtime version mismatch

**Symptom:** Phone-side `qnn-net-run` immediately fails on the `.bin`:
```
QnnDsp <E> Using newer context binary on old SDK
QnnDsp <E> Fail to get context blob with err 5000
Failed to create context from binary with err 0x1388
Could not create context from binary
```

**Why:** Our local QNN SDK was version 2.37 (community download). AI Hub compiles with 2.45. The .bin format diverged between versions. Older runtimes can't read newer binaries.

**Didn't work:** Searching for QNN SDK 2.45 download URLs — all 404. Qualcomm only ships the community version (2.37) publicly.

**What worked:** Microsoft publishes QNN runtime libs as a Maven artifact for Android:
```bash
wget https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.46.0/qnn-runtime-2.46.0.aar
unzip qnn-runtime-2.46.0.aar
# jni/arm64-v8a/libQnnHtpV69Stub.so, libQnnHtpV69Skel.so, etc.
adb push jni/arm64-v8a/libQnn*.so /data/local/tmp/gemma4_v69/
```

66 MB AAR, contains v68/v69/v73/v75/v79 stubs+skels for arm64-android. 2.46 reads 2.45 binaries fine.

---

## <a name="pybind11-bug"></a>7. ExecuTorch wheel pybind11 ABI bug

**Symptom:** When trying ExecuTorch's own QNN export path:
```
pybind11::cast_error: Unable to cast Python instance of type <class 'numpy.ndarray'> to C++ type '?'
```

Hit by every conv layer using per-channel quantization.

**Why:** `PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so` in the executorch 1.2.0 pip wheel has a broken `PYBIND11_NUMPY_DTYPE(Qnn_ScaleOffset_t, scale, offset)` registration. `np.empty(N, dtype=Qnn_ScaleOffset_t)` returns dtype `object` instead of the structured `(scale=float32, offset=int32)` dtype. C++ side's `py::array_t<Qnn_ScaleOffset_t>` cast then can't recognize the array.

**Didn't work:**
- Nightly executorch wheel (`1.4.0.dev20260524+cpu`) — same bug
- Different numpy versions — same bug

**What worked:** Two options. Workaround (Python-only): monkey-patch `node_visitor.py` to downgrade per-channel quant to per-tensor, sidestepping the broken cast:
```python
# Add to make_qnn_per_channel_config:
if os.environ.get("QNN_PYBIND_SCALE_OFFSET_WORKAROUND") == "1":
    avg_scale = float(sum(scales) / len(scales))
    return self.make_qnn_per_tensor_config({**attrs, "scale": avg_scale, "offset": 0})
```
~1-2% accuracy hit but works immediately. Set env var: `QNN_PYBIND_SCALE_OFFSET_WORKAROUND=1`.

Canonical fix: rebuild `PyQnnManagerAdaptor.so` from source against the local pybind11 v3.0.4 submodule. We didn't need this in the end (AI Hub cloud-compile path bypasses it), but the script is at `executorch/backends/qualcomm/scripts/build.sh --release`.

---

## <a name="rope-fix"></a>8. RoPE scratch tensors dead-code-eliminated (current open issue)

**Symptom:** The chain runs end-to-end on the NPU, produces real token IDs, but the decoded text is gibberish (`- [*\n\n\\`).

**Why:** When AI Hub compiles part 0, it sees that 4 specific `Unsqueeze` outputs are not consumed by any node IN PART 0 (they're consumed by parts 1-4, which it doesn't know about). The compiler does dead-code elimination and drops them. Parts 1-4 expect them as inputs but receive zeros instead. Zero RoPE = wrong position encoding everywhere = wrong attention pattern.

Phone log identifies the missing tensors:
```
[part 0->1] input '_text_model_layers_0_self_attn_Unsqueeze_output_0' (elems=8192) un-matched, zero
[part 0->1] input '_text_model_layers_4_self_attn_Unsqueeze_output_0' (elems=16384) un-matched, zero
```

**Two viable fixes** (the runtime-side one is in progress as of writing):

**(A) Inject RoPE host-side** (no AI Hub recompile, ~50 μs per inference):
```cpp
// In gemma4_runner.cpp, before RunChainOnce:
void BuildRopeTensors() {
    // Gemma RoPE: theta_i = base^(-2i/d)
    // Local layers (sliding): base=10000, head_dim=128
    // Global layers (full): base=1000000, head_dim=256
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim/2; i++) {
            float theta = pow(base, -2.0f * i / head_dim);
            rope_cos[pos][i] = cos(pos * theta);
            rope_sin[pos][i] = sin(pos * theta);
        }
    }
}
```
Bind these to the missing tensor slots in `RunPartPrefill` instead of zeros.

**(B) Re-export ONNX with explicit outputs** (no runtime change, but needs AI Hub recompile):
In `split_sha_5way.py`, when building each sub-model, scan the SHA model for any tensor in this part that's also referenced by a downstream part. Add those tensors as explicit `model.graph.output[]` entries even if part 0 itself doesn't use them as graph outputs. That prevents the compiler from eliminating them.

---

## What we learned (general lessons)

1. **The error message often lies.** "Tensor 105 and 106 have mismatching datatypes" said FP32 vs FP16 — both were FP16 in our model. The dtype mismatch was created by the compiler's own internal pass.

2. **Multi-step pipelines hide which step actually failed.** AI Hub does: ONNX validate → ONNX simplify → DLC convert → context binary generate. The `exit code 15` we got told us nothing — we had to download logs and grep through 25 KB of output to find the actual `[ERROR]` line buried under hundreds of `[INFO]` lines.

3. **NPU constraints are real and undocumented.** The 2 GB context limit, the 32-bit addressing, the dead-code-elimination across parts — none of this is in public Qualcomm docs. We learned by hitting walls.

4. **Other people have hit these.** ExecuTorch's `use_mha2sha=True`, Qualcomm's `OnnxSplitter` utility, the QNN runtime AAR on Maven — all exist because everyone trying to run an LLM on Hexagon NPU hits the same walls. Search before you reinvent.

5. **Parallel agent-driven debug beats single-threaded debugging by ~5×.** This project would have taken multiple weeks of single-threaded work. With Claude agents spawning sub-tasks in parallel (try fix A while researching B while polling C), it took ~1.5 days of wall-clock.
