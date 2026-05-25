#!/usr/bin/env python3
"""Hybrid prefill+decode export for Gemma 4 E2B.

Produces two ONNX graphs:
- prefill.onnx: input_ids [1, 128], per_layer_inputs [1, 128, 35, 256]
                outputs logits [1, 128, V] + initial K/V cache for the 15
                "OWN_KV" layers (L0..L14).
- decode.onnx:  input_ids [1, 1],  per_layer_inputs [1, 1, 35, 256]
                + past_k_l{i}, past_v_l{i} for i in OWN_KV_LAYERS
                outputs logits [1, 1, V] + new_k_l{i}, new_v_l{i}

The cache stored for each OWN_KV layer is the post-RoPE, post-norm K and V
tensors of shape (1, num_kv_heads=1, MAX_CACHE_LEN-PREFILL_LEN, head_dim).

Cache layout:
- prefill produces (1, 1, PREFILL_LEN, head_dim)  K/V tensors per OWN_KV layer
- decode consumes (1, 1, MAX_PAST=PREFILL_LEN+MAX_DECODE-1, head_dim) past K/V
  + produces (1, 1, MAX_PAST+1, head_dim) updated K/V    -- WAIT no:
  decode treats past as fixed-size (1, 1, MAX_PAST, head_dim) and outputs
  (1, 1, MAX_PAST, head_dim) where the new K/V is appended at the end and
  the oldest is pushed off the front (ring-buffer style).

To make ONNX shapes static the runner is expected to pre-allocate a
fixed-size scratch buffer of shape (1, 1, MAX_PAST, head_dim) and rotate it
itself; this script exports two graphs whose ONNX I/O are simply:
  past_k_l{i}: (1, 1, MAX_PAST, head_dim) [fp16]
  past_v_l{i}: (1, 1, MAX_PAST, head_dim) [fp16]
  new_k_l{i}:  (1, 1, MAX_PAST+1, head_dim) [fp16]  -- concat result
  new_v_l{i}:  (1, 1, MAX_PAST+1, head_dim) [fp16]
Runner side: drop new_k/v[..., :1, :] before feeding back as past.

Sharing: layers 15-34 reuse L13 (sliding) / L14 (full) KV, so we only need
to export 15 KV slots.
"""
from __future__ import annotations

import os
import sys
import time
import gc
import resource
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure we pick up the SHA monkey-patch.
sys.path.insert(0, "/home/hardoker77/gemma4_e2b_v69")


CKPT = Path("/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it")
OUT_DIR = Path("/home/hardoker77/gemma4_e2b_v69/exported_onnx_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFILL_LEN = 128
MAX_POSITION = 2048
MAX_PAST = MAX_POSITION - 1  # decode adds 1 new token per call


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


t0 = time.time()
def log(msg):
    print(f"[+{time.time()-t0:6.1f}s RSS={rss_gb():5.2f}GB] {msg}", flush=True)


# ============================================================================
# fp16 patches (same as export_seq32.py)
# ============================================================================
_orig_finfo = torch.finfo
_fp16_finfo = _orig_finfo(torch.float16)


class _Fp16ForcingFinfo:
    def __init__(self, dtype):
        if dtype in (torch.float32, torch.float64):
            self._inner = _fp16_finfo
        else:
            self._inner = _orig_finfo(dtype)

    def __getattr__(self, name):
        return getattr(self._inner, name)


torch.finfo = _Fp16ForcingFinfo
log("Patched torch.finfo -> fp16 limits")


from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from transformers.models.gemma4 import modeling_gemma4  # noqa: E402
from transformers import masking_utils  # noqa: E402


def _rmsnorm_forward_fp16(self, hidden_states: torch.Tensor) -> torch.Tensor:
    in_dtype = hidden_states.dtype
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    normed = hidden_states * torch.pow(mean_squared, -0.5)
    if self.with_scale:
        normed = normed * self.weight.to(in_dtype)
    return normed.to(in_dtype)


modeling_gemma4.Gemma4RMSNorm.forward = _rmsnorm_forward_fp16


def _rotary_forward_fp16(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
    target_dtype = x.dtype

    inv_freq_expanded = inv_freq[None, :, None].to(target_dtype).expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].to(target_dtype)

    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(target_dtype), sin.to(target_dtype)


modeling_gemma4.Gemma4TextRotaryEmbedding.forward = _rotary_forward_fp16


def _scaled_embed_forward(self, input_ids: torch.Tensor):
    out = nn.Embedding.forward(self, input_ids)
    return out * self.embed_scale.to(out.dtype)


modeling_gemma4.Gemma4TextScaledWordEmbedding.forward = _scaled_embed_forward

_orig_eager_mask = masking_utils.eager_mask


def _eager_mask_fp16(*args, **kwargs):
    kwargs["dtype"] = torch.float16
    return _orig_eager_mask(*args, **kwargs)


masking_utils.eager_mask = _eager_mask_fp16
try:
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = _eager_mask_fp16
except Exception:
    pass
try:
    _orig_sdpa_mask = masking_utils.sdpa_mask

    def _sdpa_mask_fp16(*args, **kwargs):
        m = _orig_sdpa_mask(*args, **kwargs)
        if isinstance(m, torch.Tensor) and m.dtype in (torch.float32, torch.float64):
            m = m.to(torch.float16)
        return m
    masking_utils.sdpa_mask = _sdpa_mask_fp16
except Exception:
    pass


def _no_packed_seq(position_ids):
    return None


masking_utils.find_packed_sequence_indices = _no_packed_seq


# ============================================================================
# SHA monkey-patch (must come before model load — same as sha export)
# ============================================================================
from source_patches import mha2sha_attention  # noqa: E402,F401
log("Installed SHA attention patch")


# ============================================================================
# Identify OWN_KV layers and shared-layer KV provenance
# ============================================================================
config = AutoConfig.from_pretrained(CKPT)
text_config = config.text_config
text_config.dtype = "float16"
text_config._attn_implementation = "eager"
N_LAYERS = text_config.num_hidden_layers
LAYER_TYPES = list(text_config.layer_types)
NUM_KV_SHARED = text_config.num_kv_shared_layers
FIRST_SHARED = N_LAYERS - NUM_KV_SHARED  # =15

# Compute is_kv_shared and store_full_length_kv per layer (mirror HF logic).
OWN_KV_LAYERS = []
STORE_FULL_FOR_TYPE = {}  # layer_type -> idx that stores full-length KV
PREV = LAYER_TYPES[:FIRST_SHARED]
for i in range(N_LAYERS):
    is_shared = (i >= FIRST_SHARED) and (FIRST_SHARED > 0)
    if not is_shared:
        OWN_KV_LAYERS.append(i)
        lt = LAYER_TYPES[i]
        try:
            rev_idx = list(reversed(PREV)).index(lt)
            last_idx = len(PREV) - 1 - rev_idx
            if i == last_idx:
                STORE_FULL_FOR_TYPE[lt] = i
        except ValueError:
            pass

# Each OWN_KV layer has its own head_dim depending on layer_type.
HEAD_DIM_FOR_LAYER = {}
for i in OWN_KV_LAYERS:
    lt = LAYER_TYPES[i]
    is_sliding = (lt == "sliding_attention")
    hd = text_config.head_dim if is_sliding else (
        text_config.global_head_dim if text_config.global_head_dim else text_config.head_dim
    )
    HEAD_DIM_FOR_LAYER[i] = hd

NUM_KV_HEADS = text_config.num_key_value_heads
log(f"OWN_KV layers: {OWN_KV_LAYERS}  STORE_FULL: {STORE_FULL_FOR_TYPE}")
log(f"head_dims: {HEAD_DIM_FOR_LAYER}")
log(f"num_kv_heads={NUM_KV_HEADS}")


# ============================================================================
# Patch Gemma4TextAttention.forward to use a global KV_IO dict.
#
# KV_IO[i] = ("past_k", "past_v") tensors before the layer call.
# After the call we set KV_IO_OUT[i] = (new_k, new_v).
# Shared layers read SHARED_KV_HOLDER[layer_type] which is populated by the
# store_full layer.
# ============================================================================
import collections
KV_INPUT = {}   # layer_idx -> (past_k, past_v); MAX_PAST length, or None for prefill
KV_OUTPUT = {}  # layer_idx -> (new_k, new_v); full-length for shared layers
SHARED_KV = {}  # layer_type -> (k, v) full length, populated mid-forward
MODE = {"value": "prefill"}  # "prefill" or "decode"


def patched_attn_forward(self, hidden_states, position_embeddings,
                         attention_mask, shared_kv_states,
                         past_key_values=None, **kwargs):
    """Drop-in for Gemma4TextAttention.forward.

    - Ignores past_key_values (HF Cache); uses module-level KV_INPUT/OUTPUT.
    - For OWN_KV layers (non-shared): compute K/V from this token batch, then
      concat with KV_INPUT[layer_idx] if in decode mode. Write final K/V into
      KV_OUTPUT[layer_idx]. If self.store_full_length_kv, also publish to
      SHARED_KV[layer_type].
    - For shared layers: read K/V from SHARED_KV[layer_type] (no concat needed
      because it already covers full length).
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = modeling_gemma4.apply_rotary_pos_emb(
        query_states, cos, sin, unsqueeze_dim=2
    )
    query_states = query_states.transpose(1, 2)  # (B, n_heads, S, D)

    if self.is_kv_shared_layer:
        key_states, value_states = SHARED_KV[self.layer_type]
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        key_states = self.k_norm(key_states)
        key_states = modeling_gemma4.apply_rotary_pos_emb(
            key_states, cos, sin, unsqueeze_dim=2
        )
        key_states = key_states.transpose(1, 2)  # (B, n_kv, S, D)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # In decode mode, concat with past then drop the oldest token so the
        # output shape == input past shape + 1 = MAX_PAST + 1.
        # Runner is responsible for slicing off the leading 1 before next call.
        if MODE["value"] == "decode":
            past_k, past_v = KV_INPUT[self.layer_idx]
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        KV_OUTPUT[self.layer_idx] = (key_states, value_states)

        if self.store_full_length_kv:
            SHARED_KV[self.layer_type] = (key_states, value_states)

    # Now run SHA attention with the resolved K/V.
    attention_interface = modeling_gemma4.eager_attention_forward
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


modeling_gemma4.Gemma4TextAttention.forward = patched_attn_forward
log("Patched Gemma4TextAttention.forward to use external KV I/O")


# ============================================================================
# Load model
# ============================================================================
log("Loading Gemma 4 E2B in fp16...")
full_model = AutoModelForCausalLM.from_pretrained(
    CKPT, dtype=torch.float16, attn_implementation="eager", low_cpu_mem_usage=True,
)
full_model.eval()
language_model = full_model.model.language_model
lm_head = full_model.lm_head
language_model.to(torch.float16)
lm_head.to(torch.float16)
for name, buf in language_model.named_buffers():
    if buf.dtype in (torch.float32, torch.float64):
        new_buf = buf.to(torch.float16)
        parts = name.split(".")
        mod = language_model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        mod._buffers[parts[-1]] = new_buf
log(f"Loaded ({sum(p.numel() for p in language_model.parameters())/1e6:.1f}M params)")


# ============================================================================
# Wrappers
# ============================================================================
class PrefillWrapper(nn.Module):
    def __init__(self, text_model, lm_head, final_softcap, own_kv_layers):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.final_softcap = final_softcap
        self.own_kv_layers = own_kv_layers

    def forward(self, input_ids, per_layer_inputs):
        MODE["value"] = "prefill"
        KV_OUTPUT.clear()
        SHARED_KV.clear()
        per_layer_inputs = per_layer_inputs.to(torch.float16)
        inputs_embeds = self.text_model.embed_tokens(input_ids).to(torch.float16)
        out = self.text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        if self.final_softcap is not None:
            cap = self.final_softcap
            logits = torch.tanh(logits / cap) * cap
        # Collect KV outputs in fixed order.
        # K and V for each own_kv layer in ascending layer order.
        kv_out_flat = []
        for i in self.own_kv_layers:
            k, v = KV_OUTPUT[i]
            kv_out_flat.append(k.to(torch.float16))
            kv_out_flat.append(v.to(torch.float16))
        return (logits.to(torch.float16), *kv_out_flat)


class DecodeWrapper(nn.Module):
    def __init__(self, text_model, lm_head, final_softcap, own_kv_layers,
                 head_dim_for_layer, max_past):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.final_softcap = final_softcap
        self.own_kv_layers = own_kv_layers
        self.head_dim_for_layer = head_dim_for_layer
        self.max_past = max_past

    def forward(self, input_ids, per_layer_inputs, *past_kv):
        # past_kv is flat: past_k_l0, past_v_l0, past_k_l1, past_v_l1, ...
        MODE["value"] = "decode"
        KV_INPUT.clear()
        KV_OUTPUT.clear()
        SHARED_KV.clear()
        for idx, layer_i in enumerate(self.own_kv_layers):
            pk = past_kv[2 * idx].to(torch.float16)
            pv = past_kv[2 * idx + 1].to(torch.float16)
            KV_INPUT[layer_i] = (pk, pv)
        per_layer_inputs = per_layer_inputs.to(torch.float16)
        inputs_embeds = self.text_model.embed_tokens(input_ids).to(torch.float16)
        # position_ids = past_len (we pass it explicitly to keep things simple)
        # The model auto-computes if not given: it uses inputs_embeds.shape[1]
        # which would always be 1 -> position 0. We need to feed the actual
        # position. Let's just override with position_ids based on MAX_PAST.
        position_ids = torch.tensor([[self.max_past]], dtype=torch.long)
        # Build attention masks (4D, all-zero = attend everything).
        # Decode attends single new token over MAX_PAST past + 1 = MAX_PAST+1.
        S_k = self.max_past + 1
        mask = torch.zeros((1, 1, 1, S_k), dtype=torch.float16)
        attn_mask_dict = {"full_attention": mask, "sliding_attention": mask}
        out = self.text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            attention_mask=attn_mask_dict,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        if self.final_softcap is not None:
            cap = self.final_softcap
            logits = torch.tanh(logits / cap) * cap
        kv_out_flat = []
        for i in self.own_kv_layers:
            k, v = KV_OUTPUT[i]
            kv_out_flat.append(k.to(torch.float16))
            kv_out_flat.append(v.to(torch.float16))
        return (logits.to(torch.float16), *kv_out_flat)


final_softcap = getattr(text_config, "final_logit_softcapping", None)


# ============================================================================
# Mask handling for decode: model auto-builds masks from past_key_values.
# Without a Cache the model thinks past=0, so it builds a (1,1,1,1) causal
# mask. We need it to attend over the full past. Patch attention_mask creator.
# ============================================================================

# The simplest path: don't let the model build masks. We pre-build an
# all-zeros (no masking) attention mask of correct shape and pass via
# attention_mask kwarg as a dict.
def make_decode_masks(num_past_plus_one):
    """All-zero 4D mask (1,1,1, S_k) -> attend everything past."""
    m = torch.zeros((1, 1, 1, num_past_plus_one), dtype=torch.float16)
    return {"full_attention": m, "sliding_attention": m}


# ============================================================================
# Export prefill
# ============================================================================
ple_dim = text_config.hidden_size_per_layer_input

ex_ids_prefill = torch.zeros((1, PREFILL_LEN), dtype=torch.int64)
ex_ple_prefill = torch.zeros((1, PREFILL_LEN, N_LAYERS, ple_dim), dtype=torch.float16)

prefill_wrap = PrefillWrapper(language_model, lm_head, final_softcap, OWN_KV_LAYERS)
prefill_wrap.eval()

log(f"Sanity prefill forward...")
with torch.inference_mode():
    prefill_out = prefill_wrap(ex_ids_prefill, ex_ple_prefill)
log(f"Prefill returns {len(prefill_out)} tensors: logits {tuple(prefill_out[0].shape)}, "
    f"first K {tuple(prefill_out[1].shape)}")

prefill_out_names = ["logits"]
for i in OWN_KV_LAYERS:
    prefill_out_names.append(f"k_l{i}")
    prefill_out_names.append(f"v_l{i}")

prefill_onnx = OUT_DIR / "prefill.onnx"
prefill_data = "prefill.onnx.data"
log(f"Exporting prefill to {prefill_onnx} ...")
with torch.inference_mode():
    torch.onnx.export(
        prefill_wrap,
        (ex_ids_prefill, ex_ple_prefill),
        str(prefill_onnx),
        input_names=["input_ids", "per_layer_inputs"],
        output_names=prefill_out_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
log("Prefill export raw done. Re-saving with external data...")
import onnx  # noqa: E402
m = onnx.load(str(prefill_onnx), load_external_data=True)
onnx.save_model(
    m, str(prefill_onnx),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=prefill_data,
    size_threshold=1024,
    convert_attribute=False,
)
del m
gc.collect()
log(f"Prefill saved ({prefill_onnx.stat().st_size/1024:.1f} KB graph + .data)")


# ============================================================================
# Export decode
# ============================================================================
ex_ids_decode = torch.zeros((1, 1), dtype=torch.int64)
ex_ple_decode = torch.zeros((1, 1, N_LAYERS, ple_dim), dtype=torch.float16)
ex_past_kv = []
for i in OWN_KV_LAYERS:
    hd = HEAD_DIM_FOR_LAYER[i]
    pk = torch.zeros((1, NUM_KV_HEADS, MAX_PAST, hd), dtype=torch.float16)
    pv = torch.zeros((1, NUM_KV_HEADS, MAX_PAST, hd), dtype=torch.float16)
    ex_past_kv.append(pk)
    ex_past_kv.append(pv)

decode_wrap = DecodeWrapper(language_model, lm_head, final_softcap,
                            OWN_KV_LAYERS, HEAD_DIM_FOR_LAYER, MAX_PAST)
decode_wrap.eval()

log(f"Sanity decode forward...")
with torch.inference_mode():
    decode_out = decode_wrap(ex_ids_decode, ex_ple_decode, *ex_past_kv)
log(f"Decode returns {len(decode_out)} tensors: logits {tuple(decode_out[0].shape)}, "
    f"first new_k {tuple(decode_out[1].shape)}")

decode_in_names = ["input_ids", "per_layer_inputs"]
decode_out_names = ["logits"]
for i in OWN_KV_LAYERS:
    decode_in_names.append(f"past_k_l{i}")
    decode_in_names.append(f"past_v_l{i}")
    decode_out_names.append(f"new_k_l{i}")
    decode_out_names.append(f"new_v_l{i}")

decode_onnx = OUT_DIR / "decode.onnx"
decode_data = "decode.onnx.data"
log(f"Exporting decode to {decode_onnx} ...")
with torch.inference_mode():
    torch.onnx.export(
        decode_wrap,
        (ex_ids_decode, ex_ple_decode, *ex_past_kv),
        str(decode_onnx),
        input_names=decode_in_names,
        output_names=decode_out_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
log("Decode export raw done. Re-saving with external data...")
m = onnx.load(str(decode_onnx), load_external_data=True)
onnx.save_model(
    m, str(decode_onnx),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=decode_data,
    size_threshold=1024,
    convert_attribute=False,
)
del m
gc.collect()
log(f"Decode saved ({decode_onnx.stat().st_size/1024:.1f} KB graph + .data)")


# ============================================================================
# Report IO
# ============================================================================
for label, path in [("PREFILL", prefill_onnx), ("DECODE", decode_onnx)]:
    print(f"\n=== {label}  {path}")
    m = onnx.load(str(path), load_external_data=False)
    print("Inputs:")
    for inp in m.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        et = inp.type.tensor_type.elem_type
        dt = {1: "fp32", 10: "fp16", 7: "int64"}.get(et, str(et))
        print(f"  {inp.name}: {dt} {dims}")
    print("Outputs:")
    for o in m.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
        et = o.type.tensor_type.elem_type
        dt = {1: "fp32", 10: "fp16", 7: "int64"}.get(et, str(et))
        print(f"  {o.name}: {dt} {dims}")

log("Hybrid export complete.")
