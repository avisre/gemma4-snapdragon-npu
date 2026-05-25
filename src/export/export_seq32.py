#!/usr/bin/env python3
"""Export Gemma 4 E2B text model to ONNX with seq_len=32 (instead of 128).

Hypothesis: at seq_len=128 the Expand op going [1,1,1,128,256] -> [1,1,8,128,256]
(KV-repeat for GQA) trips QNN Hexagon v69 compilation. Try a smaller prefill graph
to see if shape size is the trigger.

Strategy: reuse the strict-fp16 wrapper, change only SEQ_LEN. Output to
/home/hardoker77/gemma4_e2b_v69/exported_onnx_seq32/.
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


CKPT = Path("/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it")
OUT_DIR = Path("/home/hardoker77/gemma4_e2b_v69/exported_onnx_seq32")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 32  # prefill graph — smaller than the failing 128


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


t0 = time.time()
def log(msg):
    print(f"[+{time.time()-t0:6.1f}s RSS={rss_gb():5.2f}GB] {msg}", flush=True)


# ============================================================================
# (1) Monkey-patch torch.finfo BEFORE importing the model
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
log("Patched torch.finfo -> fp16 limits for fp32/fp64 queries")


from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from transformers.models.gemma4 import modeling_gemma4  # noqa: E402


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


def _eager_attention_fp16(module, query, key, value, attention_mask,
                          dropout=0.0, scaling=None, softcap=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = modeling_gemma4.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_gemma4.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.to(query.dtype)

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


modeling_gemma4.eager_attention_forward = _eager_attention_fp16
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["eager"] = _eager_attention_fp16
except Exception:
    pass


def _scaled_embed_forward(self, input_ids: torch.Tensor):
    out = nn.Embedding.forward(self, input_ids)
    return out * self.embed_scale.to(out.dtype)


modeling_gemma4.Gemma4TextScaledWordEmbedding.forward = _scaled_embed_forward


from transformers import masking_utils  # noqa: E402

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
log("All fp16 patches installed")


log("Loading Gemma 4 E2B model in fp16...")
config = AutoConfig.from_pretrained(CKPT)
text_config = config.text_config
text_config.dtype = "float16"
text_config._attn_implementation = "eager"

full_model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    dtype=torch.float16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
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
log(f"Loaded text model ({sum(p.numel() for p in language_model.parameters())/1e6:.1f}M params)")


class Seq32Wrapper(nn.Module):
    def __init__(self, text_model, lm_head, final_softcap):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.final_softcap = final_softcap

    def forward(self, input_ids: torch.Tensor, per_layer_inputs: torch.Tensor):
        per_layer_inputs = per_layer_inputs.to(torch.float16)
        inputs_embeds = self.text_model.embed_tokens(input_ids).to(torch.float16)
        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
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
        return logits.to(torch.float16)


final_softcap = getattr(text_config, "final_logit_softcapping", None)
wrapper = Seq32Wrapper(language_model, lm_head, final_softcap)
wrapper.eval()
log("Built Seq32Wrapper")


n_layers = text_config.num_hidden_layers
ple_dim = text_config.hidden_size_per_layer_input

example_input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.int64)
example_per_layer = torch.zeros((1, SEQ_LEN, n_layers, ple_dim), dtype=torch.float16)

log(f"Running sanity forward with SEQ_LEN={SEQ_LEN}...")
with torch.inference_mode():
    out = wrapper(example_input_ids, example_per_layer)
log(f"Sanity OK: logits dtype={out.dtype} shape={tuple(out.shape)}")
assert out.dtype == torch.float16, f"expected fp16, got {out.dtype}"


out_path = OUT_DIR / "gemma4_text_seq32.onnx"
out_data = "gemma4_text_seq32.onnx.data"

log(f"Exporting to {out_path} ...")
with torch.inference_mode():
    torch.onnx.export(
        wrapper,
        (example_input_ids, example_per_layer),
        str(out_path),
        input_names=["input_ids", "per_layer_inputs"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
log(f"Export done. Re-saving with external data...")

import onnx  # noqa: E402

model = onnx.load(str(out_path), load_external_data=True)
onnx.save_model(
    model,
    str(out_path),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=out_data,
    size_threshold=1024,
    convert_attribute=False,
)
log(f"Saved {out_path} ({out_path.stat().st_size/1024:.1f} KB graph + .data)")

m = onnx.load(str(out_path), load_external_data=False)
print("\nInputs:")
for inp in m.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    et = inp.type.tensor_type.elem_type
    dt = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32", 9: "bool"}.get(et, str(et))
    print(f"  {inp.name}: {dt} {dims}")

print("\nOutputs:")
for o in m.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
    et = o.type.tensor_type.elem_type
    dt = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32", 9: "bool"}.get(et, str(et))
    print(f"  {o.name}: {dt} {dims}")

log("Export complete")
