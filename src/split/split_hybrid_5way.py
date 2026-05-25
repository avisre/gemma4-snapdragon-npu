#!/usr/bin/env python3
"""Split the hybrid prefill/decode ONNX into 5 parts each that fit on v69.

For prefill:
  part0: input_ids, per_layer_inputs -> [boundary tensors after L9] + k_l0..v_l9
  part1: [boundary after L9] -> [boundary after L19] + k_l10..v_l14 + shared KV holders (L13/L14)
  part2: [boundary after L19] + shared KV holders -> [boundary after L27]
  part3: [boundary after L27] + shared KV holders -> [norm/Cast_output_0]
  part4: [norm/Cast_output_0] -> logits

For decode: identical splits, but
  part0 also takes past_k_l0..past_v_l9 inputs
  part1 also takes past_k_l10..past_v_l14 inputs
"""

import json
import os
import subprocess
import sys

import onnx

SRC_PREFILL = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_hybrid/prefill.onnx"
SRC_DECODE = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_hybrid/decode.onnx"
OUT_DIR = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_hybrid_split"

OWN_KV_LAYERS = list(range(15))  # L0..L14

# Boundary tensors for data flow.
# Prefill includes the L0/L4 Unsqueeze tensors (shared sliding/full cos/sin
# from the rotary at prefill_len=128). Decode does NOT have these tensors
# because position_ids is a scalar so cos/sin are folded into per-layer
# constants.
B0_OUT_PREFILL = [
    "/text_model/layers.9/Mul_1_output_0",
    "/text_model/Mul_1_output_0",
    "/text_model/layers.0/self_attn/Unsqueeze_output_0",
    "/text_model/layers.0/self_attn/Unsqueeze_1_output_0",
    "/text_model/layers.4/self_attn/Unsqueeze_output_0",
    "/text_model/layers.4/self_attn/Unsqueeze_1_output_0",
]
B0_OUT_DECODE = [
    "/text_model/layers.9/Mul_1_output_0",
    "/text_model/Mul_1_output_0",
]
B1_OUT = [
    "/text_model/layers.19/Mul_1_output_0",
]
B2_OUT = [
    "/text_model/layers.27/Mul_1_output_0",
]
B3_OUT = [
    "/text_model/norm/Cast_output_0",
]

# Shared-KV holder tensors differ between prefill and decode:
# - Prefill: shared layers consume the pre-transpose RoPE'd K (Add_1) and
#   transposed V (Transpose_2) from L13/L14, plus a small Cast_7 sliding
#   metadata tensor.
# - Decode: shared layers consume Concat_2 (past_k cat new_k) and Concat_3
#   (past_v cat new_v) directly from L13/L14, full length (MAX_PAST+1).
SHARED_KV_PREFILL = [
    "/text_model/layers.13/self_attn/Add_1_output_0",
    "/text_model/layers.13/self_attn/Transpose_2_output_0",
    "/text_model/layers.13/self_attn/Cast_7_output_0",
    "/text_model/layers.14/self_attn/Add_1_output_0",
    "/text_model/layers.14/self_attn/Transpose_2_output_0",
    "/text_model/layers.14/self_attn/Cast_7_output_0",
]
SHARED_KV_DECODE = [
    "/text_model/layers.13/self_attn/Concat_2_output_0",
    "/text_model/layers.13/self_attn/Concat_3_output_0",
    "/text_model/layers.14/self_attn/Concat_2_output_0",
    "/text_model/layers.14/self_attn/Concat_3_output_0",
]


def kv_outputs(layers):
    """Per-layer KV output graph names for the given list of layers."""
    out = []
    for i in layers:
        out.append(f"k_l{i}")
        out.append(f"v_l{i}")
    return out


def kv_inputs(layers):
    """Per-layer past KV input names (decode only) for the given layers."""
    out = []
    for i in layers:
        out.append(f"past_k_l{i}")
        out.append(f"past_v_l{i}")
    return out


def kv_outputs_new(layers):
    """Per-layer new KV output names (decode)."""
    out = []
    for i in layers:
        out.append(f"new_k_l{i}")
        out.append(f"new_v_l{i}")
    return out


def build_plan(mode):
    """Return the split plan for prefill or decode."""
    assert mode in ("prefill", "decode")

    if mode == "prefill":
        k_out = kv_outputs
        past_in_p0 = []
        past_in_p1 = []
        b0 = B0_OUT_PREFILL
        shared = SHARED_KV_PREFILL
    else:
        k_out = kv_outputs_new
        past_in_p0 = kv_inputs(list(range(0, 10)))
        past_in_p1 = kv_inputs(list(range(10, 15)))
        b0 = B0_OUT_DECODE
        shared = SHARED_KV_DECODE

    return [
        dict(
            name="part0",
            inputs=["input_ids", "per_layer_inputs"] + past_in_p0,
            outputs=b0 + k_out(list(range(0, 10))),
        ),
        dict(
            name="part1",
            inputs=b0 + past_in_p1,
            outputs=B1_OUT + k_out(list(range(10, 15))) + shared,
        ),
        dict(
            name="part2",
            inputs=b0 + B1_OUT + shared,
            outputs=B2_OUT,
        ),
        dict(
            name="part3",
            inputs=b0 + B1_OUT + B2_OUT + shared,
            outputs=B3_OUT,
        ),
        dict(
            name="part4",
            inputs=B3_OUT,
            outputs=["logits"],
        ),
    ]


HELPER_SRC = r'''
import json, os, sys
import onnx
from onnx import utils as onnx_utils

src, dst, spec_json = sys.argv[1], sys.argv[2], sys.argv[3]
spec = json.loads(spec_json)

print(f"[helper] extract_model: {src} -> {dst}", flush=True)
print(f"[helper] inputs : {spec['inputs']}", flush=True)
print(f"[helper] outputs: {spec['outputs']}", flush=True)

onnx_utils.extract_model(src, dst, spec["inputs"], spec["outputs"],
                         check_model=False)

print(f"[helper] re-saving with external data sidecar", flush=True)
m = onnx.load(dst, load_external_data=True)

# Drop unused declared inputs.
used = set()
for n in m.graph.node:
    for i in n.input:
        if i:
            used.add(i)
keep_inputs = [vi for vi in m.graph.input if vi.name in used]
dropped = [vi.name for vi in m.graph.input if vi.name not in used]
if dropped:
    print(f"[helper] dropping unused declared inputs: {dropped}", flush=True)
del m.graph.input[:]
m.graph.input.extend(keep_inputs)

# Strip value_info entries that overlap with graph IO.
io_names = set()
for vi in list(m.graph.input) + list(m.graph.output):
    io_names.add(vi.name)
kept_vi = [vi for vi in m.graph.value_info if vi.name not in io_names]
removed_vi = [vi.name for vi in m.graph.value_info if vi.name in io_names]
if removed_vi:
    print(f"[helper] stripping value_info overlap with IO ({len(removed_vi)})", flush=True)
del m.graph.value_info[:]
m.graph.value_info.extend(kept_vi)

data_filename = os.path.basename(dst).replace(".onnx", ".data")
data_full = os.path.join(os.path.dirname(dst), data_filename)
for stale in (data_full, dst + ".data"):
    if os.path.exists(stale):
        os.remove(stale)
onnx.save_model(
    m, dst,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data_filename,
    size_threshold=1024,
    convert_attribute=False,
)
sz_onnx = os.path.getsize(dst) / 1024**2
sz_data = os.path.getsize(data_full) / 1024**2
print(f"[helper] DONE: onnx={sz_onnx:.1f}MB data={sz_data:.1f}MB", flush=True)
'''


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    helper_path = os.path.join(OUT_DIR, "_extract_one.py")
    with open(helper_path, "w") as fh:
        fh.write(HELPER_SRC)

    # Clean stale outputs.
    for f in os.listdir(OUT_DIR):
        fp = os.path.join(OUT_DIR, f)
        if (f.startswith("prefill_part") or f.startswith("decode_part")) and (
            f.endswith(".onnx") or f.endswith(".data")
        ):
            os.remove(fp)
        elif f.startswith("onnx__") or f.startswith("_text_model_") or \
                f == "text_model.embed_tokens.weight":
            os.remove(fp)

    for mode, src in [("prefill", SRC_PREFILL), ("decode", SRC_DECODE)]:
        plan = build_plan(mode)
        for idx, part in enumerate(plan):
            out_path = os.path.join(OUT_DIR, f"{mode}_part{idx}.onnx")
            print(f"\n[+] extracting {mode} {part['name']}")
            rc = subprocess.call(
                [sys.executable, helper_path, src, out_path, json.dumps(part)],
                env={**os.environ, "TMPDIR": "/home/hardoker77/tmp_qaihub"},
            )
            if rc != 0:
                print(f"!! {mode} {part['name']} extraction failed")
                sys.exit(rc)
            # Cleanup stray torch-export artefacts.
            for f in os.listdir(OUT_DIR):
                if f.startswith("onnx__") or f.startswith("_text_model_") or \
                        f == "text_model.embed_tokens.weight":
                    os.remove(os.path.join(OUT_DIR, f))

    print("\n[=] final sizes")
    for f in sorted(os.listdir(OUT_DIR)):
        if (f.startswith("prefill_part") or f.startswith("decode_part")) and (
            f.endswith(".onnx") or f.endswith(".data")
        ):
            sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024**2
            print(f"    {f}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
