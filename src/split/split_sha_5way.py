#!/usr/bin/env python3
"""
Split the SHA gemma4 ONNX into 5 parts so each fits within Hexagon v69's 2 GB
single-context PD limit.

We re-use the boundary tensors of the working 4-way split, then carve part3
(decoders 28-34 + final RMSNorm) and part4 (lm_head only) out of the old
part3 (which was 2.13 GB and over-limit).

We use onnx.utils.extract_model and include EVERY upstream boundary tensor in
each downstream part's input list.  This caps the backward dependency walk so
nothing reaches back to the token embedding (768 MB) or earlier-layer
matmul weights.
"""

import json
import os
import subprocess
import sys

import onnx

SRC = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_sha_clean/gemma4_sha.onnx"
OUT_DIR = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_sha_split5"

# Per-bucket boundary outputs from the previous (working) 4-way split.
B0_OUT = [
    "/text_model/layers.9/Mul_1_output_0",
    "/text_model/Mul_1_output_0",
    "/text_model/layers.0/self_attn/Unsqueeze_output_0",
    "/text_model/layers.0/self_attn/Unsqueeze_1_output_0",
    "/text_model/layers.4/self_attn/Unsqueeze_output_0",
    "/text_model/layers.4/self_attn/Unsqueeze_1_output_0",
]
B1_OUT = [
    "/text_model/layers.19/Mul_1_output_0",
    "/text_model/layers.19/self_attn/Cast_output_0",
    "/text_model/layers.19/self_attn/Cast_1_output_0",
    "/text_model/layers.19/self_attn/Cast_9_output_0",
]
B2_OUT = [
    "/text_model/layers.27/Mul_1_output_0",
    "/text_model/layers.24/self_attn/Cast_output_0",
    "/text_model/layers.24/self_attn/Cast_1_output_0",
    "/text_model/layers.24/self_attn/Cast_9_output_0",
]
# New boundary for cutting lm_head off
B3_OUT = [
    "/text_model/norm/Cast_output_0",
]

SPLIT_PLAN = [
    dict(
        name="part0",
        inputs=["input_ids", "per_layer_inputs"],
        outputs=B0_OUT,
    ),
    dict(
        name="part1",
        inputs=B0_OUT,
        outputs=B1_OUT,
    ),
    dict(
        name="part2",
        # Include B0 (esp. /text_model/layers.9/Mul_1_output_0) so the walk
        # never goes past layer 9 to the embedding.
        inputs=B0_OUT + B1_OUT,
        outputs=B2_OUT,
    ),
    dict(
        name="part3",
        # decoders 28..34 + final RMSNorm, NO lm_head
        inputs=B0_OUT + B1_OUT + B2_OUT,
        outputs=B3_OUT,
    ),
    dict(
        name="part4",
        # lm_head only
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

# extract_model loads the full source ONNX (with external data) and writes a
# self-contained sub-ONNX with embedded weights at <dst>.
onnx_utils.extract_model(src, dst, spec["inputs"], spec["outputs"],
                         check_model=False)

print(f"[helper] re-saving with external data sidecar", flush=True)
m = onnx.load(dst, load_external_data=True)

# Strip any inputs that ended up being unused by the sub-graph (extract_model
# keeps every declared input even when it's not consumed by any kept node).
# Leaving dangling inputs causes AI Hub to reject the model.
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

# Strip value_info entries that overlap with graph IO.  ONNX's IR forbids a
# tensor name appearing in both value_info and graph.input/graph.output, but
# onnx.utils.extract_model leaves these duplicates in place and AI Hub then
# rejects the model with:
#   "Tensors {...} occur in value_info but also in model IO."
io_names = set()
for vi in list(m.graph.input) + list(m.graph.output):
    io_names.add(vi.name)
kept_vi = [vi for vi in m.graph.value_info if vi.name not in io_names]
removed_vi = [vi.name for vi in m.graph.value_info if vi.name in io_names]
if removed_vi:
    print(f"[helper] stripping value_info overlap with IO: {removed_vi}", flush=True)
del m.graph.value_info[:]
m.graph.value_info.extend(kept_vi)

# Save with external data sidecar named <basename>.data.
data_filename = os.path.basename(dst).replace(".onnx", ".data")
data_full = os.path.join(os.path.dirname(dst), data_filename)
for stale in (data_full, dst + ".data"):
    if os.path.exists(stale):
        os.remove(stale)
onnx.save_model(
    m,
    dst,
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

    # Clean stale outputs from previous run.
    for f in os.listdir(OUT_DIR):
        fp = os.path.join(OUT_DIR, f)
        if f.startswith("part") and (f.endswith(".onnx") or f.endswith(".data") or f.endswith(".onnx.data")):
            os.remove(fp)

    for idx, part in enumerate(SPLIT_PLAN):
        out_path = os.path.join(OUT_DIR, f"part{idx}.onnx")
        print(f"\n[+] extracting {part['name']}")
        rc = subprocess.call(
            [sys.executable, helper_path, SRC, out_path, json.dumps(part)],
            env={**os.environ, "TMPDIR": "/home/hardoker77/tmp_qaihub"},
        )
        if rc != 0:
            print(f"!! {part['name']} extraction failed")
            sys.exit(rc)

    print("\n[=] final sizes")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("part") and (f.endswith(".onnx") or f.endswith(".data")):
            sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024**2
            print(f"    {f}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
