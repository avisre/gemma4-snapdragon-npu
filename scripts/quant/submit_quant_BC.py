#!/usr/bin/env python3
"""Variants B & C: submit_compile_job with --quantize_full_type int8 / int4.

Reuses the existing 5 GB SHA ONNX zip - no re-export. AI Hub will apply
post-training quantization on the server side using its own default calibration.
"""
import os
import sys
import json
import traceback

os.environ['QAI_HUB_API_TOKEN'] = 'mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v'
os.environ['TMPDIR'] = '/home/hardoker77/tmp_qaihub'
os.makedirs('/home/hardoker77/tmp_qaihub', exist_ok=True)

import qai_hub as hub

ZIP = "/home/hardoker77/gemma4_e2b_v69/aihub/gemma4_sha.onnx.zip"

device = [d for d in hub.get_devices() if "Galaxy S22 5G" in d.name and d.os == "13"][0]
print(f"DEVICE: {device.name} / OS {device.os}", flush=True)

INPUT_SPECS = {
    "input_ids":        ((1, 32),         "int64"),
    "per_layer_inputs": ((1, 32, 35, 256), "float16"),
}

VARIANTS = [
    ("B_int8_full",
     "--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io "
     "--quantize_full_type int8 --quantize_io",
     "gemma4_sha_int8"),
    ("C_int4_weights",
     "--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io "
     "--quantize_full_type int4 --quantize_io",
     "gemma4_sha_int4"),
    ("D_w8a16",
     "--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io "
     "--quantize_full_type w8a16 --quantize_io",
     "gemma4_sha_w8a16"),
]

results = {}
for label, opts, name in VARIANTS:
    print(f"\n--- {label} ---", flush=True)
    print(f"  options: {opts}", flush=True)
    try:
        job = hub.submit_compile_job(
            model=ZIP, device=device,
            input_specs=INPUT_SPECS,
            options=opts,
            name=name,
        )
        results[label] = {"job_id": job.job_id, "url": job.url, "options": opts, "name": name}
        print(f"  JOB: {job.job_id} | {job.url}", flush=True)
    except Exception as e:
        results[label] = {"error": str(e), "options": opts, "name": name}
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()

out_path = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/quant_BCD_jobs.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}", flush=True)
print(json.dumps(results, indent=2), flush=True)
