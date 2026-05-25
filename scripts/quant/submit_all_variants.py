#!/usr/bin/env python3
"""Submit all quantization variants reusing the already-uploaded model.

Variants:
  B - compile_job --quantize_full_type int8 --quantize_io
  C - compile_job --quantize_full_type int4 --quantize_io
  D - compile_job --quantize_full_type w8a16 --quantize_io
  A - submit_quantize_job(W8A16) with real Gemma-4 calibration data
"""
import os
import sys
import json
import pickle
import traceback

os.environ['QAI_HUB_API_TOKEN'] = 'mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v'
os.environ['TMPDIR'] = '/home/hardoker77/tmp_qaihub'
os.makedirs('/home/hardoker77/tmp_qaihub', exist_ok=True)

import qai_hub as hub

MID_FILE = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/uploaded_model.json"
CALIB_FILE = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/calib_data.pkl"
OUT = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/all_variants_jobs.json"

with open(MID_FILE) as f:
    model_id = json.load(f)["model_id"]
print(f"Reusing uploaded model: {model_id}", flush=True)
model = hub.get_model(model_id)

device = [d for d in hub.get_devices() if "Galaxy S22 5G" in d.name and d.os == "13"][0]
print(f"DEVICE: {device.name} / OS {device.os}", flush=True)

INPUT_SPECS = {
    "input_ids":        ((1, 32),         "int64"),
    "per_layer_inputs": ((1, 32, 35, 256), "float16"),
}

results = {}

COMPILE_VARIANTS = [
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

for label, opts, name in COMPILE_VARIANTS:
    print(f"\n--- {label} ---", flush=True)
    print(f"  options: {opts}", flush=True)
    try:
        job = hub.submit_compile_job(
            model=model, device=device,
            input_specs=INPUT_SPECS, options=opts, name=name,
        )
        results[label] = {"kind": "compile", "job_id": job.job_id, "url": job.url,
                          "options": opts, "name": name}
        print(f"  JOB: {job.job_id} | {job.url}", flush=True)
    except Exception as e:
        results[label] = {"kind": "compile", "error": str(e), "options": opts, "name": name}
        print(f"  FAIL: {e}", flush=True)
        traceback.print_exc()
    # Persist incrementally
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

# Variant A: submit_quantize_job with real calibration data (PTQ path)
print(f"\n--- A_quantize_w8a16 ---", flush=True)
try:
    with open(CALIB_FILE, "rb") as f:
        calib_data = pickle.load(f)
    print(f"  calib loaded: keys={list(calib_data.keys())} n={len(calib_data['input_ids'])}", flush=True)
    qjob = hub.submit_quantize_job(
        model=model,
        calibration_data=calib_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="gemma4_sha_qjob_w8a16",
    )
    results["A_quantize_w8a16"] = {
        "kind": "quantize", "job_id": qjob.job_id, "url": qjob.url,
        "weights": "INT8", "activations": "INT16",
        "note": "After this succeeds, follow with submit_compile_job on the quantized model",
    }
    print(f"  QJOB: {qjob.job_id} | {qjob.url}", flush=True)
except Exception as e:
    results["A_quantize_w8a16"] = {"kind": "quantize", "error": str(e)}
    print(f"  FAIL: {e}", flush=True)
    traceback.print_exc()

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}", flush=True)
print(json.dumps(results, indent=2), flush=True)
