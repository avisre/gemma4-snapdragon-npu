#!/usr/bin/env python3
"""Variant A: submit_quantize_job (W8A16 PTQ) then chain submit_compile_job.

Uses real Gemma-4 calibration data built by build_calib.py.
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

ZIP = "/home/hardoker77/gemma4_e2b_v69/aihub/gemma4_sha.onnx.zip"
CALIB = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/calib_data.pkl"
OUT = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/variant_A_jobs.json"

with open(CALIB, "rb") as f:
    calib_data = pickle.load(f)
print(f"Loaded calib: keys={list(calib_data.keys())} n={len(calib_data['input_ids'])}", flush=True)

device = [d for d in hub.get_devices() if "Galaxy S22 5G" in d.name and d.os == "13"][0]
print(f"DEVICE: {device.name} / OS {device.os}", flush=True)

results = {}

# A1: W8A16 - best balance of accuracy and size
print("\n--- A1: submit_quantize_job W8A16 ---", flush=True)
try:
    qjob = hub.submit_quantize_job(
        model=ZIP,
        calibration_data=calib_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="gemma4_sha_qjob_w8a16",
    )
    results["A1_w8a16_quantize"] = {
        "job_id": qjob.job_id, "url": qjob.url,
        "weights": "INT8", "activations": "INT16",
    }
    print(f"  QUANTIZE: {qjob.job_id} | {qjob.url}", flush=True)
except Exception as e:
    results["A1_w8a16_quantize"] = {"error": str(e)}
    print(f"  FAIL: {e}", flush=True)
    traceback.print_exc()

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}", flush=True)
print(json.dumps(results, indent=2), flush=True)
