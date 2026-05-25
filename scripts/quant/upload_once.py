#!/usr/bin/env python3
"""Upload the 5 GB SHA ONNX zip ONCE and persist its model_id for reuse."""
import os
import json

os.environ['QAI_HUB_API_TOKEN'] = 'mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v'
os.environ['TMPDIR'] = '/home/hardoker77/tmp_qaihub'
os.makedirs('/home/hardoker77/tmp_qaihub', exist_ok=True)

import qai_hub as hub

ZIP = "/home/hardoker77/gemma4_e2b_v69/aihub/gemma4_sha.onnx.zip"
OUT = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/uploaded_model.json"

print(f"Uploading {ZIP} ({os.path.getsize(ZIP)/1e9:.2f} GB) ONCE for reuse...", flush=True)
m = hub.upload_model(ZIP, name="gemma4_sha_5gb_for_quant")
print(f"MODEL_ID: {m.model_id}", flush=True)

with open(OUT, "w") as f:
    json.dump({"model_id": m.model_id, "name": "gemma4_sha_5gb_for_quant"}, f, indent=2)
print(f"Saved -> {OUT}", flush=True)
