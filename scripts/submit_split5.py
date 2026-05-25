#!/usr/bin/env python3
"""Submit all 5 parts of the SHA gemma4 split to AI Hub for Hexagon v69."""
import json
import os
import sys
import time

os.environ.setdefault("QAI_HUB_API_TOKEN", "mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v")
os.environ.setdefault("TMPDIR", "/home/hardoker77/tmp_qaihub")

import onnx
import qai_hub as hub

SPLIT_DIR = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_sha_split5"
ZIP_DIR = "/home/hardoker77/gemma4_e2b_v69/aihub"

device = [d for d in hub.get_devices() if "Galaxy S22 5G" in d.name and d.os == "13"][0]
print(f"[+] device: {device.name} os={device.os}")

DTYPE_MAP = {1: "float32", 7: "int64", 10: "float16"}

ids_file = "/home/hardoker77/gemma4_e2b_v69/aihub/sha5_job_ids.json"
existing = {}
if os.path.exists(ids_file):
    try:
        for rec in json.load(open(ids_file)):
            existing[rec["part"]] = rec["job_id"]
    except Exception:
        pass

jobs = []
for i in range(5):
    if i in existing:
        try:
            j = hub.get_job(existing[i])
            print(f"PART {i}: resumed {j.job_id} | {j.url}")
            jobs.append((i, j))
            continue
        except Exception as e:
            print(f"PART {i}: stale job id {existing[i]} ({e}); resubmitting")

    onnx_path = os.path.join(SPLIT_DIR, f"part{i}.onnx")
    m = onnx.load(onnx_path, load_external_data=False)
    specs = {}
    for inp in m.graph.input:
        shape = tuple(d.dim_value if d.HasField("dim_value") else 1
                      for d in inp.type.tensor_type.shape.dim)
        dt = DTYPE_MAP.get(inp.type.tensor_type.elem_type, "float16")
        specs[inp.name] = (shape, dt)
    print(f"[+] part{i} input_specs: {specs}")

    zip_path = os.path.join(ZIP_DIR, f"sha5_part{i}.onnx.zip")
    last_err = None
    for attempt in range(5):
        try:
            j = hub.submit_compile_job(
                model=zip_path,
                device=device,
                input_specs=specs,
                options="--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io",
                name=f"gemma4_sha5_part{i}",
            )
            print(f"PART {i}: {j.job_id} | {j.url}")
            jobs.append((i, j))
            # Save incrementally so a later failure doesn't lose this id.
            with open(ids_file, "w") as fh:
                json.dump([{"part": pp, "job_id": jj.job_id, "name": jj.name, "url": jj.url} for pp, jj in jobs], fh, indent=2)
            break
        except Exception as e:
            last_err = e
            backoff = 30 * (attempt + 1)
            print(f"PART {i}: submit attempt {attempt+1} failed ({e!r}); sleeping {backoff}s")
            time.sleep(backoff)
    else:
        print(f"PART {i}: GIVING UP after retries; last error: {last_err!r}")

print(f"\n[=] {len(jobs)}/5 jobs submitted; polling status")
DONE = {"SUCCESS", "FAILED", "CANCELLED"}
last_codes = {}
for attempt in range(40):  # up to 60 minutes
    print(f"\n=== {time.strftime('%H:%M:%S')} (attempt {attempt+1}) ===", flush=True)
    all_done = True
    for i, j in jobs:
        try:
            s = j.get_status()
            code = s.code
        except Exception as e:
            code = f"ERR:{e}"[:60]
        print(f"  part{i} {j.job_id}: {code}  {j.name}", flush=True)
        last_codes[i] = code
        if code not in DONE:
            all_done = False
    if all_done:
        break
    time.sleep(90)

print("\n[=] downloading successful targets")
for i, j in jobs:
    try:
        if j.get_status().code == "SUCCESS":
            out = f"/home/hardoker77/gemma4_e2b_v69/aihub/gemma4_sha5_part{i}.bin"
            j.get_target_model().download(out)
            sz = os.path.getsize(out) / 1024**2
            print(f"DOWNLOADED part{i}: {out}  {sz:.1f} MB")
        else:
            print(f"SKIP part{i}: status={j.get_status().code}")
    except Exception as e:
        print(f"DOWNLOAD ERROR part{i}: {e}")
