#!/usr/bin/env python3
"""Poll all submitted quant jobs and download .bin on success."""
import os
import sys
import json
import time

os.environ['QAI_HUB_API_TOKEN'] = 'mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v'
os.environ['TMPDIR'] = '/home/hardoker77/tmp_qaihub'

import qai_hub as hub

JOB_FILES = [
    "/home/hardoker77/gemma4_e2b_v69/aihub/quant/quant_BCD_jobs.json",
    "/home/hardoker77/gemma4_e2b_v69/aihub/quant/variant_A_jobs.json",
]
DOWNLOAD_DIR = "/home/hardoker77/gemma4_e2b_v69/aihub/quant/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def load_jobs():
    jobs = {}
    for f in JOB_FILES:
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
            for label, info in d.items():
                if "job_id" in info:
                    jobs[label] = info
    return jobs

POLL_INTERVAL = 90
MAX_MIN = 30
deadline = time.time() + MAX_MIN * 60
status = {}

while time.time() < deadline:
    jobs = load_jobs()
    if not jobs:
        print("No job json files yet.", flush=True)
        time.sleep(POLL_INTERVAL)
        continue

    all_terminal = True
    for label, info in jobs.items():
        if label in status and status[label].get("terminal"):
            continue
        try:
            job = hub.get_job(info["job_id"])
            jstatus = job.get_status()
            scode = jstatus.symbol if hasattr(jstatus, "symbol") else str(jstatus)
            terminal = jstatus.finished if hasattr(jstatus, "finished") else False
            status[label] = {
                "job_id": info["job_id"],
                "url": info["url"],
                "status": scode,
                "terminal": terminal,
                "success": jstatus.success if hasattr(jstatus, "success") else None,
            }
            print(f"[{time.strftime('%H:%M:%S')}] {label} -> {scode} (terminal={terminal})", flush=True)
            if not terminal:
                all_terminal = False
            elif jstatus.success and hasattr(job, "get_target_model"):
                try:
                    tm = job.get_target_model()
                    if tm is not None:
                        dst = os.path.join(DOWNLOAD_DIR, f"{label}.bin")
                        if not os.path.exists(dst):
                            print(f"  downloading -> {dst}", flush=True)
                            tm.download(dst)
                            sz = os.path.getsize(dst)
                            status[label]["bin_size"] = sz
                            status[label]["bin_path"] = dst
                            print(f"  -> {sz/1e6:.1f} MB", flush=True)
                except Exception as e:
                    print(f"  download err: {e}", flush=True)
        except Exception as e:
            print(f"  {label} err: {e}", flush=True)
            all_terminal = False

    with open("/home/hardoker77/gemma4_e2b_v69/aihub/quant/poll_status.json", "w") as f:
        json.dump(status, f, indent=2)

    if all_terminal:
        print("All jobs terminal. Done.", flush=True)
        break
    time.sleep(POLL_INTERVAL)

print("\n=== FINAL STATUS ===", flush=True)
print(json.dumps(status, indent=2), flush=True)
