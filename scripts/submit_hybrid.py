#!/usr/bin/env python3
"""Build AI Hub zips for all hybrid (prefill+decode) x 5 parts and submit."""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

os.environ.setdefault("QAI_HUB_API_TOKEN", "mm5u9vq5om8x3l2w6jpziyorlv0dhsrcrhaie14v")
os.environ.setdefault("TMPDIR", "/home/hardoker77/tmp_qaihub")

import onnx
import qai_hub as hub

SPLIT_DIR = "/home/hardoker77/gemma4_e2b_v69/exported_onnx_hybrid_split"
ZIP_DIR = "/home/hardoker77/gemma4_e2b_v69/aihub"
os.makedirs(ZIP_DIR, exist_ok=True)

DTYPE_MAP = {1: "float32", 7: "int64", 10: "float16"}


def build_zip(mode, idx):
    """Build {ZIP_DIR}/hybrid_{mode}_part{idx}.onnx.zip in AI Hub format."""
    onnx_src = os.path.join(SPLIT_DIR, f"{mode}_part{idx}.onnx")
    data_src = os.path.join(SPLIT_DIR, f"{mode}_part{idx}.data")
    if not os.path.exists(onnx_src) or not os.path.exists(data_src):
        raise FileNotFoundError(f"Missing {onnx_src} or {data_src}")

    # The .onnx file references its sidecar by basename ({mode}_part{idx}.data).
    # AI Hub expects {name}.onnx/model.onnx + {name}.onnx/model.data.
    # We need to rewrite the external_data location in the onnx graph to 'model.data'.
    # Easier: load the model with external data, re-save under new layout.
    work = os.path.join(ZIP_DIR, f"_tmp_hybrid_{mode}_part{idx}")
    if os.path.exists(work):
        shutil.rmtree(work)
    inner = os.path.join(work, f"hybrid_{mode}_part{idx}.onnx")
    os.makedirs(inner, exist_ok=True)

    # Stage by symlink: copy onnx file as 'model.onnx' with relocated external_data.
    m = onnx.load(onnx_src, load_external_data=True)
    onnx.save_model(
        m, os.path.join(inner, "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
        convert_attribute=False,
    )
    del m

    zip_path = os.path.join(ZIP_DIR, f"hybrid_{mode}_part{idx}.onnx.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"  zipping {inner} -> {zip_path}")
    # Use system zip to preserve large-file support; subprocess is faster than zipfile.
    rc = subprocess.call(
        ["zip", "-0", "-r", os.path.abspath(zip_path),
         os.path.basename(inner)],
        cwd=work,
    )
    if rc != 0:
        raise RuntimeError(f"zip failed rc={rc}")
    sz = os.path.getsize(zip_path) / 1024**2
    print(f"  zip size: {sz:.1f} MB")

    # Clean up staging
    shutil.rmtree(work)
    return zip_path


def main():
    device = [d for d in hub.get_devices() if "Galaxy S22 5G" in d.name and d.os == "13"][0]
    print(f"[+] device: {device.name} os={device.os}")

    ids_file = os.path.join(ZIP_DIR, "hybrid_job_ids.json")
    existing = {}
    if os.path.exists(ids_file):
        try:
            for rec in json.load(open(ids_file)):
                existing[f"{rec['mode']}_{rec['part']}"] = rec["job_id"]
        except Exception:
            pass

    jobs = []
    for mode in ["prefill", "decode"]:
        for i in range(5):
            key = f"{mode}_{i}"
            if key in existing:
                try:
                    j = hub.get_job(existing[key])
                    print(f"{key}: resumed {j.job_id} | {j.url}")
                    jobs.append((mode, i, j))
                    continue
                except Exception as e:
                    print(f"{key}: stale job id {existing[key]} ({e}); resubmitting")

            print(f"\n[+] building zip for {key}")
            zip_path = build_zip(mode, i)

            onnx_path = os.path.join(SPLIT_DIR, f"{mode}_part{i}.onnx")
            m = onnx.load(onnx_path, load_external_data=False)
            specs = {}
            for inp in m.graph.input:
                shape = tuple(d.dim_value if d.HasField("dim_value") else 1
                              for d in inp.type.tensor_type.shape.dim)
                dt = DTYPE_MAP.get(inp.type.tensor_type.elem_type, "float16")
                specs[inp.name] = (shape, dt)
            print(f"  input_specs ({len(specs)}): {list(specs.keys())[:6]}...")

            last_err = None
            for attempt in range(5):
                try:
                    j = hub.submit_compile_job(
                        model=zip_path,
                        device=device,
                        input_specs=specs,
                        options="--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io",
                        name=f"gemma4_hybrid_{key}",
                    )
                    print(f"{key}: {j.job_id} | {j.url}")
                    jobs.append((mode, i, j))
                    with open(ids_file, "w") as fh:
                        json.dump([{"mode": mm, "part": pp, "job_id": jj.job_id,
                                    "name": jj.name, "url": jj.url}
                                   for mm, pp, jj in jobs], fh, indent=2)
                    break
                except Exception as e:
                    last_err = e
                    backoff = 30 * (attempt + 1)
                    print(f"{key}: submit attempt {attempt+1} failed ({e!r}); sleeping {backoff}s")
                    time.sleep(backoff)
            else:
                print(f"{key}: GIVING UP; last error: {last_err!r}")

    print(f"\n[=] {len(jobs)}/10 jobs submitted; polling status")
    DONE = {"SUCCESS", "FAILED", "CANCELLED"}
    for attempt in range(40):  # up to ~60 minutes
        print(f"\n=== {time.strftime('%H:%M:%S')} (attempt {attempt+1}) ===", flush=True)
        all_done = True
        for mode, i, j in jobs:
            try:
                s = j.get_status()
                code = s.code
            except Exception as e:
                code = f"ERR:{e}"[:60]
            print(f"  {mode}_part{i} {j.job_id}: {code}", flush=True)
            if code not in DONE:
                all_done = False
        if all_done:
            break
        time.sleep(90)

    print("\n[=] downloading successful targets")
    for mode, i, j in jobs:
        try:
            if j.get_status().code == "SUCCESS":
                out = f"/home/hardoker77/gemma4_e2b_v69/aihub/gemma4_hybrid_{mode}_part{i}.bin"
                j.get_target_model().download(out)
                sz = os.path.getsize(out) / 1024**2
                print(f"DOWNLOADED {mode}_part{i}: {out}  {sz:.1f} MB")
            else:
                print(f"SKIP {mode}_part{i}: status={j.get_status().code}")
        except Exception as e:
            print(f"DOWNLOAD ERROR {mode}_part{i}: {e}")


if __name__ == "__main__":
    main()
