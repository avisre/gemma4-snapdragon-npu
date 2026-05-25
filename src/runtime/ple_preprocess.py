"""Per-Layer Embeddings (PLE) preprocessor for Gemma 4 E2B on Hexagon v69.

The PLE table (model.language_model.embed_tokens_per_layer.weight) is 4.7 GB
(262144 x 8960 BF16). The Hexagon v69 NPU graph cannot hold it, so we
externalize the lookup: CPU computes `per_layer_inputs` for each batch BEFORE
invoking the NPU graph, and the NPU consumes it as an input.

Formula (verified from transformers/models/gemma4/modular_gemma4.py
Gemma4TextModel.get_per_layer_inputs):

    raw  = embed_tokens_per_layer.weight[input_ids]          # (B, S, 8960)
    raw *= sqrt(hidden_size_per_layer_input)                 # = sqrt(256) = 16
    per_layer_inputs = raw.reshape(B, S, num_layers, dim)    # (B, S, 35, 256)

The scale is baked into the serialized binary at conversion time, so the
runtime path on CPU/Android is a pure gather + reshape.

Binary file layout (little-endian):

    offset  size  field
    0       4     magic            b"PLE1"
    4       4     vocab_size       uint32   (262144)
    8       4     num_layers       uint32   (35)
    12      4     ple_dim          uint32   (256)
    16      4     dtype            uint32   (1 = fp16, 2 = bf16, 3 = fp32)
    20      4     embed_scale_baked uint32  (1 = scale already applied)
    24      ...   weights          vocab * num_layers * ple_dim * sizeof(dtype)
                                   row-major layout (vocab, num_layers*ple_dim)

Usage:
    # one-time conversion from HF checkpoint
    python ple_preprocess.py convert \
        --hf /home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it \
        --out /home/hardoker77/gemma4_e2b_v69/runtime/ple_weights.bin

    # runtime use (Python reference)
    pre = PLEPreprocessor("ple_weights.bin")
    per_layer_inputs = pre.lookup(input_ids)   # (B, S, 35, 256) fp16
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

# --- format constants (must match ple_preprocess.h) ---
MAGIC = b"PLE1"
HEADER_FMT = "<4sIIIII"   # magic, vocab, num_layers, dim, dtype, scale_baked
HEADER_SIZE = struct.calcsize(HEADER_FMT)

DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_FP32 = 3
_DTYPE_TO_NP = {DTYPE_FP16: np.float16, DTYPE_FP32: np.float32}

# --- model-specific constants for Gemma 4 E2B (from config.json) ---
VOCAB_SIZE = 262144
NUM_LAYERS = 35
PLE_DIM = 256
HF_TENSOR_NAME = "model.language_model.embed_tokens_per_layer.weight"
EMBED_SCALE = float(np.sqrt(PLE_DIM))   # 16.0


def _bf16_bytes_to_fp32(raw: bytes) -> np.ndarray:
    """Upcast raw bf16 bytes to a contiguous fp32 ndarray."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


# ============================================================
# Conversion (run on dev machine, not on device)
# ============================================================

def convert_from_hf(hf_dir: str, out_path: str, out_dtype: int = DTYPE_FP16) -> None:
    """Read PLE tensor from HF safetensors and write packed binary."""
    from safetensors import safe_open  # local import; only needed for conversion

    st_path = os.path.join(hf_dir, "model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(st_path)

    print(f"[ple] opening {st_path}", flush=True)
    # Use torch backend because safetensors numpy backend cannot decode BF16.
    import torch  # noqa: F401  (only needed for conversion)
    with safe_open(st_path, framework="pt") as f:
        if HF_TENSOR_NAME not in f.keys():
            raise KeyError(f"{HF_TENSOR_NAME} not in checkpoint")
        sl = f.get_slice(HF_TENSOR_NAME)
        shape = tuple(sl.get_shape())
        dtype = sl.get_dtype()
        print(f"[ple] tensor {HF_TENSOR_NAME}: shape={shape} dtype={dtype}")
        assert shape == (VOCAB_SIZE, NUM_LAYERS * PLE_DIM), shape

        # Stream-load (full tensor is ~4.7 GB BF16; fp32 would be ~9.4 GB).
        # Get the torch tensor (still bf16) and reinterpret as int16 (same bit
        # width, numpy-compatible). We keep `u16_np` as a *view* of the torch
        # storage — no extra ~4.7 GB copy. Chunk-slice it below for upcast.
        raw_t = f.get_tensor(HF_TENSOR_NAME).contiguous()  # torch.Tensor bf16
        u16_np = raw_t.view(torch.int16).numpy()  # (V, L*D) int16 view of bf16
        u16_np = u16_np.view(np.uint16)
        assert u16_np.shape == (VOCAB_SIZE, NUM_LAYERS * PLE_DIM), u16_np.shape

    print("[ple] upcasting BF16 -> FP32 (in chunks)", flush=True)
    # chunk over vocab to keep peak RAM manageable (~9.4 GB total fp32; do 16k rows at a time)
    chunk_rows = 16384
    np_out_dtype = _DTYPE_TO_NP[out_dtype]
    out_arr = np.empty((VOCAB_SIZE, NUM_LAYERS * PLE_DIM), dtype=np_out_dtype)
    for start in range(0, VOCAB_SIZE, chunk_rows):
        end = min(start + chunk_rows, VOCAB_SIZE)
        # u16_np is a view of the torch bf16 storage; slice as bytes then upcast
        slice_bytes = u16_np[start:end].tobytes()
        chunk_f32 = _bf16_bytes_to_fp32(slice_bytes).reshape(end - start, -1)
        chunk_f32 *= EMBED_SCALE  # bake the sqrt(256) scale in once
        out_arr[start:end] = chunk_f32.astype(np_out_dtype, copy=False)
        if start % (chunk_rows * 4) == 0:
            print(f"[ple]   converted rows {start}/{VOCAB_SIZE}", flush=True)

    # Free the torch tensor / source view before writing the output
    del u16_np, raw_t
    _write_binary(out_path, out_arr, out_dtype)


def _write_binary(out_path: str, arr: np.ndarray, dtype_id: int) -> None:
    np_dtype = _DTYPE_TO_NP[dtype_id]
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    assert arr.shape == (VOCAB_SIZE, NUM_LAYERS * PLE_DIM)
    header = struct.pack(
        HEADER_FMT, MAGIC, VOCAB_SIZE, NUM_LAYERS, PLE_DIM, dtype_id, 1
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        arr.tofile(f)
    size_mb = (HEADER_SIZE + arr.nbytes) / (1024 * 1024)
    print(f"[ple] wrote {out_path} ({size_mb:.1f} MB, dtype={np_dtype})")


# ============================================================
# Runtime (mirror of C++ path)
# ============================================================

class PLEPreprocessor:
    """Memory-maps the packed PLE binary and serves lookup() at runtime."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            hdr = f.read(HEADER_SIZE)
        magic, vocab, num_layers, dim, dtype_id, scale_baked = struct.unpack(HEADER_FMT, hdr)
        if magic != MAGIC:
            raise ValueError(f"bad magic in {path}: {magic!r}")
        if dtype_id not in _DTYPE_TO_NP:
            raise ValueError(f"unsupported dtype id {dtype_id}")
        self.vocab_size = vocab
        self.num_layers = num_layers
        self.ple_dim = dim
        self.np_dtype = _DTYPE_TO_NP[dtype_id]
        self.scale_baked = bool(scale_baked)

        # mmap whole file then carve out the weights view
        self._mmap = np.memmap(
            path,
            dtype=np.uint8,
            mode="r",
            offset=0,
        )
        weights_bytes = self._mmap[HEADER_SIZE:]
        self.weights = np.frombuffer(
            weights_bytes, dtype=self.np_dtype
        ).reshape(vocab, num_layers * dim)

    def lookup(self, input_ids: np.ndarray) -> np.ndarray:
        """Gather + reshape. Returns (B, S, num_layers, ple_dim)."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got {input_ids.shape}")
        if input_ids.dtype not in (np.int32, np.int64, np.uint32):
            input_ids = input_ids.astype(np.int32)
        b, s = input_ids.shape
        gathered = self.weights[input_ids]                       # (B, S, 8960)
        out = gathered.reshape(b, s, self.num_layers, self.ple_dim)
        if not self.scale_baked:
            out = out * np.float16(np.sqrt(self.ple_dim))
        return out


# ============================================================
# CLI / self-test
# ============================================================

def _cmd_convert(args):
    convert_from_hf(args.hf, args.out, out_dtype=DTYPE_FP16)


def _cmd_test(args):
    """Verify lookup matches HF reference for a small batch."""
    pre = PLEPreprocessor(args.bin)
    # batch=1, seq_len=128 to match the NPU graph
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, pre.vocab_size, size=(1, 128), dtype=np.int32)
    out = pre.lookup(input_ids)
    print(f"[ple] lookup ok: input_ids={input_ids.shape} -> out={out.shape} dtype={out.dtype}")
    assert out.shape == (1, 128, NUM_LAYERS, PLE_DIM), out.shape

    if args.compare_hf:
        from safetensors import safe_open
        with safe_open(os.path.join(args.compare_hf, "model.safetensors"), framework="numpy") as f:
            raw = f.get_tensor(HF_TENSOR_NAME)
        # bf16 raw bytes
        if str(raw.dtype) == "bfloat16" or raw.dtype == np.dtype("V2"):
            buf = raw.tobytes()
            ref_f32 = _bf16_bytes_to_fp32(buf).reshape(VOCAB_SIZE, NUM_LAYERS * PLE_DIM)
        else:
            ref_f32 = raw.astype(np.float32)
        ref = (ref_f32[input_ids] * EMBED_SCALE).reshape(1, 128, NUM_LAYERS, PLE_DIM)
        diff = np.abs(out.astype(np.float32) - ref).max()
        rel = diff / (np.abs(ref).max() + 1e-6)
        print(f"[ple] vs HF reference: max_abs_diff={diff:.4g}  rel={rel:.4g}")
        # fp16 round-off only
        assert rel < 1e-2, f"PLE mismatch vs HF reference (rel={rel})"
        print("[ple] PASS")


def main(argv=None):
    p = argparse.ArgumentParser(description="Gemma4 E2B PLE preprocessor")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert HF safetensors -> packed binary")
    c.add_argument("--hf", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_convert)

    t = sub.add_parser("test", help="self-test the packed binary")
    t.add_argument("--bin", required=True)
    t.add_argument("--compare-hf", default=None,
                   help="optional HF checkpoint dir to validate against")
    t.set_defaults(func=_cmd_test)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
