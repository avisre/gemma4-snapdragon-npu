"""Host-side inference loop scaffolding for Gemma 4 E2B on Hexagon v69.

Wiring (CPU side; NPU graph is mocked for now):

    prompt (str)
        |
        v
    GemmaTokenizer.encode  -->  input_ids (1, S<=128)  int32
        |
        +------------------------------+
        |                              |
        v                              v
    PLEPreprocessor.lookup        npu_forward(input_ids, per_layer_inputs)
    -> per_layer_inputs                |
       (1, S, 35, 256) fp16            v
                                  logits (1, S, 262144) fp16
                                       |
                                       v
                                  sampler.sample(logits[-1])
                                       |
                                       v
                                  next token id
                                       |
       (KV cache lives inside NPU graph; we just track prefix len)
                                       |
                                       v
                                  feed (1, 1) decode-step
                                       ...

For now `npu_forward` is a deterministic random stub so the loop can be
exercised end-to-end on the dev machine without the real model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sampler import sample
from tokenizer import GemmaTokenizer, EOS_ID

# Optional: PLE preprocessor (real binary may not exist yet on this machine)
try:
    from ple_preprocess import PLEPreprocessor, NUM_LAYERS, PLE_DIM
except ImportError:                            # pragma: no cover
    PLEPreprocessor = None
    NUM_LAYERS, PLE_DIM = 35, 256

# Graph I/O shapes (match the QNN/Hexagon export):
SEQ_LEN = 128
VOCAB = 262_144


# ======================================================================
# NPU mock
# ======================================================================

def npu_forward(
    input_ids: np.ndarray,           # (1, 128) int32
    per_layer_inputs: np.ndarray,    # (1, 128, 35, 256) fp16
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Stub for the Hexagon v69 graph. Returns logits (1, 128, 262144) fp16.

    Shapes are validated so the wiring catches dimension bugs even with
    the real graph absent.
    """
    if input_ids.shape != (1, SEQ_LEN):
        raise ValueError(f"input_ids shape {input_ids.shape} != (1, {SEQ_LEN})")
    if per_layer_inputs.shape != (1, SEQ_LEN, NUM_LAYERS, PLE_DIM):
        raise ValueError(
            f"per_layer_inputs shape {per_layer_inputs.shape} "
            f"!= (1, {SEQ_LEN}, {NUM_LAYERS}, {PLE_DIM})"
        )
    if rng is None:
        rng = np.random.default_rng(int(input_ids.sum()) & 0xFFFFFFFF)
    # cheap random surrogate; small magnitude -> softmax stays well-behaved
    return rng.standard_normal((1, SEQ_LEN, VOCAB), dtype=np.float32).astype(np.float16)


# ======================================================================
# KV-cache state
# ======================================================================

@dataclass
class KVCacheState:
    """Lightweight handle: the real K/V tensors live inside the NPU graph.

    We only track what the host needs:
      - prefix_len: number of tokens already consumed by the graph
      - eos_seen:   set after sampling EOS
    """
    prefix_len: int = 0
    eos_seen: bool = False

    def advance(self, n: int = 1) -> None:
        self.prefix_len += n


# ======================================================================
# Engine
# ======================================================================

class InferenceEngine:

    def __init__(
        self,
        checkpoint_dir: str = "/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it",
        ple_bin: Optional[str] = "/home/hardoker77/gemma4_e2b_v69/runtime/ple_weights.bin",
        seed: int = 0,
    ):
        self.tokenizer = GemmaTokenizer(checkpoint_dir)
        self.rng = np.random.default_rng(seed)

        # PLE is optional at scaffold time. If the binary isn't there, we
        # fall back to zeros so the loop still runs.
        self.ple: Optional[PLEPreprocessor] = None
        if PLEPreprocessor is not None and ple_bin and os.path.exists(ple_bin):
            self.ple = PLEPreprocessor(ple_bin)
            print(f"[engine] PLE loaded from {ple_bin}")
        else:
            print(f"[engine] PLE binary missing — using zero per_layer_inputs stub")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pad_to_window(self, ids: np.ndarray) -> np.ndarray:
        """Right-pad to (1, SEQ_LEN) with PAD=0."""
        n = ids.shape[-1]
        if n > SEQ_LEN:
            # for the skeleton we just keep the most recent window
            ids = ids[..., -SEQ_LEN:]
            n = SEQ_LEN
        out = np.zeros((1, SEQ_LEN), dtype=np.int32)
        out[0, :n] = ids.reshape(-1)
        return out

    def _per_layer(self, padded_ids: np.ndarray) -> np.ndarray:
        if self.ple is not None:
            return self.ple.lookup(padded_ids).astype(np.float16, copy=False)
        return np.zeros((1, SEQ_LEN, NUM_LAYERS, PLE_DIM), dtype=np.float16)

    def _forward(self, padded_ids: np.ndarray, last_pos: int) -> np.ndarray:
        """Run the NPU graph and return logits at `last_pos`."""
        per_layer = self._per_layer(padded_ids)
        logits = npu_forward(padded_ids, per_layer, rng=self.rng)
        return logits[0, last_pos, :]            # (vocab,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(self, prompt: str, kv: Optional[KVCacheState] = None
                ) -> tuple[np.ndarray, KVCacheState, list[int]]:
        """Tokenize, run one prefill pass, return (last-token logits, kv, ids)."""
        ids = self.tokenizer.encode(prompt, add_bos=True)
        if kv is None:
            kv = KVCacheState()
        padded = self._pad_to_window(np.asarray(ids, dtype=np.int32))
        last_pos = min(len(ids), SEQ_LEN) - 1
        logits = self._forward(padded, last_pos=last_pos)
        kv.advance(min(len(ids), SEQ_LEN))
        return logits, kv, ids

    def decode_step(
        self,
        prev_token: int,
        kv: KVCacheState,
        history_ids: list[int],
    ) -> np.ndarray:
        """One decode step: append prev_token, run graph, return next logits."""
        history_ids.append(int(prev_token))
        padded = self._pad_to_window(np.asarray(history_ids, dtype=np.int32))
        # the just-appended token sits at min(len-1, SEQ_LEN-1)
        last_pos = min(len(history_ids), SEQ_LEN) - 1
        logits = self._forward(padded, last_pos=last_pos)
        kv.advance(1)
        return logits

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_on_eos: bool = True,
    ) -> str:
        """Greedy/sampled decode loop. Returns the full output string."""
        logits, kv, ids = self.prefill(prompt)
        produced: list[int] = []
        for _ in range(max_tokens):
            tok = sample(
                logits, temperature=temperature,
                top_k=top_k, top_p=top_p, rng=self.rng,
            )
            if stop_on_eos and tok == EOS_ID:
                kv.eos_seen = True
                break
            produced.append(tok)
            logits = self.decode_step(tok, kv, ids)
        return self.tokenizer.decode(produced)


# ----------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------

def _selftest() -> None:
    engine = InferenceEngine(seed=123)
    out = engine.generate(
        "hello",
        max_tokens=8,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        stop_on_eos=False,
    )
    print(f"[engine] generate('hello', max_tokens=8) -> {out!r}")
    print("[engine] OK")


if __name__ == "__main__":
    _selftest()
