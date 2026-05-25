"""Host-side logit sampler for Gemma 4 E2B (NumPy, CPU only).

The NPU graph emits logits in fp16; this module accepts fp16/fp32 and
runs temperature + top-k + top-p filtering followed by multinomial draw.

The on-device fast path lives in sampler.h/.cpp (ARMv8.2 fp16 NEON);
this file is the reference / dev-machine implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _to_float32(logits: np.ndarray) -> np.ndarray:
    """Accept fp16/fp32/bf16-as-uint16; return contiguous fp32 1D vector."""
    arr = np.asarray(logits)
    # NPU graph shape is (1, 128, vocab) for prefill or (1, 1, vocab) for decode.
    # Caller is expected to slice to the last token already, but be liberal:
    if arr.ndim == 3:
        arr = arr[0, -1, :]
    elif arr.ndim == 2:
        arr = arr[-1, :]
    elif arr.ndim != 1:
        raise ValueError(f"unexpected logits shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out everything except the top-k by setting to -inf."""
    if k <= 0 or k >= logits.shape[0]:
        return logits
    # argpartition is O(n); the last k entries are the largest (unordered).
    idx = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, -np.inf)
    mask[idx] = logits[idx]
    return mask


def _apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep smallest set of tokens whose cumulative prob >= p."""
    if p >= 1.0 or p <= 0.0:
        return logits
    # softmax once for the cumulative test
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    order = np.argsort(-probs)         # descending
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    # keep up to and including the first index that crosses p
    cutoff = int(np.searchsorted(cum, p)) + 1
    cutoff = max(cutoff, 1)
    keep = order[:cutoff]
    mask = np.full_like(logits, -np.inf)
    mask[keep] = logits[keep]
    return mask


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        # degenerate (all -inf etc); fall back to uniform over finite entries
        finite = np.isfinite(x)
        out = np.zeros_like(x)
        if finite.any():
            out[finite] = 1.0 / finite.sum()
        return out
    return e / s


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample one token id from `logits`.

    Args:
        logits: shape (vocab,) or (1, vocab) or (1, S, vocab) — last token used.
        temperature: <=0 means greedy argmax.
        top_k: 0 disables. Applied before top_p.
        top_p: 1.0 disables. Applied after top_k.
        rng: optional np.random.Generator for determinism.
    """
    x = _to_float32(logits)

    # greedy short-circuit (also handles temp == 0 stably)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(x))

    x = x / float(temperature)
    x = _apply_top_k(x, int(top_k))
    x = _apply_top_p(x, float(top_p))

    probs = _softmax(x)
    if rng is None:
        rng = np.random.default_rng()
    # multinomial with one draw is just a single CDF lookup
    r = float(rng.random())
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, r))
    if idx >= probs.shape[0]:
        idx = int(probs.shape[0] - 1)
    return idx


# ----------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------

def _selftest() -> None:
    rng = np.random.default_rng(0)

    # synthetic logits: token 42 is by far the largest -> greedy must pick it
    vocab = 1024
    logits = rng.standard_normal(vocab).astype(np.float32)
    logits[42] = 50.0
    g = sample(logits, temperature=0.0)
    print(f"[sampler] greedy -> {g}")
    assert g == 42, g

    # top-k=5 + temp=1 should still strongly prefer token 42
    counts = np.zeros(vocab, dtype=np.int64)
    for _ in range(500):
        t = sample(logits, temperature=1.0, top_k=5, top_p=1.0, rng=rng)
        counts[t] += 1
    print(f"[sampler] top_k=5 most-sampled = {int(np.argmax(counts))} "
          f"({int(counts.max())}/500)")
    assert int(np.argmax(counts)) == 42

    # top-p=0.9 with a sharply peaked head: 10 tokens at logit=20 dominate
    # softmax mass, so the nucleus is contained in those 10.
    peaked = np.zeros(vocab, dtype=np.float32)
    peaked[:10] = 20.0
    seen = set()
    for _ in range(200):
        seen.add(sample(peaked, temperature=1.0, top_p=0.9, rng=rng))
    print(f"[sampler] top_p=0.9 distinct tokens seen = {len(seen)}")
    assert seen.issubset(set(range(10))), seen - set(range(10))

    print("[sampler] OK")


if __name__ == "__main__":
    _selftest()
