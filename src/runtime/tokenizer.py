"""Host-side Gemma 4 E2B tokenizer wrapper.

Thin facade over HuggingFace AutoTokenizer. Used by the Python inference
loop that drives the Hexagon v69 NPU graph; the on-device path uses the
C++ SentencePiece wrapper in tokenizer.h/.cpp.

Special tokens (canonical for Gemma 4 E2B, mirrors tokenizer.h):
    PAD = 0
    EOS = 1   (also <end_of_turn> = 106 per config.json eos_token_id list)
    BOS = 2
    UNK = 3
Vocab size: 262_144
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union

from transformers import AutoTokenizer

DEFAULT_CHECKPOINT = "/home/hardoker77/gemma4_e2b_v69/checkpoints/gemma-4-e2b-it"

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
VOCAB_SIZE = 262_144


class GemmaTokenizer:
    """Wraps HF AutoTokenizer with a small, stable surface."""

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT):
        ckpt = Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt}")
        # use_fast=True picks tokenizer.json (rust); falls back to slow if absent.
        self.checkpoint_dir = str(ckpt)
        self.hf = AutoTokenizer.from_pretrained(self.checkpoint_dir, use_fast=True)
        # sanity-check vocab; HF reports `len(tokenizer)` which can include
        # added tokens, so just check it is at least the model vocab.
        if len(self.hf) < VOCAB_SIZE:
            # not fatal — some sentencepiece dumps under-report — just warn
            print(f"[tokenizer] warning: HF reports len={len(self.hf)} < {VOCAB_SIZE}")

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        prompt: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode UTF-8 `prompt` -> list[int]. Adds BOS by default."""
        # add_special_tokens=False so we control BOS/EOS placement explicitly.
        ids = self.hf.encode(prompt, add_special_tokens=False)
        if add_bos and (not ids or ids[0] != BOS_ID):
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(
        self,
        token_ids: Union[int, Sequence[int], Iterable[int]],
        skip_special: bool = True,
    ) -> str:
        """Decode token id(s) -> UTF-8 string."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        else:
            token_ids = list(token_ids)
        return self.hf.decode(token_ids, skip_special_tokens=skip_special)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE


# ----------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------

def _selftest() -> None:
    tk = GemmaTokenizer()
    text = "hello"
    ids = tk.encode(text, add_bos=False)
    back = tk.decode(ids)
    print(f"[tokenizer] encode({text!r}) -> {ids}")
    print(f"[tokenizer] decode({ids}) -> {back!r}")
    assert back.strip() == text, f"roundtrip mismatch: {back!r} != {text!r}"
    print("[tokenizer] roundtrip OK")


if __name__ == "__main__":
    _selftest()
