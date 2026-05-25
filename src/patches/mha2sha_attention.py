"""MHA->SHA monkey-patch for Gemma 4 attention.

Background
----------
QNN Hexagon v69 rejects the ONNX ``Expand`` op that
``transformers.models.gemma4.modeling_gemma4.repeat_kv`` produces when it
broadcasts the GQA K/V heads from ``num_key_value_heads`` to
``num_attention_heads``.  The ``view+repeat+reshape`` workaround in
``repeat_kv_static.py`` removes the explicit ``Expand`` but ONNX export
still synthesises an ``Expand`` (via dynamo's ``-1`` guard) once the K/V
tensor enters the ``Q @ K^T`` matmul because the GQA Q-dimension
broadcasts implicitly.

The Qualcomm ExecuTorch flow solves this with ``use_mha2sha=True`` -
a graph pass that splits multi-head / grouped-query attention into
``num_attention_heads`` independent single-head attentions BEFORE QNN
sees the graph.  Each head has its own slice of Q, the appropriate
shared slice of K/V, and runs ``softmax(Q_h K_h^T) V_h`` in isolation.
There is no broadcast on the heads dimension at all because the heads
dimension is literally unrolled into ``num_attention_heads`` parallel
sub-graphs.

This module implements the same transform at the PyTorch ``eager``
attention level.  We replace the module-level
``eager_attention_forward`` with a Python loop over
``num_attention_heads``; each iteration slices the appropriate Q head
and the shared K/V head, does a (1, S, D) x (1, D, S) matmul, softmaxes,
and weights V.  When ONNX-exported the loop is unrolled (``range(n_heads)``
is a Python ``int`` known at trace time) so the resulting graph has
``n_heads`` independent attention chains and **no Expand on the head
dimension**.

Cost
----
- We trade one batched matmul ``(B, H, S, D) x (B, H, D, S)`` for ``H``
  separate matmuls of shape ``(B, 1, S, D) x (B, 1, D, S)``.  On Hexagon
  the per-head MAC cost is the same; the loop only inflates the graph
  description (``H`` more nodes), not the runtime FLOPs.
- The output ``cat`` is along the heads dim; reshape/transpose semantics
  are identical to the batched version so downstream ``o_proj`` sees the
  same ``(B, S, H*D)`` tensor.

Usage
-----
    import source_patches.mha2sha_attention  # auto-applies on import

or explicitly::

    from source_patches.mha2sha_attention import sha_eager_attention_forward
    import transformers.models.gemma4.modeling_gemma4 as g4
    g4.eager_attention_forward = sha_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = sha_eager_attention_forward
    except Exception:
        pass
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sha_eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
):
    """Per-head (single-head attention, SHA) replacement for Gemma4 GQA.

    Args:
        query: (B, n_heads, S, head_dim)
        key:   (B, n_kv_heads, S_k, head_dim)  (n_kv_heads <= n_heads)
        value: (B, n_kv_heads, S_k, head_dim)
        attention_mask: (B, 1 or n_heads, S, S_k) or None
        scaling: 1/sqrt(head_dim) typically
        softcap: gemma2/gemma4 attn-logit softcap (optional)

    Returns:
        attn_output: (B, S, n_heads, head_dim) - same layout as the
            stock ``eager_attention_forward`` after its trailing
            ``transpose(1, 2).contiguous()``.
        attn_weights: None (not needed for inference; matches the
            ``output_attentions=False`` contract used at export time).
    """
    if scaling is None:
        scaling = module.head_dim ** -0.5

    n_heads = query.shape[1]
    n_kv = key.shape[1]
    # n_rep is a Python int known at trace time.
    n_rep = n_heads // n_kv

    in_dtype = query.dtype

    # Precompute the transposed K once per kv-head to share across the
    # n_rep Q-heads that map to it.  This is a NOOP from QNN's POV
    # (a single Transpose, no head broadcast).
    key_t = key.transpose(-2, -1)  # (B, n_kv, D, S_k)

    outputs = []
    for h in range(n_heads):
        kv_h = h // n_rep  # which K/V head this Q head reads
        q_h = query[:, h:h + 1, :, :]         # (B, 1, S, D)
        k_h_t = key_t[:, kv_h:kv_h + 1, :, :]  # (B, 1, D, S_k)
        v_h = value[:, kv_h:kv_h + 1, :, :]    # (B, 1, S_k, D)

        scores = torch.matmul(q_h, k_h_t) * scaling  # (B, 1, S, S_k)

        if softcap is not None:
            scores = scores / softcap
            scores = torch.tanh(scores)
            scores = scores * softcap

        if attention_mask is not None:
            # mask is (B, 1, S, S_k) for GQA - share across heads
            mask_slice = attention_mask[:, :1, :, :] if attention_mask.shape[1] > 1 else attention_mask
            scores = scores + mask_slice.to(scores.dtype)

        # softmax stays in working dtype (fp16 during export) to match the
        # _eager_attention_fp16 wrappers used in the existing scripts.
        probs = F.softmax(scores, dim=-1)
        if dropout and module.training:
            probs = F.dropout(probs, p=float(dropout), training=True)

        out_h = torch.matmul(probs, v_h)  # (B, 1, S, D)
        outputs.append(out_h)

    # (B, n_heads, S, D) then transpose to (B, S, n_heads, D) to match the
    # contract expected by the caller (which then reshapes to (B, S, H*D)).
    attn_output = torch.cat(outputs, dim=1)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.to(in_dtype), None


def apply() -> None:
    """Install the SHA forward in every Gemma3/Gemma4 module that
    defines ``eager_attention_forward``.  Idempotent."""
    targets = [
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.gemma4.modeling_gemma4",
    ]
    for name in targets:
        try:
            mod = __import__(name, fromlist=["eager_attention_forward"])
        except Exception:
            continue
        if getattr(mod, "eager_attention_forward", None) is not sha_eager_attention_forward:
            mod.eager_attention_forward = sha_eager_attention_forward

    # Also register in the global attention-function registry so anything
    # that resolves "eager" by name (rather than capturing the function
    # at module-import time) sees the SHA implementation.
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = sha_eager_attention_forward
    except Exception:
        pass


apply()
