"""Static-shape replacement for HuggingFace transformers ``repeat_kv``.

Problem
-------
The stock implementation in
``transformers/models/gemma3/modeling_gemma3.py`` (line 268) and
``transformers/models/gemma4/modeling_gemma4.py`` (line 800) is::

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

When the model is captured with ``torch.onnx.export(dynamo=True)`` the
symbolic tracer sees that ``batch`` and ``slen`` are ``SymInt`` values
(they came from ``hidden_states.shape`` and the export dynamic_shapes
contract marks batch and sequence as dynamic).  ``Tensor.expand`` uses
the sentinel ``-1`` to mean "keep current size", so the dynamo decomposer
emits a guarded ``Where(Equal(shape_i, -1), current_size_i, shape_i)``
for each dimension, fed by ``ConstantOfShape``.  The chain
``ConstantOfShape -> Equal -> Where -> Expand`` is what QAIRT refuses to
constant-fold even though every branch evaluates to the same literal
shape ``[1, num_kv_heads, n_rep, seq, head_dim]`` at runtime.

Fix
---
We bypass ``expand`` entirely.  ``repeat_interleave`` lowers to a
``Tile``/``Reshape`` pair that takes integer literals (``n_rep`` is a
Python int known at trace time) so no Where guard is produced.  As a
defensive fallback we also offer a pure ``view + tile`` form that uses
only the *known-static* ``n_rep`` and ``num_kv_heads`` from the module
config; the batch and seqlen dimensions are inferred via ``-1`` placed
through ``view``/``reshape`` which dynamo lowers to plain ``Reshape``
(no Where).

Apply the patch with::

    import source_patches.repeat_kv_static  # noqa: F401  (auto-applies)

or explicitly::

    from source_patches.repeat_kv_static import patched_repeat_kv
    import transformers.models.gemma3.modeling_gemma3 as g3
    import transformers.models.gemma4.modeling_gemma4 as g4
    g3.repeat_kv = patched_repeat_kv
    g4.repeat_kv = patched_repeat_kv
"""

from __future__ import annotations

import torch


def patched_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Static-shape GQA K/V repeat that avoids the Where->Expand chain.

    Equivalent to ``torch.repeat_interleave(hidden_states, n_rep, dim=1)``
    but written as ``view + repeat (Tile) + reshape`` so that every shape
    operand is either a Python ``int`` (``n_rep``) or the literal ``-1``
    sentinel that ONNX lowers to a plain ``Reshape`` (not a Where guard).

    Input  : (batch, num_key_value_heads, seqlen, head_dim)
    Output : (batch, num_key_value_heads * n_rep, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    # n_rep is a Python int from the module config -> constant in the
    # exported graph.  num_kv_heads and head_dim are static once the
    # config is loaded; only batch and seqlen are dynamic SymInts, and
    # we propagate those exclusively through ``-1`` placeholders so the
    # exporter cannot emit a Where(Equal(...), ...) guard for them.
    num_kv_heads = hidden_states.size(1)
    head_dim = hidden_states.size(-1)

    # (B, Hkv, S, D) -> (B, Hkv, 1, S, D)
    x = hidden_states.unsqueeze(2)
    # Tile only along the new repeat axis.  ``repeat`` takes a list of
    # Python ints, so the resulting graph node is a constant Tile -
    # no dynamic shape construction is needed.
    x = x.repeat(1, 1, n_rep, 1, 1)
    # Collapse (Hkv, n_rep) -> Hkv * n_rep with a single reshape that
    # only needs the *static* product ``num_kv_heads * n_rep`` and
    # ``head_dim``; batch and seqlen ride through as ``-1``.
    # Using contiguous() here is cheap (repeat already produced a
    # contiguous buffer) and guarantees view-compatibility on all
    # backends.
    return x.contiguous().view(-1, num_kv_heads * n_rep, x.size(3), head_dim)


def apply() -> None:
    """Install the patch in every Gemma3/Gemma4 module that defines it.

    Idempotent: safe to call multiple times.
    """
    targets = [
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.gemma4.modeling_gemma4",
    ]
    for name in targets:
        try:
            mod = __import__(name, fromlist=["repeat_kv"])
        except Exception:
            continue
        if getattr(mod, "repeat_kv", None) is not patched_repeat_kv:
            mod.repeat_kv = patched_repeat_kv


# Auto-apply on import so a single ``import source_patches.repeat_kv_static``
# from the export script is enough.
apply()
