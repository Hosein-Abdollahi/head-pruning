"""Turn importance scores into a pruned model, and measure what it saved.

Two things this module insists on being honest about:

1. STRUCTURED, NOT SPARSE. Zeroing individual weights ("unstructured pruning")
   makes a model *sparse* but not *smaller* or *faster* on normal hardware — the
   zeros still occupy memory and still get multiplied. Removing whole attention
   heads is structured pruning: the parameters genuinely leave and inference
   genuinely speeds up. Since the request is "lower resources", only structured
   pruning delivers it, so that's what this does.

2. MASK-BASED vs PHYSICAL REMOVAL. This applies a head_mask (the head still
   exists but is gated to zero), which measures the *accuracy* effect of pruning
   exactly, without surgery on the weight matrices. The measured FLOP/param
   savings are what physical removal *would* yield — reported as such, not
   pretended to be already-realised speedups. Physically deleting the head
   dimensions is a mechanical follow-up; the mask gives the honest quality curve.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .importance import HeadImportance, _find_attention_blocks


@contextmanager
def apply_head_mask(model, keep_mask):
    """Temporarily gate attention heads via forward hooks (same mechanism as the
    importance scorer, so eval matches scoring). keep_mask: (n_layers, n_heads),
    1 = keep, 0 = prune. Yields, then removes the hooks."""
    if keep_mask is None:
        yield
        return
    blocks = _find_attention_blocks(model)
    handles = []
    for i, (attn, H) in enumerate(blocks):
        gate = keep_mask[i].to(next(model.parameters()).device)

        def make_hook(g, H=H):
            def hook(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                B, T, E = tensor.shape
                hd = E // H
                tensor = (tensor.view(B, T, H, hd) * g.view(1, 1, H, 1)).reshape(B, T, E)
                return (tensor,) + tuple(out[1:]) if isinstance(out, tuple) else tensor
            return hook

        handles.append(attn.register_forward_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@dataclass
class PruneResult:
    fraction: float             # fraction of heads removed
    heads_removed: int
    heads_total: int
    param_reduction: float      # fraction of attention params that would leave
    kept_mask: torch.Tensor     # (n_layers, n_heads), 1 = keep


def build_prune_mask(imp: HeadImportance, fraction: float) -> PruneResult:
    """Remove the `fraction` least-important heads. Returns a keep-mask.

    At least one head per layer is always kept — a layer with zero heads is a
    dead layer, which is a different (harsher) intervention than head pruning and
    would confound the curve."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")

    L, H = imp.n_layers, imp.n_heads
    total = L * H
    k = int(round(fraction * total))

    mask = torch.ones(L, H)
    if k > 0:
        ranked = imp.ranked()                     # least important first
        removed = 0
        per_layer_kept = [H] * L
        for (l, h, _score) in ranked:
            if removed >= k:
                break
            if per_layer_kept[l] <= 1:            # never empty a layer
                continue
            mask[l, h] = 0.0
            per_layer_kept[l] -= 1
            removed += 1
    else:
        removed = 0

    return PruneResult(
        fraction=fraction,
        heads_removed=removed,
        heads_total=total,
        param_reduction=removed / total,
        kept_mask=mask,
    )


@torch.no_grad()
def eval_loss(model, input_ids, labels, keep_mask=None) -> float:
    with apply_head_mask(model, keep_mask):
        out = model(input_ids=input_ids, labels=labels)
    return float(out.loss)


@torch.no_grad()
def measure_latency(model, input_ids, keep_mask=None, runs=5) -> float:
    """Mean forward-pass wall-clock (ms). With masking the head still runs, so
    this is an UPPER bound on the physically-pruned model's latency, not the
    saving itself — reported honestly as such in the README."""
    with apply_head_mask(model, keep_mask):
        for _ in range(2):                               # warmup
            model(input_ids=input_ids)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(input_ids=input_ids)
        return (time.perf_counter() - t0) / runs * 1000.0
