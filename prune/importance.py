"""Attention-head importance scoring, from scratch.

Implements the sensitivity-based head importance of Michel, Levy & Neubig,
"Are Sixteen Heads Really Better Than One?" (NeurIPS 2019). Each head h gets a
gate variable xi_h (=1 in the live model), and its importance is the expected
sensitivity of the loss to that gate:

    I_h = E_(x,y) | dL(x,y) / d xi_h |

A head the loss barely reacts to when you nudge its gate is a head you can
remove. One backward pass over an eval batch gives every head's score at once.

THIS IS A REIMPLEMENTATION, NOT A NEW METHOD. The method is Michel et al. 2019;
the value here is a clean from-scratch implementation with an honest analysis of
the accuracy/size tradeoff. See the README — the framing matters.

THE GUARD THAT MATTERS. A subtle failure sinks this silently: if the importance
scores don't actually *discriminate* between heads (all near-equal, or all near
zero because the model is degenerate or the eval text is trivial), then "prune
the least important heads" is really "prune arbitrary heads." The numbers still
compute, a ranking still comes out, and the plot still renders — it's just noise.
`check_discrimination` refuses to proceed when the scores carry no signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HeadImportance:
    scores: torch.Tensor        # (n_layers, n_heads), higher = more important
    n_layers: int
    n_heads: int

    def ranked(self) -> list[tuple[int, int, float]]:
        """All heads as (layer, head, score), least important first."""
        out = []
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                out.append((l, h, float(self.scores[l, h])))
        out.sort(key=lambda t: t[2])
        return out

    def discrimination_ratio(self) -> float:
        """How much the scores spread, normalised. Near 0 => all heads look the
        same => the ranking is meaningless. This is the health check."""
        s = self.scores.flatten()
        if s.numel() == 0 or float(s.mean()) == 0:
            return 0.0
        return float(s.std() / (s.mean().abs() + 1e-12))


def _find_attention_blocks(model):
    """Locate each transformer block's attention module and return
    (attn_module, n_heads) per layer. Supports GPT-2 / distilgpt2 layout; extend
    here for other architectures."""
    # GPT-2 family: model.transformer.h[i].attn
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        cfg = model.config
        n_heads = getattr(cfg, "n_head", None) or cfg.num_attention_heads
        return [(blk.attn, n_heads) for blk in tr.h]
    raise NotImplementedError(
        "Only the GPT-2 / distilgpt2 layout is wired up. Add this model's "
        "attention-module path to _find_attention_blocks()."
    )


def compute_head_importance(model, input_ids, labels) -> HeadImportance:
    """One backward pass -> importance for every head.

    Michel's method needs each head's gate to be a real leaf tensor that the loss
    gradient flows into. HuggingFace's built-in `head_mask` is applied for
    inference-time masking and does NOT reliably propagate a gradient back to the
    mask across versions (that's the RuntimeError an earlier version of this code
    hit). So instead we own the gate: a forward hook on each attention module
    multiplies its output by a per-head gate, and we read that gate's gradient.
    """
    model.eval()
    blocks = _find_attention_blocks(model)
    n_layers = len(blocks)
    n_heads = blocks[0][1]

    gates, handles = [], []
    for (attn, H) in blocks:
        gate = torch.ones(H, requires_grad=True, device=input_ids.device)
        gates.append(gate)

        def make_hook(g, H=H):
            def hook(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                B, T, E = tensor.shape
                head_dim = E // H
                tensor = tensor.view(B, T, H, head_dim) * g.view(1, 1, H, 1)
                tensor = tensor.reshape(B, T, E)
                if isinstance(out, tuple):
                    return (tensor,) + tuple(out[1:])
                return tensor
            return hook

        handles.append(attn.register_forward_hook(make_hook(gate)))

    try:
        out = model(input_ids=input_ids, labels=labels)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        if any(g.grad is None for g in gates):
            raise RuntimeError(
                "A head gate received no gradient — the attention hook isn't in "
                "the gradient path. Check _find_attention_blocks for this model."
            )
        scores = torch.stack([g.grad.abs().detach() for g in gates])  # (L, H)
    finally:
        for h in handles:
            h.remove()

    return HeadImportance(scores=scores, n_layers=n_layers, n_heads=n_heads)


def check_discrimination(imp: HeadImportance, min_ratio: float = 0.15) -> None:
    """Refuse to proceed on importance scores that carry no signal.

    A degenerate model (memorised its data) or a trivial eval text produces
    near-zero, near-uniform gradients. The pipeline would still 'work' — rank
    heads, prune, plot a curve — but every number would be noise, and 'prune
    least important' would silently become 'prune random'. Better to fail here
    than to publish a meaningless frontier.
    """
    r = imp.discrimination_ratio()
    if r < min_ratio:
        raise ValueError(
            f"Head-importance scores barely discriminate (spread ratio {r:.3f} "
            f"< {min_ratio}). The ranking would be arbitrary. Likely causes: the "
            f"eval text is too short/trivial, the model is degenerate, or the "
            f"head_mask gate isn't wired to the loss. Fix the eval signal before "
            f"pruning — a frontier built on this is noise."
        )
