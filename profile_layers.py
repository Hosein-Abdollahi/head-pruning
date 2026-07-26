"""Where are the prunable heads? A layer-wise view of head importance.

Michel et al. report *how many* heads are redundant. This shows *where* they
live. Two outputs from the same importance scores run.py already computes:

  1. A heatmap of per-head importance (layer x head) — the raw structure.
  2. A per-layer summary: for each layer, what fraction of its heads survive a
     global importance threshold. Early and late layers often behave very
     differently, and that difference says something about how the model uses
     attention that a single pruning curve hides.

    python profile_layers.py --model gpt2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from prune.importance import check_discrimination, compute_head_importance
from run import EVAL_TEXT

ROOT = Path(__file__).parent


def layer_survival(imp, global_fraction=0.5):
    """For a global prune of `global_fraction` least-important heads, what
    fraction of each layer's heads survive? Reveals which layers the pruner
    empties and which it leaves intact."""
    flat = imp.scores.flatten()
    k = int(round(global_fraction * flat.numel()))
    thresh = flat.kthvalue(max(1, k)).values if k > 0 else flat.min() - 1
    kept = (imp.scores > thresh)                       # (L, H) bool
    return [float(kept[l].float().mean()) for l in range(imp.n_layers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prune-fraction", type=float, default=0.5)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").to(args.device)
    ids = enc["input_ids"]

    imp = compute_head_importance(model, ids, ids)
    check_discrimination(imp)
    L, H = imp.n_layers, imp.n_heads
    print(f"model: {args.model} | {L} layers x {H} heads\n")

    survival = layer_survival(imp, args.prune_fraction)
    print(f"At a global {args.prune_fraction:.0%} prune, per-layer survival:")
    print(f"{'layer':>6}{'heads kept':>12}{'survival':>10}")
    print("-" * 28)
    for l in range(L):
        kept = int(round(survival[l] * H))
        bar = "#" * kept + "." * (H - kept)
        print(f"{l:>6}{kept:>4}/{H:<2}   {survival[l]:>6.0%}  {bar}")

    # save profile
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"layer_profile_{args.model.replace('/', '_')}.csv",
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "mean_importance", "survival_at_prune"])
        for l in range(L):
            w.writerow([l, round(float(imp.scores[l].mean()), 6),
                        round(survival[l], 3)])

    # heatmap
    try:
        _heatmap(imp, out / f"heatmap_{args.model.replace('/', '_')}.png",
                 args.model)
        print(f"\nwrote {out}/heatmap_{args.model.replace('/', '_')}.png")
    except ImportError:
        print("\n(matplotlib not installed; skipped heatmap)")

    # interpretation
    hi = max(range(L), key=lambda l: survival[l])
    lo = min(range(L), key=lambda l: survival[l])
    print(f"\nLayer {hi} is most pruning-resistant "
          f"({survival[hi]:.0%} kept); layer {lo} is most prunable "
          f"({survival[lo]:.0%} kept). Pruning is not uniform across depth.")


def _heatmap(imp, path, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = imp.scores.numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(scores, aspect="auto", cmap="viridis")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Per-head importance — {model_name}\n(brighter = more important)")
    fig.colorbar(im, ax=ax, label="importance |dL/dξ|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
