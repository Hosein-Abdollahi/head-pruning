"""Run the pruning sweep on a real model and find the elbow.

    python run.py                              # distilgpt2, CPU, ~2 min
    python run.py --model gpt2 --device cuda

The output is the accuracy/size frontier: at each pruning fraction, the loss
(perplexity) of the model with that many least-important heads removed, plus a
random-pruning baseline. The story Michel et al. found — and that this
reproduces — is an *elbow*: a flat stretch where pruning is nearly free, then a
cliff. Where the elbow sits is the model's answer to "how many heads are
redundant?"

The random baseline is not decoration. If importance-based pruning tracks the
random baseline, the importance scores add nothing, and the whole method is
theatre. The gap between them is the evidence that importance means something.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from prune.importance import check_discrimination, compute_head_importance
from prune.pruner import build_prune_mask, eval_loss, measure_latency

ROOT = Path(__file__).parent

EVAL_TEXT = (
    "Artificial intelligence has transformed how software is written and tested. "
    "Large language models can summarise documents, translate between languages, "
    "and answer questions about complex topics. However, these models are "
    "expensive to run, and researchers study ways to make them smaller and "
    "faster without losing accuracy. One approach removes redundant components "
    "from the network. The capital of France is Paris, and water boils at one "
    "hundred degrees Celsius at sea level."
)


def random_mask(imp, fraction, seed):
    """Prune a random `fraction` of heads (never emptying a layer) — the baseline
    importance-based pruning must beat to justify itself."""
    g = torch.Generator().manual_seed(seed)
    noise = type(imp)(scores=torch.rand(imp.n_layers, imp.n_heads, generator=g),
                      n_layers=imp.n_layers, n_heads=imp.n_heads)
    return build_prune_mask(noise, fraction).kept_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    enc = tok(EVAL_TEXT, return_tensors="pt").to(args.device)
    ids, labels = enc["input_ids"], enc["input_ids"]

    print(f"model: {args.model} | device: {args.device}")

    # --- importance, with the integrity gate --------------------------------
    imp = compute_head_importance(model, ids, labels)
    print(f"heads: {imp.n_layers} layers x {imp.n_heads} = "
          f"{imp.n_layers * imp.n_heads}")
    print(f"discrimination ratio: {imp.discrimination_ratio():.3f}")
    check_discrimination(imp)          # refuses to proceed on noise
    print("importance scores carry signal — proceeding\n")

    base = eval_loss(model, ids, labels)
    base_ppl = float(torch.tensor(base).exp())
    base_ms = measure_latency(model, ids)
    print(f"baseline: loss {base:.4f}  ppl {base_ppl:.1f}  {base_ms:.1f} ms\n")

    fracs = [float(x) for x in args.fractions.split(",")]
    rows = []
    print(f"{'frac':>6}{'heads':>7}{'importance ppl':>16}{'random ppl':>13}"
          f"{'gap':>8}")
    print("-" * 50)
    for f in fracs:
        pr = build_prune_mask(imp, f)
        imp_loss = eval_loss(model, ids, labels, pr.kept_mask)
        rnd_loss = eval_loss(model, ids, labels, random_mask(imp, f, args.seed))
        imp_ppl = float(torch.tensor(imp_loss).exp())
        rnd_ppl = float(torch.tensor(rnd_loss).exp())
        gap = rnd_ppl - imp_ppl
        print(f"{f:>6.2f}{pr.heads_removed:>4}/{pr.heads_total:<2}"
              f"{imp_ppl:>16.2f}{rnd_ppl:>13.2f}{gap:>8.2f}")
        rows.append({
            "fraction": f, "heads_removed": pr.heads_removed,
            "heads_total": pr.heads_total,
            "importance_ppl": round(imp_ppl, 3),
            "random_ppl": round(rnd_ppl, 3),
            "importance_loss": round(imp_loss, 4),
            "param_reduction": round(pr.param_reduction, 3),
        })

    # --- find the elbow: last fraction before ppl rises > 10% over baseline --
    elbow = 0.0
    for r in rows:
        if r["importance_ppl"] <= base_ppl * 1.10:
            elbow = r["fraction"]
    print("-" * 50)
    print(f"\nELBOW: up to {elbow:.0%} of heads prune with <10% perplexity rise "
          f"(importance-based). Beyond that, quality falls off.")
    matched = [r for r in rows if r["importance_ppl"] < r["random_ppl"]]
    print(f"importance beat random on {len(matched)}/{len(rows)} fractions "
          f"(if this is most of them, the scores are meaningful).")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "model": args.model, "baseline_ppl": round(base_ppl, 3),
        "elbow_fraction": elbow,
        "importance_beat_random": len(matched), "n_fractions": len(rows),
    }, indent=2))
    print(f"\nwrote {out}/frontier.csv, summary.json")

    try:
        _plot(rows, base_ppl, out / "frontier.png", args.model)
        print(f"wrote {out}/frontier.png")
    except ImportError:
        print("(matplotlib not installed; skipped plot)")


def _plot(rows, base_ppl, path, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [r["fraction"] for r in rows]
    imp = [r["importance_ppl"] for r in rows]
    rnd = [r["random_ppl"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fr, imp, "o-", label="importance-based", linewidth=2)
    ax.plot(fr, rnd, "s--", label="random baseline", linewidth=2, alpha=0.7)
    ax.axhline(base_ppl, color="gray", ls=":", label="unpruned")
    ax.axhline(base_ppl * 1.1, color="crimson", ls=":", alpha=0.5,
               label="+10% threshold")
    ax.set_xlabel("fraction of attention heads removed")
    ax.set_ylabel("perplexity (lower = better)")
    ax.set_title(f"Head pruning frontier — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
