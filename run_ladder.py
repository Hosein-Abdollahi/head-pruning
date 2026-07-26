"""Sweep a ladder of models and show how prunable redundancy scales with size.

    python run_ladder.py                                  # distilgpt2, gpt2, gpt2-medium
    python run_ladder.py --models distilgpt2,gpt2

The single-model run (run.py) shows *that* pruning works on one model. This asks
a question Michel et al. didn't: **does the amount of prunable redundancy scale
with model size?** The two-model result in the README hints at it (distilled
models have less to prune); this turns the hint into a curve by measuring the
elbow across a size ladder.

Output: results/ladder.csv + ladder_elbow.png (elbow fraction vs model size).
The hypothesis, if it holds: bigger models carry more redundancy, so the elbow
moves right as the model grows — and the already-distilled model sits below the
trend, because distillation already spent that redundancy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from prune.importance import check_discrimination, compute_head_importance
from prune.pruner import build_prune_mask, eval_loss
from run import EVAL_TEXT, random_mask

ROOT = Path(__file__).parent

# param counts (millions) for the standard GPT-2 ladder, for the x-axis
KNOWN_SIZES = {
    "distilgpt2": 82, "gpt2": 124, "gpt2-medium": 355,
    "gpt2-large": 774, "gpt2-xl": 1558,
}


def elbow_fraction(rows, base_ppl, tol=0.10) -> float:
    """Largest fraction whose importance-pruned perplexity stays within tol of
    baseline. This is the 'free lunch' boundary."""
    e = 0.0
    for r in rows:
        if r["importance_ppl"] <= base_ppl * (1 + tol):
            e = r["fraction"]
    return e


def sweep_one(model_name, device, fractions, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").to(device)
    ids = enc["input_ids"]

    imp = compute_head_importance(model, ids, ids)
    check_discrimination(imp)
    base = eval_loss(model, ids, ids)
    base_ppl = float(torch.tensor(base).exp())

    rows = []
    for f in fractions:
        pr = build_prune_mask(imp, f)
        il = eval_loss(model, ids, ids, pr.kept_mask)
        rl = eval_loss(model, ids, ids, random_mask(imp, f, seed))
        rows.append({
            "fraction": f,
            "importance_ppl": float(torch.tensor(il).exp()),
            "random_ppl": float(torch.tensor(rl).exp()),
        })
    n_heads = imp.n_layers * imp.n_heads
    won = sum(1 for r in rows if r["importance_ppl"] < r["random_ppl"])
    return {
        "model": model_name,
        "params_m": KNOWN_SIZES.get(model_name, None),
        "n_layers": imp.n_layers, "n_heads_per_layer": imp.n_heads,
        "total_heads": n_heads,
        "discrimination": round(imp.discrimination_ratio(), 3),
        "baseline_ppl": round(base_ppl, 2),
        "elbow": elbow_fraction(rows, base_ppl),
        "importance_wins": won, "n_fractions": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="distilgpt2,gpt2,gpt2-medium")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    summaries = []
    print(f"{'model':<16}{'params':>8}{'heads':>7}{'discrim':>9}"
          f"{'elbow':>7}{'imp wins':>10}")
    print("-" * 57)
    for m in models:
        s = sweep_one(m, args.device, fracs, args.seed)
        summaries.append(s)
        p = f"{s['params_m']}M" if s["params_m"] else "?"
        print(f"{m:<16}{p:>8}{s['total_heads']:>7}{s['discrimination']:>9}"
              f"{s['elbow']:>6.0%}{s['importance_wins']:>7}/{s['n_fractions']}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "ladder.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    (out / "ladder.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out}/ladder.csv, ladder.json")

    try:
        _plot_elbow(summaries, out / "ladder_elbow.png")
        print(f"wrote {out}/ladder_elbow.png")
    except ImportError:
        print("(matplotlib not installed; skipped plot)")

    # the interpretation, stated plainly
    sized = [s for s in summaries if s["params_m"]]
    if len(sized) >= 2:
        sized.sort(key=lambda s: s["params_m"])
        trend = "rises" if sized[-1]["elbow"] > sized[0]["elbow"] else "does not rise"
        print(f"\nElbow {trend} with model size across the ladder "
              f"({sized[0]['model']} {sized[0]['elbow']:.0%} -> "
              f"{sized[-1]['model']} {sized[-1]['elbow']:.0%}).")


def _plot_elbow(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(s["params_m"], s["elbow"], s["model"]) for s in summaries
           if s["params_m"]]
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] * 100 for p in pts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", linewidth=2, markersize=9)
    for x, y, name in pts:
        ax.annotate(name, (x, y * 100), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("model size (million parameters)")
    ax.set_ylabel("elbow — % heads prunable for <10% ppl rise")
    ax.set_title("Prunable redundancy vs model size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
