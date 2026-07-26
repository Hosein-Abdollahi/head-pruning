"""Tests for the head-pruning logic.

The guard tests matter most: the whole study is worthless if importance scores
that carry no signal are allowed through, because then 'prune least important'
silently becomes 'prune random' and every downstream number is noise.
"""

import pytest
import torch

from prune.importance import HeadImportance, check_discrimination
from prune.pruner import build_prune_mask


def _imp(scores):
    t = torch.tensor(scores, dtype=torch.float32)
    return HeadImportance(scores=t, n_layers=t.shape[0], n_heads=t.shape[1])


# --- the integrity guard --------------------------------------------------

def test_guard_passes_on_discriminating_scores():
    imp = _imp([[0.1, 0.9, 0.5], [0.2, 0.8, 0.4]])
    check_discrimination(imp)                       # should not raise


def test_guard_fires_on_uniform_scores():
    """All heads equal => ranking is arbitrary => must refuse."""
    imp = _imp([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="discriminate"):
        check_discrimination(imp)


def test_guard_fires_on_zero_scores():
    """The degenerate case the prototype actually hit: a memorised model gives
    zero gradient everywhere, so every head scores 0."""
    imp = _imp([[0.0, 0.0], [0.0, 0.0]])
    assert imp.discrimination_ratio() == 0.0
    with pytest.raises(ValueError):
        check_discrimination(imp)


def test_discrimination_ratio_grows_with_spread():
    tight = _imp([[0.49, 0.50, 0.51]])
    wide = _imp([[0.1, 0.5, 0.9]])
    assert wide.discrimination_ratio() > tight.discrimination_ratio()


# --- prune mask construction ----------------------------------------------

def test_prunes_least_important_first():
    imp = _imp([[0.9, 0.1, 0.8, 0.2]])              # heads 1,3 are weakest
    mask = build_prune_mask(imp, 0.5).kept_mask
    assert mask[0, 1] == 0 and mask[0, 3] == 0      # weak ones removed
    assert mask[0, 0] == 1 and mask[0, 2] == 1      # strong ones kept


def test_never_empties_a_layer():
    """A layer with zero heads is a different, harsher intervention that would
    confound the frontier."""
    imp = _imp([[0.1, 0.2], [0.3, 0.4]])
    mask = build_prune_mask(imp, 0.9).kept_mask     # ask to remove 90%
    assert (mask.sum(dim=1) >= 1).all()


def test_fraction_zero_removes_nothing():
    imp = _imp([[0.1, 0.9], [0.5, 0.5]])
    r = build_prune_mask(imp, 0.0)
    assert r.heads_removed == 0
    assert (r.kept_mask == 1).all()


def test_param_reduction_matches_heads_removed():
    imp = _imp([[0.1, 0.2, 0.3, 0.4, 0.5]])
    r = build_prune_mask(imp, 0.4)
    assert abs(r.param_reduction - r.heads_removed / r.heads_total) < 1e-9


def test_ranked_is_least_important_first():
    imp = _imp([[0.5, 0.1], [0.9, 0.3]])
    ranked = imp.ranked()
    scores = [s for _, _, s in ranked]
    assert scores == sorted(scores)
    assert ranked[0][2] == pytest.approx(0.1)       # weakest first


def test_invalid_fraction_rejected():
    imp = _imp([[0.1, 0.2]])
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            build_prune_mask(imp, bad)


# --- layer-wise profile (feature #2) --------------------------------------

def test_layer_survival_varies_by_layer():
    """Late layers should be prunable differently from early ones — the whole
    point of the layer profile."""
    from profile_layers import layer_survival
    # layer 0 all-important, layer 1 all-weak
    imp = _imp([[0.9, 0.9, 0.9], [0.01, 0.01, 0.01]])
    surv = layer_survival(imp, 0.5)
    assert surv[0] > surv[1]                        # strong layer survives more


def test_layer_survival_full_prune_keeps_nothing_extra():
    from profile_layers import layer_survival
    imp = _imp([[0.5, 0.5], [0.5, 0.5]])
    surv = layer_survival(imp, 0.0)                 # prune nothing
    assert all(s == 1.0 for s in surv)              # everything survives


# --- ladder elbow (feature #1) --------------------------------------------

def test_elbow_is_last_fraction_within_tolerance():
    from run_ladder import elbow_fraction
    rows = [{"fraction": 0.0, "importance_ppl": 50},
            {"fraction": 0.1, "importance_ppl": 53},   # +6%, ok
            {"fraction": 0.2, "importance_ppl": 54},   # +8%, ok
            {"fraction": 0.3, "importance_ppl": 60}]   # +20%, no
    assert elbow_fraction(rows, 50, tol=0.10) == 0.2


def test_elbow_zero_when_first_prune_already_breaks():
    from run_ladder import elbow_fraction
    rows = [{"fraction": 0.0, "importance_ppl": 50},
            {"fraction": 0.1, "importance_ppl": 80}]   # +60%, breaks immediately
    assert elbow_fraction(rows, 50, tol=0.10) == 0.0


# --- integration: the model path that actually broke ----------------------
# These build a tiny REAL GPT-2 (no download) and exercise the hook-based
# gradient mechanism end to end — the exact code that failed with the
# `head_mask received no gradient` bug. The unit tests above feed hand-made
# importance tensors and would NOT catch a regression here.

import pytest


def _tiny_gpt2():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.GPT2Config(n_layer=3, n_head=4, n_embd=64, vocab_size=100,
                        n_positions=32)
    model = tf.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 100, (2, 16))
    return model, ids


def test_importance_flows_gradient_on_real_gpt2():
    """The regression guard: compute_head_importance must return real per-head
    scores via the hooks. This is the exact path that raised
    'head_mask received no gradient'."""
    from prune.importance import compute_head_importance
    model, ids = _tiny_gpt2()
    imp = compute_head_importance(model, ids, ids)
    assert imp.scores.shape == (3, 4)               # (n_layers, n_heads)
    assert imp.scores.abs().sum() > 0               # gradient actually flowed
    assert not imp.scores.isnan().any()


def test_importance_scores_discriminate_on_real_model():
    """A randomly-initialised real model gives a non-degenerate signal, so the
    guard should pass (not fire)."""
    from prune.importance import compute_head_importance, check_discrimination
    model, ids = _tiny_gpt2()
    imp = compute_head_importance(model, ids, ids)
    assert imp.discrimination_ratio() > 0.15
    check_discrimination(imp)                        # must not raise


def test_eval_loss_hook_changes_output_when_heads_pruned():
    """apply_head_mask (the eval-time hook) must actually gate heads — pruning
    should change the loss, proving the mask is applied, not silently ignored."""
    from prune.importance import compute_head_importance
    from prune.pruner import build_prune_mask, eval_loss
    model, ids = _tiny_gpt2()
    imp = compute_head_importance(model, ids, ids)
    base = eval_loss(model, ids, ids)
    heavy = eval_loss(model, ids, ids,
                      build_prune_mask(imp, 0.8).kept_mask)
    assert base != heavy                             # mask had an effect


def test_pruning_none_matches_unmasked():
    """A keep-mask of all ones must equal no mask — the hook must be a true
    identity when nothing is pruned (guards against the hook corrupting output)."""
    from prune.importance import compute_head_importance
    from prune.pruner import build_prune_mask, eval_loss
    model, ids = _tiny_gpt2()
    imp = compute_head_importance(model, ids, ids)
    unmasked = eval_loss(model, ids, ids)
    all_kept = eval_loss(model, ids, ids, build_prune_mask(imp, 0.0).kept_mask)
    assert abs(unmasked - all_kept) < 1e-4
