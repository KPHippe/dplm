import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fast_dplm.diagnostic_sampling import generate_diagnostic, nucleus
from fast_dplm.sampling import generate, transition
from test_fast_dplm import tiny


def original_functions():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'src/byprot/models/utils.py').read_text())
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef)
                and x.name in ('top_k_top_p_filtering', 'topk_masking')]
    tree2 = ast.parse((root/'src/byprot/models/dplm2/dplm2.py').read_text())
    selected += [f for c in tree2.body if isinstance(c, ast.ClassDef) for f in c.body
                 if isinstance(f, ast.FunctionDef) and f.name == '_reparam_decoding']
    scope = dict(torch=torch, np=np, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'released_reference', 'exec'), scope)
    return scope


def records(n=7):
    return [dict(aa=np.arange(n, dtype=np.uint8)%20, fast_id=np.arange(n, dtype=np.uint16),
                 residue_id=np.arange(3, 3+n), chain_end=np.arange(n) == n-1)]


def test_nucleus_matches_actual_released_filter_before_temperature():
    logits = torch.randn(2, 8, 43, generator=torch.Generator().manual_seed(7))
    expected = original_functions()['top_k_top_p_filtering'](logits.clone(), top_p=.95)
    torch.testing.assert_close(nucleus(logits, .95), expected, rtol=0, atol=0)
    assert (nucleus(torch.tensor([[[10., 0., -5.]]]), .95).isfinite().sum(-1) == 1).all()


def test_fresh_transition_matches_released_unconditional_reparameterization():
    old = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    proposal = old+20
    score = torch.tensor([[-8., -9., -3., -.4, -.5, -7., -6., -5.]])
    previous = torch.tensor([[True, False, True, False, False, True, False, True]])
    real = torch.ones_like(previous)
    previous_scores = torch.zeros_like(score)
    obj = SimpleNamespace(aa_type=1, struct_type=0, aa_mask_id=-1, struct_mask_id=-2)
    mask, token, _ = original_functions()['_reparam_decoding'](obj, old.clone(), previous_scores.clone(),
        proposal, score, 'reparam-uncond-deterministic-linear', previous,
        torch.ones_like(old), real, t=1, max_step=2)
    ours, ours_mask, _ = transition(old, proposal, previous, score, real, .5,
                                    torch.Generator(), stochastic=False)
    assert torch.equal(mask, ours_mask)
    assert torch.equal(token[~mask], ours[~ours_mask])  # Stale IDs under a mask are deliberately hidden.
    stale, stale_mask, _ = transition(old, proposal, previous, score, real, .5,
        torch.Generator(), stochastic=False, previous_scores=previous_scores)
    assert not torch.equal(stale_mask, mask)


def test_legacy_diagnostic_exactly_replays_existing_sampler():
    model = tiny('index').eval()
    blank = records(7)
    blank[0]['aa'][:] = 0
    blank[0]['fast_id'][:] = 0
    actual = generate_diagnostic(model, blank, steps=5, seed=12, policy='legacy')[0]
    expected = generate(model, 7, steps=5, seed=12)[0]
    for key in ('aa', 'fast_id', 'chain_end'):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('condition', ['aa', 'fast'])
@pytest.mark.parametrize('policy', ['legacy', 'fresh_scores', 'release_cool', 'release_default'])
def test_conditioning_is_preserved_and_targets_hidden(condition, policy):
    model = tiny('index').eval()
    inputs = records()
    out = generate_diagnostic(model, inputs, steps=4, seed=3, policy=policy, condition=condition)[0]
    key = 'aa' if condition == 'aa' else 'fast_id'
    np.testing.assert_array_equal(out[key], inputs[0][key])
    changed = [{k: v.copy() for k, v in inputs[0].items()}]
    hidden = 'fast_id' if condition == 'aa' else 'aa'
    changed[0][hidden][:] = 19
    again = generate_diagnostic(model, changed, steps=4, seed=3, policy=policy, condition=condition)[0]
    np.testing.assert_array_equal(out[hidden], again[hidden])
