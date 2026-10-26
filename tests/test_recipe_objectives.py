import ast
import types
from pathlib import Path

import pytest
import torch

from fast_dplm.data import BOS, EOS, MASK, RESIDUE
from fast_dplm.model import loss_sums, noise_plan
from fast_dplm.objectives import loss_plan, released_mixup_input
from test_dplm2_recipe import recipe
from test_fast_dplm import inputs


def example():
    model = recipe().eval()
    batch, _ = inputs(model)
    batch = {k: v.repeat(4, 1) for k, v in batch.items()}
    noising = noise_plan(batch, torch.Generator().manual_seed(1234), balanced=True)
    return model, batch, noising


def native_mixup_reference(model, batch):
    """Extract actual released methods without importing its legacy dependencies."""
    path = Path(__file__).resolve().parents[1]/'src/byprot/models/dplm2/dplm2.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'MultimodalDiffusionProteinLanguageModel')
    names = {'self_mixup', 'get_mixup_xt', 'get_non_special_symbol_mask', 'get_modality_type'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    obj = types.SimpleNamespace(pad_id=model.special['pad'], aa_mask_id=model.special['mask'],
        struct_mask_id=model.struct_special[MASK], aa_bos_id=model.special['bos'],
        aa_eos_id=model.special['eos'], struct_bos_id=model.struct_special[BOS],
        struct_eos_id=model.struct_special[EOS], aa_type=1, struct_type=0, pad_type=2)
    for name in names:
        setattr(obj, name, types.MethodType(namespace[name], obj))
    empty = torch.zeros_like(batch['state'], dtype=torch.bool)
    def forward(input_ids, single_modality):
        out = model(batch, empty, empty, single_modality, token_input_ids=input_ids)
        return {'logits': torch.cat([out['struct_loss_logits'], out['aa']], dim=1)}
    obj.forward = forward
    return obj


def test_default_objective_is_exact_and_does_not_mutate_noising():
    model, batch, noising = example()
    plan = loss_plan(batch, noising)
    for key in plan:
        torch.testing.assert_close(plan[key], noising[key], atol=0, rtol=0)
        assert plan[key].data_ptr() != noising[key].data_ptr()
    output = model(batch, noising['struct_mask'], noising['aa_mask'], noising['independent'])
    for a, b in zip(loss_sums(model, output, batch, plan), loss_sums(model, output, batch, noising)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_released_mixup_matches_native_input_logits_masks_and_gradients():
    model, batch, noising = example()
    reference = native_mixup_reference(model, batch)
    ids = model.input_ids(batch, noising['struct_mask'], noising['aa_mask'])
    captured = []
    original = reference.forward
    def capture(**kwargs):
        captured.append(kwargs['input_ids'].clone())
        return original(**kwargs)
    reference.forward = capture
    expected, mask = reference.self_mixup(ids, noising['independent'])
    actual_ids = released_mixup_input(model, batch, noising)
    assert not actual_ids.requires_grad
    torch.testing.assert_close(actual_ids, captured[1], rtol=0, atol=0)
    actual = model(batch, noising['struct_mask'], noising['aa_mask'], noising['independent'],
                   token_input_ids=actual_ids)
    logits = torch.cat([actual['struct_loss_logits'], actual['aa']], dim=1)
    torch.testing.assert_close(logits, expected['logits'], rtol=0, atol=0)
    objective = loss_plan(batch, noising, self_mixup='released')
    torch.testing.assert_close(torch.cat([objective['struct_mask'], objective['aa_mask']], dim=1), mask)
    param = model.net.lm_head.decoder.weight
    actual_grad, = torch.autograd.grad(logits.square().mean(), param)
    ref_grad, = torch.autograd.grad(expected['logits'].square().mean(), param)
    torch.testing.assert_close(actual_grad, ref_grad, atol=0, rtol=0)


def test_proposal_has_no_grad_and_fully_masked_input_hides_targets():
    model, batch, noising = example()
    noising['struct_mask'] = batch['state'] == RESIDUE
    noising['aa_mask'] = noising['struct_mask'].clone()
    modes = []
    handle = model.register_forward_pre_hook(lambda *_: modes.append(torch.is_grad_enabled()))
    first = released_mixup_input(model, batch, noising)
    changed = {k: v.clone() for k, v in batch.items()}
    changed['fast'][noising['struct_mask']] = 4095
    changed['aa'][noising['aa_mask']] = model.aa_map[-1]
    second = released_mixup_input(model, changed, noising)
    handle.remove()
    assert modes == [False, False]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    real = torch.cat([batch['state'] == RESIDUE]*2, dim=1)
    original = model.input_ids(batch, noising['struct_mask'], noising['aa_mask'])
    torch.testing.assert_close(first[~real], original[~real], atol=0, rtol=0)


@pytest.mark.parametrize('mixup', ['off', 'released'])
def test_constant_weights_match_independent_loop_and_preserve_task_mass(mixup):
    _, batch, noising = example()
    before = loss_plan(batch, noising, self_mixup=mixup)
    actual = loss_plan(batch, noising, self_mixup=mixup, noise_weighting='constant_mass')
    for mask, weight in [('struct_mask', 'sw'), ('aa_mask', 'aw')]:
        for task in range(4):
            rows = [i for i in range(len(batch['state'])) if int(noising['task'][i]) == task]
            values = [float(before[weight][i]) for i in rows for j in range(batch['state'].shape[1])
                      if before[mask][i,j]]
            if not values:
                continue
            expected = sum(values)/len(values)
            assert all(float(actual[weight][i]) == pytest.approx(expected) for i in rows)
            old_mass = sum(float(before[weight][i]) for i in rows for j in range(batch['state'].shape[1])
                           if before[mask][i,j])
            new_mass = sum(float(actual[weight][i]) for i in rows for j in range(batch['state'].shape[1])
                           if actual[mask][i,j])
            assert new_mass == pytest.approx(old_mass, rel=1e-6)
    # Native conditional t=0 track weights are preserved under all-token mixup.
    if mixup == 'released':
        for task, weight in [(1, 'aw'), (2, 'sw')]:
            torch.testing.assert_close(actual[weight][noising['task']==task], before[weight][noising['task']==task])


def test_global_objective_loss_and_gradients_equal_sliced_accumulation():
    model, batch, noising = example()
    objective = loss_plan(batch, noising, self_mixup='released', noise_weighting='constant_mass')
    # Independent logits test the global loss reduction, avoiding model dropout/RNG.
    b, n = batch['state'].shape
    vocab = model.net.config.vocab_size
    sl = torch.randn(b, n, vocab, requires_grad=True)
    al = torch.randn(b, n, vocab, requires_grad=True)
    def output(s, a):
        return dict(struct=[s[..., 33:4129]], struct_loss_logits=s, struct_offset=33, aa=a)
    sums, denominator = loss_sums(model, output(sl, al), batch, objective)
    full = .5*(sums/denominator).sum()
    accumulated = 0.
    for lo, hi in [(0,3),(3,5),(5,8)]:
        mb = {k:v[lo:hi] for k,v in batch.items()}
        mp = {k:v[lo:hi] for k,v in objective.items()}
        numer, _ = loss_sums(model, output(sl[lo:hi], al[lo:hi]), mb, mp)
        accumulated = accumulated + .5*(numer/denominator).sum()
    torch.testing.assert_close(full, accumulated)
    a = torch.autograd.grad(full, (sl,al), retain_graph=True)
    b = torch.autograd.grad(accumulated, (sl,al))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)


def test_empty_supervision_is_finite_and_bad_options_fail():
    model, batch, noising = example()
    noising['struct_mask'].zero_()
    noising['aa_mask'].zero_()
    plan = loss_plan(batch, noising, noise_weighting='constant_mass')
    output = model(batch, plan['struct_mask'], plan['aa_mask'], plan['independent'])
    sums, counts = loss_sums(model, output, batch, plan)
    assert torch.equal(counts, torch.zeros(2,dtype=counts.dtype))
    assert torch.isfinite(sums).all() and sums.sum()==0
    sums.sum().backward()
    with pytest.raises(ValueError, match='self-mixup'):
        loss_plan(batch, noising, self_mixup='unknown')
    with pytest.raises(ValueError, match='noise weighting'):
        loss_plan(batch, noising, noise_weighting='unknown')


def test_no_grad_proposal_preserves_trainable_gradients_under_bfloat16_autocast():
    # The two forwards share an autocast region in training. Ensure a no-grad
    # proposal cannot cause cached casts to detach the gradient-bearing pass.
    model, batch, noising = example()
    objective = loss_plan(batch, noising, self_mixup='released', noise_weighting='constant_mass')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        ids = released_mixup_input(model, batch, noising)
        out = model(batch, noising['struct_mask'], noising['aa_mask'], noising['independent'],
                    token_input_ids=ids)
        sums, counts = loss_sums(model, out, batch, objective)
        loss = .5*(sums/counts).sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
