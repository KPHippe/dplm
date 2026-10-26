import numpy as np
import pytest
import torch
from transformers import EsmConfig

from fast_dplm.data import RESIDUE, TokenCache, collate
from fast_dplm.model import FastDPLM, loss_sums, noise_plan
from fast_dplm.sampling import generate, transition

VOCAB = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E',
         'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W',
         'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']


def tiny(mode="components"):
    cfg = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=4, intermediate_size=64, pad_token_id=1,
                    mask_token_id=32, position_embedding_type="rotary", token_dropout=True,
                    layer_norm_eps=1e-5,  # Match the actual pretrained DPLM config.
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return FastDPLM(cfg, VOCAB, mode=mode)


def records():
    return [dict(aa=np.arange(n) % 20, fast_id=np.arange(n, dtype=np.int64),
                 residue_id=np.arange(n) + 12, chain_end=np.arange(n) == n - 1)
            for n in (3, 5)]


def inputs(model):
    batch = collate(records(), model.aa_map, model.special)
    plan = noise_plan(batch, torch.Generator().manual_seed(44))
    return batch, plan


def test_exhaustive_tuple_conversion_matches_fast():
    from fast.models.bottlenecks.fsp import FSP
    model = tiny()
    ids = torch.arange(4096)
    component = model.components(ids)
    assert torch.equal(model.pack(component), ids)
    fsp = FSP([8]*4)
    assert torch.equal(component, fsp.indices_to_level_indices(ids))
    assert torch.equal(model.pack(component), fsp.level_indices_to_indices(component).long())
    with pytest.raises(ValueError):
        model.pack(torch.tensor([[8, 0, 0, 0]]))


@pytest.mark.parametrize("mode", ["index", "components"])
def test_masked_input_does_not_read_target_and_independent_blocks_cross_attention(mode):
    model = tiny(mode).eval()
    batch, _ = inputs(model)
    real = batch["state"] == RESIDUE
    zero = torch.zeros_like(real)
    out = model(batch, real, zero)
    altered = {k: v.clone() for k, v in batch.items()}
    altered["fast"][real] = 4095
    other = model(altered, real, zero)
    torch.testing.assert_close(out["struct"][0], other["struct"][0], rtol=0, atol=0)
    independent = torch.ones(len(real), dtype=torch.bool)
    before = model(batch, zero, zero, independent)["struct"][0]
    altered["aa"][real] = model.aa_map[19]
    altered["fast"] = batch["fast"]
    after = model(altered, zero, zero, independent)["struct"][0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    with pytest.raises(ValueError):
        altered["fast"][real] = 4096
        model(altered, real, zero)


@pytest.mark.parametrize("mode", ["index", "components"])
def test_loss_matches_explicit_loop_values_and_gradients_including_zero_one(mode):
    model = tiny(mode)
    batch, plan = inputs(model)
    real = batch["state"] == RESIDUE
    plan["struct_mask"], plan["aa_mask"] = real, real
    classes = [8]*4 if mode == "components" else [4096]
    output = {"struct": [torch.randn(*real.shape, n, requires_grad=True) for n in classes],
              "aa": torch.randn(*real.shape, 33, requires_grad=True)}
    sums, counts = loss_sums(model, output, batch, plan)
    expected = [torch.tensor(0.), torch.tensor(0.)]
    for b in range(len(real)):
        for i in range(real.shape[1]):
            if not real[b, i]:
                continue
            token = int(batch["fast"][b, i])
            labels = [(token // (8**j)) % 8 for j in range(4)] if mode == "components" else [token]
            for logits, label in zip(output["struct"], labels, strict=True):
                expected[0] = expected[0] - logits[b, i].log_softmax(-1)[label] * plan["sw"][b]
            expected[1] = expected[1] - output["aa"][b, i].log_softmax(-1)[batch["aa"][b, i]] * plan["aw"][b]
    torch.testing.assert_close(sums, torch.stack(expected))
    tensors = [*output["struct"], output["aa"]]
    actual_grad = torch.autograd.grad(sums.sum(), tensors, retain_graph=True)
    ref_grad = torch.autograd.grad(sum(expected), tensors)
    for a, b in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b)
    assert counts.tolist() == [8, 8]
    plan["struct_mask"] = torch.zeros_like(real)
    plan["aa_mask"] = torch.zeros_like(real)
    empty, count = loss_sums(model, output, batch, plan)
    assert empty.tolist() == [0, 0] and count.tolist() == [0, 0]
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(empty.sum(), tensors))


def test_global_normalization_matches_microbatch_gradient():
    model = tiny()
    batch, plan = inputs(model)
    plan["struct_mask"] = batch["state"] == RESIDUE
    out = model(batch, plan["struct_mask"], plan["aa_mask"], plan["independent"])
    sums, counts = loss_sums(model, out, batch, plan)
    denom = counts.clamp_min(1)
    (0.5*(sums/denom).sum()).backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    for i in range(2):
        b = {k: v[i:i+1] for k, v in batch.items()}
        q = {k: v[i:i+1] for k, v in plan.items()}
        out = model(b, q["struct_mask"], q["aa_mask"], q["independent"])
        part, _ = loss_sums(model, out, b, q)
        (0.5*(part/denom).sum()).backward()
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=1e-6, rtol=1e-4)


def test_paired_attention_matches_two_independent_hf_rotary_tracks():
    from transformers.models.esm.modeling_esm import EsmSelfAttention
    model = tiny().eval()
    paired = model.net.esm.encoder.layer[0].attention.self
    reference = EsmSelfAttention(model.net.config).eval()
    reference.load_state_dict(paired.state_dict(), strict=True)
    x = torch.randn(2, 10, 32, requires_grad=True)
    mask = torch.full((2, 1, 10, 10), -torch.inf)
    mask[:, :, :5, :5] = 0
    mask[:, :, 5:, 5:] = 0
    actual = paired(x, attention_mask=mask)[0]
    expected = torch.cat([reference(x[:, :5])[0], reference(x[:, 5:])[0]], 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["index", "components"])
def test_hidden_aa_and_padding_do_not_change_valid_predictions(mode):
    model = tiny(mode).eval()
    batch, _ = inputs(model)
    real = batch["state"] == RESIDUE
    zero = torch.zeros_like(real)
    before = model(batch, zero, real)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["aa"][real] = model.aa_map[19]
    after = model(changed, zero, real)
    torch.testing.assert_close(before["aa"], after["aa"], rtol=0, atol=0)
    # Removing only padding must not change either track's real predictions.
    short = {k: v[:1, :5] for k, v in batch.items()}
    alone = model(short, zero[:1, :5], real[:1, :5])
    for a, b in zip([*before["struct"], before["aa"]],
                    [*alone["struct"], alone["aa"]], strict=True):
        torch.testing.assert_close(a[:1, :5], b, atol=1e-6, rtol=1e-5)


def test_transition_does_not_replace_accepted_tokens_and_completes_last_step():
    g = torch.Generator().manual_seed(3)
    old, candidate = torch.tensor([[2, 3, 4]]), torch.tensor([[8, 8, 8]])
    prev = torch.tensor([[False, True, False]])
    result, mask, _ = transition(old, candidate, prev, torch.zeros(1, 3),
                               torch.ones(1, 3, dtype=torch.bool), 0., g)
    assert result.tolist() == [[2, 8, 4]] and not mask.any()


def test_remasking_uses_confidence_of_the_retained_token():
    old = torch.tensor([[2, 0]])
    candidate = torch.tensor([[7, 8]])
    previous = torch.tensor([[False, True]])
    result, mask, score = transition(old, candidate, previous, torch.tensor([[-100., -2.]]),
        torch.ones_like(previous), 0.5, torch.Generator(), stochastic=False,
        previous_scores=torch.tensor([[-0.1, -torch.inf]]))
    assert result.tolist() == [[2, 0]] and mask.tolist() == [[False, True]]
    torch.testing.assert_close(score[0, 0], torch.tensor(-0.1))
    assert torch.isneginf(score[0, 1])


@pytest.mark.parametrize("mode", ["index", "components"])
def test_generation_deterministic_valid_and_strict_reload(mode):
    model = tiny(mode).eval()
    sample = generate(model, 6, count=2, steps=4, seed=5)
    clone = FastDPLM(**model.specification())
    clone.load_state_dict(model.state_dict(), strict=True)
    again = generate(clone, 6, count=2, steps=4, seed=5)
    for a, b in zip(sample, again, strict=True):
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        assert a["fast_id"].max() < 4096 and a["aa"].max() < 20
        assert a["chain_end"].sum() == 1 and a["chain_end"][-1]


def test_cache_integrity_sampling_and_resume_address(tmp_path):
    from fast.dplm_bridge import write_cache
    rec = []
    for i, r in enumerate(records()):
        rec.append(r | dict(entry_id=str(i), split="train", source=str(i), cluster="c"))
    root = tmp_path / "cache"
    write_cache(root, rec, dict(levels=[8]*4, fingerprint="test"))
    cache = TokenCache(root)
    assert cache.profile["mean_residues"] == [3., 5.]
    np.testing.assert_allclose(cache.probabilities, [5/8, 3/8])
    continuation = [cache.draw_update(i, 8, 2) for i in range(3, 6)]
    torch.rand(300)
    assert continuation == [TokenCache(root).draw_update(i, 8, 2) for i in range(3, 6)]
    with open(root / "aa.npy", "ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        TokenCache(root)
