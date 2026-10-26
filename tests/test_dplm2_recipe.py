import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from fast_dplm.data import RESIDUE, sha256
from fast_dplm.model import loss_sums, noise_plan
from fast_dplm.provenance import decoder_provenance
from fast_dplm.recipe import DPLM2Recipe, LoRALinear, model_from_spec
from fast_dplm.sampling import generate
from test_fast_dplm import VOCAB, inputs, tiny


def recipe(rank=2):
    return DPLM2Recipe(tiny().net.config.to_dict(), VOCAB, lora_rank=rank)


def test_zero_initialized_lora_and_frozen_trunk_update_boundary():
    base = torch.nn.Linear(5, 7)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(4, 5)
    torch.testing.assert_close(base(x), layer(x), rtol=0, atol=0)
    model = recipe().train()
    batch, plan = inputs(model)
    before = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}
    trainable = {k: v for k, v in model.named_parameters() if v.requires_grad}
    assert all('.a.weight' in k or '.b.weight' in k or '.embeddings.' in k or '.lm_head.' in k for k in trainable)
    optim = torch.optim.AdamW(trainable.values(), lr=1e-3)
    sums, count = loss_sums(model, model(batch, plan['struct_mask'], plan['aa_mask']), batch, plan)
    (.5*(sums/count.clamp_min(1)).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    optim.step()
    assert any('.b.weight' in k and p.abs().sum() > 0 for k, p in trainable.items())
    for k, p in model.named_parameters():
        if k in before:
            assert p.grad is None
            torch.testing.assert_close(p, before[k], rtol=0, atol=0)


def test_recipe_matches_released_native_forward_and_trainable_gradients(monkeypatch):
    # Import the actual repository module with only its registration decorator stubbed.
    # This exercises the original embedding, LM head and attention implementation.
    registry = types.ModuleType('byprot.models')
    registry.register_model = lambda name: lambda cls: cls
    monkeypatch.setitem(sys.modules, 'byprot.models', registry)
    path = Path(__file__).resolve().parents[1]/'src/byprot/models/dplm2/modules/dplm2_modeling_esm.py'
    spec = importlib.util.spec_from_file_location('recipe_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = recipe(rank=0).eval()
    ref = module.EsmForDPLM2(model.net.config, dropout=.1, vocab_size=model.net.config.vocab_size).eval()
    ref.esm.contact_head = None
    ref.load_state_dict(model.net.state_dict(), strict=True)
    ref.requires_grad_(False)
    ref.esm.embeddings.word_embeddings.requires_grad_(True)
    ref.lm_head.requires_grad_(True)
    batch, plan = inputs(model)
    independent = torch.tensor([False, True])
    actual = model(batch, plan['struct_mask'], plan['aa_mask'], independent)
    ids = model.input_ids(batch, plan['struct_mask'], plan['aa_mask'])
    valid = ids.ne(1)
    n = ids.shape[1]//2
    mask = torch.zeros(len(ids), model.net.config.num_attention_heads, 2*n, 2*n)
    mask.masked_fill_(~valid[:, None, None, :], -torch.inf)
    mask[independent, :, :n, n:] = -torch.inf
    mask[independent, :, n:, :n] = -torch.inf
    type_ids = ((ids < 33) & valid).int().masked_fill(~valid, 2)
    expected = ref(input_ids=ids, inputs_embeds=ref.esm.embeddings(ids, attention_mask=valid),
                   attention_mask=mask, type_ids=type_ids)['logits']
    actual_logits = torch.cat([actual['struct_loss_logits'], actual['aa']], 1)
    torch.testing.assert_close(actual_logits, expected, atol=2e-6, rtol=2e-5)
    actual_logits.square().mean().backward()
    expected.square().mean().backward()
    for name, param in model.net.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, dict(ref.named_parameters())[name].grad, atol=2e-6, rtol=2e-5)


def test_pretrained_rows_preserved_and_strict_recipe_reload(tmp_path):
    parent = tiny('index').net
    parent.config.save_pretrained(tmp_path)
    (tmp_path/'vocab.txt').write_text('\n'.join(VOCAB)+'\n')
    torch.save(parent.state_dict(), tmp_path/'pytorch_model.bin')
    model = DPLM2Recipe.from_dplm(tmp_path, lora_rank=2).eval()
    torch.testing.assert_close(model.net.esm.embeddings.word_embeddings.weight[:33],
                               parent.esm.embeddings.word_embeddings.weight, atol=0, rtol=0)
    torch.testing.assert_close(model.net.lm_head.decoder.weight[:33], parent.lm_head.decoder.weight, atol=0, rtol=0)
    clone = model_from_spec(model.specification())
    clone.load_state_dict(model.state_dict(), strict=True)
    for a, b in zip(generate(model, 5, steps=3), generate(clone, 5, steps=3), strict=True):
        for key in a:
            assert (a[key] == b[key]).all()


def test_shared_vocabulary_loss_and_balanced_tasks():
    model = recipe()
    batch, _ = inputs(model)
    batch = {k: v.repeat(4, 1) for k, v in batch.items()}
    q = noise_plan(batch, torch.Generator().manual_seed(5), balanced=True)
    assert torch.bincount(q['task']).tolist() == [2]*4
    assert not q['aa_mask'][q['task'] == 1].any()
    assert not q['struct_mask'][q['task'] == 2].any()
    assert torch.equal(q['sw'][q['task']==0], q['aw'][q['task']==0])
    out = model(batch, q['struct_mask'], q['aa_mask'], q['independent'])
    sums, _ = loss_sums(model, out, batch, q)
    expected = torch.zeros(2)
    for row in range(len(batch['state'])):
        for col in range(batch['state'].shape[1]):
            if q['struct_mask'][row,col]:
                expected[0] -= out['struct_loss_logits'][row,col].log_softmax(-1)[33+batch['fast'][row,col]]*q['sw'][row]
            if q['aa_mask'][row,col]:
                expected[1] -= out['aa'][row,col].log_softmax(-1)[batch['aa'][row,col]]*q['aw'][row]
    torch.testing.assert_close(sums, expected)


def test_recipe_masks_hide_targets_and_independent_attention_blocks_content():
    model = recipe().eval()
    batch, _ = inputs(model)
    real = batch['state']==RESIDUE
    before = model(batch, real, real)
    changed = {k: v.clone() for k,v in batch.items()}
    changed['fast'][real] = 4095
    changed['aa'][real] = model.aa_map[19]
    after = model(changed, real, real)
    torch.testing.assert_close(before['struct_loss_logits'], after['struct_loss_logits'], atol=0, rtol=0)
    torch.testing.assert_close(before['aa'], after['aa'], atol=0, rtol=0)
    zero = torch.zeros_like(real)
    independent = torch.ones(len(real), dtype=torch.bool)
    a = model(batch, real, zero, independent)
    b = model(changed, real, zero, independent)
    torch.testing.assert_close(a['struct_loss_logits'], b['struct_loss_logits'], atol=0, rtol=0)


def test_decoder_lineage_rejects_wrong_parent_and_mutated_checkpoint(tmp_path):
    (tmp_path/'weights').write_text('weights')
    (tmp_path/'config').write_text('config')
    manifest = dict(fingerprint='parent', checkpoint_sha256='parent-sha', levels=[8]*4)
    lineage = dict(schema='fast-dplm-decoder-lineage-v1', cache_fingerprint='parent',
        parent_checkpoint_sha256='parent-sha', levels=[8]*4, fingerprint='new',
        checkpoint=str(tmp_path/'weights'), checkpoint_sha256=sha256(tmp_path/'weights'),
        config=str(tmp_path/'config'), config_sha256=sha256(tmp_path/'config'))
    path = tmp_path/'lineage.json'
    path.write_text(json.dumps(lineage))
    assert decoder_provenance(manifest, path)['fingerprint'] == 'new'
    with pytest.raises(ValueError, match='match token cache'):
        decoder_provenance(manifest | dict(fingerprint='different'), path)
    (tmp_path/'weights').write_text('changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        decoder_provenance(manifest, path)
