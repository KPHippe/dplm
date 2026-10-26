import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from fast_dplm.generation_validation import make_generation_panel, record_hash
from fast_dplm.native import digest, remap_panel, validate_native_manifest
from fast_dplm.provenance import decoder_provenance
from fast_dplm.recipe import DPLM2Recipe
from fast_dplm.model import loss_sums
from test_fast_dplm import VOCAB, inputs, tiny
from test_generation_validation import Cache


def native_manifest():
    contract = dict(vocabulary_size=8192, checkpoint_sha256='checkpoint', config_sha256='config')
    return dict(schema='native-dplm-cache-v1', representation='dplm2_native_lfq8192',
                levels=[2]*13, fingerprint='native-lfq8k:'+digest(contract), tokenizer_contract=contract,
                checkpoint_sha256='checkpoint', config_sha256='config')


def test_native_identity_rejects_wrong_vocabulary_checkpoint_and_fast_lineage(tmp_path):
    manifest = native_manifest()
    validate_native_manifest(manifest)
    for key, value in [('levels', [8]*4), ('fingerprint', 'fast-fingerprint'), ('checkpoint_sha256', 'other')]:
        with pytest.raises(ValueError, match='identity'):
            validate_native_manifest(manifest | {key: value})
    with pytest.raises(ValueError, match='FAST decoder lineage'):
        decoder_provenance(manifest, tmp_path/'unused.json')


def test_remap_preserves_requests_and_topology_but_binds_native_tokens():
    original = Cache()
    native = copy.deepcopy(original)
    native.digest = 'native'
    for record in native.records:
        record['fast_id'] = record['fast_id']+4096
    gv = make_generation_panel(original, per_source_bin=1, joint_samples=[dict(length=100, seed=42)])
    panel = dict(cache_digest=original.digest, samples=gv['joint_samples'], generation_validation=gv)
    converted = remap_panel(panel, native, original)
    assert converted['samples'] == panel['samples']
    assert converted['generation_validation']['representation'] == 'dplm2_native_lfq8192'
    assert converted['generation_validation']['entries'][0]['record_sha256'] == record_hash(native[0])
    assert converted['generation_validation']['entries'][0]['record_sha256'] != gv['entries'][0]['record_sha256']
    assert panel['cache_digest'] == 'frozen'
    native.records[0]['aa'][0] = 9
    with pytest.raises(ValueError, match='sequence/topology'):
        remap_panel(panel, native, original)


def test_native8192_shared_vocabulary_has_finite_gradients_and_residue_bounds():
    torch.set_num_threads(2)
    model = DPLM2Recipe(tiny().net.config.to_dict(), VOCAB, levels=[2]*13, lora_rank=2)
    batch, plan = inputs(model)
    batch['fast'][plan['struct_mask']] = 8191
    output = model(batch, plan['struct_mask'], plan['aa_mask'])
    assert output['struct'][0].shape[-1] == 8192
    assert output['struct_loss_logits'].shape[-1] == 8229
    numerator, count = loss_sums(model, output, batch, plan)
    (.5*(numerator/count.clamp_min(1)).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    batch['fast'][plan['struct_mask']] = 8192
    with pytest.raises(ValueError, match='outside vocabulary'):
        model.input_ids(batch, plan['struct_mask'], plan['aa_mask'])


def test_native_scoring_failure_denominator_and_no_false_joint_fold_score():
    path = Path(__file__).parents[1]/'scripts/native_control_evaluate.py'
    spec = importlib.util.spec_from_file_location('native_scoring_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(status='ok', ca_rmsd_a=1., ca_lddt=.8), dict(status='failed')]
    summary = module.summarize(rows, 3)
    assert summary['below_2a'] == 1 and summary['success_denominator'] == 3
    assert summary['failed_or_missing'] == 2
    summary = module.summarize([dict(status='ok')], 1, fold_requested=False)
    assert summary['below_2a'] is None and summary['median_ca_rmsd_a'] is None
