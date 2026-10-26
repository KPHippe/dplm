import copy
import json

import numpy as np
import pytest
import torch

from fast_dplm.evaluation import evaluate_panel
from fast_dplm.generation_metrics import ca_geometry, sequence_metrics, sequence_panel, summarize_scoring
from fast_dplm.generation_validation import (
    digest_json, generate_validation, make_generation_panel, sequence_condition, validate_generation_panel,
)
from test_fast_dplm import tiny


class Cache:
    digest = 'frozen'
    sources = ['test_source']

    def __init__(self):
        self.entries = [dict(source='test_source', entry_id=str(i), split='val', cluster=str(i),
            length=n, file_sha256='file'+str(i)) for i, n in enumerate((60, 174, 288, 402))]
        self.records = [dict(aa=np.arange(e['length'], dtype=np.uint8)%20,
            fast_id=np.arange(e['length'], dtype=np.uint16)%4096,
            residue_id=np.arange(3, e['length']+3, dtype=np.int32),
            chain_end=np.arange(e['length']) == e['length']-1) for e in self.entries]

    def __getitem__(self, i):
        return {k: v.copy() for k, v in self.records[i].items()}


def test_panel_determinism_rejects_test_duplicates_and_mutated_tokens():
    cache = Cache()
    panel = make_generation_panel(cache, per_source_bin=1)
    assert panel == make_generation_panel(cache, per_source_bin=1)
    validate_generation_panel(cache, panel)
    bad = copy.deepcopy(panel)
    bad['entries'].append(bad['entries'][0])
    with pytest.raises(ValueError, match='duplicate'):
        validate_generation_panel(cache, bad)
    cache.records[0]['fast_id'][0] += 1
    with pytest.raises(ValueError, match='tokens'):
        validate_generation_panel(cache, panel)
    cache.entries[0]['split'] = 'test'
    with pytest.raises(ValueError, match='validation membership'):
        make_generation_panel(cache, per_source_bin=1)


def test_sampler_boundary_discards_tokens_and_coordinates(monkeypatch, tmp_path):
    import fast_dplm.generation_validation as module
    cache = Cache()
    panel = make_generation_panel(cache, per_source_bin=1, sampling_steps=2)
    panel['entries'] = panel['entries'][:1]
    native = cache[0] | dict(coords=np.ones((60,37,3)))
    input_record = sequence_condition(native)
    assert set(input_record) == {'aa', 'fast_id', 'residue_id', 'chain_end'}
    assert not input_record['fast_id'].any()
    def fake_sampler(model, records, **kwargs):
        assert kwargs['condition'] == 'aa'
        assert set(records[0]) == set(input_record)
        assert not records[0]['fast_id'].any()
        return records
    monkeypatch.setattr(module, 'generate_diagnostic', fake_sampler)
    model = tiny('index')
    result = generate_validation(model, cache, panel, tmp_path/'samples', dict(fingerprint='test', levels=[8]*4))
    assert result['requested'] == result['generated'] == 1
    assert (tmp_path/'samples/entry_0000/manifest.json').exists()


def test_generation_failures_remain_in_denominator(monkeypatch, tmp_path):
    import fast_dplm.generation_validation as module
    cache = Cache()
    panel = make_generation_panel(cache, per_source_bin=1)
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(module, 'generate_diagnostic', fail)
    report = generate_validation(tiny('index'), cache, panel, tmp_path/'samples', dict(fingerprint='test'))
    assert report['requested'] == 4 and report['generated'] == 0
    assert all(r['status'] == 'failed' and 'injected' in r['error'] for r in report['rows'])


def test_optional_periodic_hook_preserves_rng_and_mode_and_avoids_generation_between_milestones(tmp_path):
    torch.set_num_threads(2)
    cache = Cache()
    generation = make_generation_panel(cache, per_source_bin=1, sampling_steps=2,
        joint_samples=[dict(length=7, seed=2)])
    generation['entries'] = generation['entries'][:1]
    panel = dict(schema='fast-dplm-eval-panel-v1', cache_digest=cache.digest, entries=[], tasks=[],
        mask_seed=3, samples=generation['joint_samples'], sampling_steps=2, generation_validation=generation)
    model = tiny('index').train()
    before = torch.get_rng_state().clone()
    evaluate_panel(model, cache, panel, tmp_path/'cheap', 1, dict(fingerprint='test'), sampling=False)
    report = json.loads((tmp_path/'cheap/full_mask_denoising.json').read_text())
    assert {r['task'] for r in report['rows']} == set(generation['full_mask_tasks'])
    assert all(r['residues'] == 60 for r in report['rows'])
    assert report['summary']['structure_without_sequence']['softmax_domain'] == 'FAST structure vocabulary only'
    assert 'shared AA/FAST' in report['summary']['sequence_without_structure']['softmax_domain']
    assert not (tmp_path/'cheap/conditional_samples').exists()
    assert model.training and torch.equal(before, torch.get_rng_state())
    evaluate_panel(model, cache, panel, tmp_path/'sparse', 2, dict(fingerprint='test'), sampling=True)
    assert (tmp_path/'sparse/conditional_samples/manifest.json').exists()
    assert (tmp_path/'sparse/sequence_metrics.json').exists()
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert report['panel_sha256'] == digest_json(generation)
    evaluate_panel(model, cache, panel, tmp_path/'sparse', 2, dict(fingerprint='test'), sampling=True)
    sample = tmp_path/'sparse/conditional_samples/entry_0000/sample_0000.npz'
    sample.write_bytes(sample.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='integrity'):
        evaluate_panel(model, cache, panel, tmp_path/'sparse', 2, dict(fingerprint='test'), sampling=True)


def test_sequence_metrics_short_constant_and_diversity_denominators():
    assert sequence_metrics(np.array([1,2,3]))['repeat4_fraction'] is None
    row = sequence_metrics(np.zeros(8, dtype=int))
    assert row['repeat4_fraction'] == pytest.approx(.8)
    assert row['longest_identical_run'] == 8 and row['aa_entropy_bits'] == 0
    report = sequence_panel([dict(row, status='ok'), dict(row, status='ok')], 3)
    assert report['unique_sequences'] == 1 and report['failed_or_missing'] == 1
    assert report['unique_fraction_scored'] == .5
    with pytest.raises(ValueError):
        sequence_metrics(np.array([20]))


def test_ca_metrics_independent_pair_loop_proper_rotation_and_reflection():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(9, 3))*4
    mobile = reference + rng.normal(size=reference.shape)*.4
    values = []
    for i in range(len(reference)):
        for j in range(i+1, len(reference)):
            if np.linalg.norm(reference[i]-reference[j]) < 15:
                error = abs(np.linalg.norm(reference[i]-reference[j])-np.linalg.norm(mobile[i]-mobile[j]))
                values.append(sum(error < t for t in (.5,1,2,4))/4)
    scored = ca_geometry(mobile, reference)
    assert scored['ca_lddt'] == pytest.approx(np.mean(values))
    q, _ = np.linalg.qr(rng.normal(size=(3,3)))
    q[:, 0] *= np.linalg.det(q)
    moved = ca_geometry(mobile@q+np.array([5,7,-3]), reference)
    assert moved['ca_rmsd_a'] == pytest.approx(scored['ca_rmsd_a'])
    assert moved['ca_lddt'] == scored['ca_lddt']
    reflected = reference.copy()
    reflected[:, 0] *= -1
    assert ca_geometry(reflected, reference)['ca_rmsd_a'] > 1
    assert ca_geometry(reflected, reference)['ca_lddt'] == 1
    with pytest.raises(ValueError):
        ca_geometry(np.full((4,3), np.nan), np.zeros((4,3)))


def test_scoring_separates_identity_backbone_chemistry_and_counts_failures():
    chem = dict(emitted_atoms=10, peptide_bonds=2, peptide_outside_1p1_1p6_a=1, severe_clash_pairs=20)
    rows = [dict(id='a', status='ok', exact_identity=True, backbone_below_2a=True,
                 ca_rmsd_a=1., ca_lddt=.9, chemistry=chem), dict(id='b', status='failed')]
    result = summarize_scoring(rows, 2)
    assert result['identity_and_backbone']['fraction'] == .5
    assert result['failed'] == 1 and result['pooled_peptide_violation_fraction'] == .5
    assert result['chemistry_gate'] is result['full_atom_readiness'] is None
    unscored = summarize_scoring(rows, 2, fold_requested=False)
    assert unscored['backbone_below_2a'] is None and unscored['fold_status'] == 'not_requested'
    with pytest.raises(ValueError, match='explicit result'):
        summarize_scoring(rows[:1], 2)
