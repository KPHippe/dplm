import copy
import json

import numpy as np
import pytest

from fast_dplm.data import sha256
from fast_dplm.refolding import MODEL, SETTINGS, FoldCache, fold_request, parse_fold
from evaluate_fast_samples_esmfold2 import evaluate
from run_fast_esmfold2_panel import verify_bundle


def response(sequence='AAAA'):
    xyz = np.full((len(sequence), 37, 3), None, dtype=object)
    xyz[:, 1] = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])[:len(sequence)]
    return dict(model=MODEL, coordinates=xyz.tolist(), plddt=[.8]*len(sequence), ptm=.7,
                created='test', tokens_used=20)


def test_request_is_versioned_and_sequence_only():
    request = fold_request('AAAA')
    assert request == dict(SETTINGS, sequence='AAAA')
    assert request['model'] == 'esmfold2-2026-05'
    assert not {'coordinates', 'msa', 'structure', 'templates'} & request.keys()
    for invalid in ('AA|AA', 'AXA', 'aaA', 'AA', 'A'*769):
        with pytest.raises(ValueError):
            fold_request(invalid)


def test_parse_preserves_missing_atoms_and_scales_residue_confidence():
    xyz, mask, confidence = parse_fold(response(), 'AAAA')
    assert xyz.shape == (4, 37, 3) and mask.sum() == 4
    assert mask[:, 1].all() and np.isnan(xyz[:, 0]).all()
    np.testing.assert_allclose(confidence, 80.)


@pytest.mark.parametrize('change', ['model', 'length', 'ca', 'partial', 'plddt_scale',
                                    'plddt_nan', 'plddt_shape', 'sequence', 'order', 'chains'])
def test_invalid_response_fails_closed(change):
    r = response()
    if change == 'model':
        r['model'] = 'esmfold2-fast-2026-05'
    elif change == 'length':
        r['coordinates'].pop()
    elif change == 'ca':
        r['coordinates'][0][1] = [None]*3
    elif change == 'partial':
        r['coordinates'][0][0] = [None, 1., 2.]
    elif change == 'plddt_scale':
        r['plddt'][0] = 80.
    elif change == 'plddt_nan':
        r['plddt'][0] = float('nan')
    elif change == 'plddt_shape':
        r['plddt'] = [[.8]*37]*4
    elif change == 'sequence':
        r['sequence'] = 'RRRR'
    elif change == 'order':
        r['residue_index'] = [0, 2, 1, 3]
    elif change == 'chains':
        r['entity_id'] = [0, 0, 1, 1]
    with pytest.raises(ValueError):
        parse_fold(r, 'AAAA')


def test_cache_reuses_prediction_and_rejects_tampering(tmp_path, monkeypatch):
    cache = FoldCache(tmp_path)
    calls = []
    monkeypatch.setattr(cache, '_post', lambda request: calls.append(request) or response())
    first, receipt = cache.get('AAAA')
    second, again = cache.get('AAAA')
    assert first == second and receipt == again and len(calls) == 1
    path = tmp_path/receipt['cache_key']/'response.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        cache.get('AAAA')
    assert len(calls) == 1  # Corruption must never silently trigger a new fold.


def make_decoded(root, count=1, mismatch=False):
    root.mkdir()
    records = []
    for i in range(count):
        sample = root/f'sample_{i:04d}'
        sample.mkdir()
        xyz, mask, _ = parse_fold(response(), 'AAAA')
        # Different global pose still has zero proper-rotation RMSD.
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        xyz = xyz @ rotation + 7.
        np.savez(sample/'structure.npz', xyz=xyz, atom_mask=mask,
                 aa=np.full(4, 1 if mismatch else 0), residue_id=np.arange(1, 5), chain_id=np.zeros(4, dtype=int))
        (sample/'sequence_dplm.fasta').write_text('>generated\nAAAA\n')
        record = dict(name=sample.name, length=4, aa_agreement=0. if mismatch else 1.,
                      exact_sequence_agreement=not mismatch, structure_sha256=sha256(sample/'structure.npz'))
        (sample/'diagnostics.json').write_text(json.dumps(record))
        records.append(record)
    (root/'report.json').write_text(json.dumps(records))
    return root


def test_two_renderers_share_fold_but_keep_identity_and_chemistry_separate(tmp_path, monkeypatch):
    cache = FoldCache(tmp_path/'cache')
    calls = []
    monkeypatch.setattr(cache, '_post', lambda request: calls.append(request) or response())
    good = make_decoded(tmp_path/'good')
    mismatch = make_decoded(tmp_path/'mismatch', mismatch=True)
    before = sha256(mismatch/'sample_0000/structure.npz')
    a = evaluate(good, tmp_path/'out_a', cache)['rows'][0]
    b = evaluate(mismatch, tmp_path/'out_b', cache)['rows'][0]
    assert len(calls) == 1 and a['fold_receipt'] == b['fold_receipt']
    assert a['refold_ca_rmsd_a'] < 1e-12 and b['refold_ca_rmsd_a'] < 1e-12
    assert a['identity_and_backbone_consistent'] and not b['identity_and_backbone_consistent']
    assert b['backbone_consistent_2a'] and b['aa_agreement'] == 0.
    assert b['refold_mean_plddt'] == 80. and 'chemistry' in b
    assert sha256(mismatch/'sample_0000/structure.npz') == before


def test_failed_folds_stay_in_denominator_and_retry_on_resume(tmp_path, monkeypatch):
    cache = FoldCache(tmp_path/'cache')
    decoded = make_decoded(tmp_path/'decoded', count=2)
    calls = []

    def intermittent(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError('temporary fold failure')
        return response()

    monkeypatch.setattr(cache, '_post', intermittent)
    with pytest.raises(RuntimeError, match='incomplete'):
        evaluate(decoded, tmp_path/'out', cache)
    partial = json.loads((tmp_path/'out/report.json').read_text())
    assert partial['requested'] == 2 and partial['refolded'] == 1
    assert partial['rows'][0]['status'] == 'refold_failed'
    completed = evaluate(decoded, tmp_path/'out', cache)
    assert completed['requested'] == completed['refolded'] == 2
    assert len(calls) == 2


def test_successful_cache_never_records_wrong_model(tmp_path, monkeypatch):
    cache = FoldCache(tmp_path)
    wrong = copy.deepcopy(response())
    wrong['model'] = 'esm3-open-2024-03'
    monkeypatch.setattr(cache, '_post', lambda _: wrong)
    with pytest.raises(ValueError, match='model'):
        cache.get('AAAA')
    assert not list(tmp_path.glob('*/receipt.json'))


def test_overlay_integrity_rejects_changed_file(tmp_path):
    source = tmp_path/'script.py'
    source.write_text('# frozen')
    (tmp_path/'bundle.json').write_text(json.dumps(dict(files={'script.py': sha256(source)})))
    verify_bundle(tmp_path)
    source.write_text('# modified')
    with pytest.raises(ValueError, match='frozen'):
        verify_bundle(tmp_path)
