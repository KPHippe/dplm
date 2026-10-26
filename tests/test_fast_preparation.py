import json

import numpy as np
import pytest
import torch

from fast_dplm.data import TokenCache, sha256
from fast_dplm.evaluation import evaluate_panel, make_panel
from fast_dplm.training import learning_rate, repair_log
from test_fast_dplm import records, tiny


def make_cache(path, selected):
    from fast.dplm_bridge import write_cache
    rows = [records()[i % 2] | dict(entry_id=str(i), source='source', split='val', cluster=str(i % 2)) for i in selected]
    write_cache(path, rows, dict(fingerprint='test', levels=[8]*4, checkpoint_sha256='parent'))


def test_shards_match_flat_cache_even_when_cluster_crosses_shards(tmp_path):
    make_cache(tmp_path/'flat', [0,1,2,3])
    root = tmp_path/'corpus'
    for name, ids in [('a',[0,1]), ('b',[2,3])]:
        make_cache(root/name, ids)
    m = dict(schema='fast-dplm-corpus-v1', fingerprint='test', levels=[8]*4, checkpoint_sha256='parent',
        entries=4, residues=16, shards=[dict(path=n, sha256=sha256(root/n/'manifest.json')) for n in ('a','b')])
    (root/'manifest.json').write_text(json.dumps(m))
    flat, sharded = TokenCache(tmp_path/'flat'), TokenCache(root)
    assert flat.profile == sharded.profile
    for step in range(5):
        assert flat.draw_update(step, 20, 3) == sharded.draw_update(step, 20, 3)
    for i in range(4):
        for k, v in flat[i].items():
            np.testing.assert_array_equal(v, sharded[i][k])
    (root/'a/manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        TokenCache(root)


def test_periodic_evaluation_preserves_rng_and_training_mode(tmp_path):
    make_cache(tmp_path/'cache', [0,1])
    cache = TokenCache(tmp_path/'cache')
    panel = make_panel(cache, per_source=2, lengths=[5], per_length=1, sampling_steps=3)
    model = tiny().train()
    before = torch.get_rng_state().clone()
    evaluate_panel(model, cache, panel, tmp_path/'evaluation', 0, dict(fingerprint='test'))
    assert model.training and torch.equal(before, torch.get_rng_state())
    report = json.loads((tmp_path/'evaluation/denoising.json').read_text())
    assert len(report['tasks']) == 4 and report['tasks'][1]['masked_counts'][1] == 0
    assert report['tasks'][2]['weighted_losses'][0] is None
    evaluate_panel(model, cache, panel, tmp_path/'evaluation', 0, dict(fingerprint='test'), sampling=False)
    assert torch.equal(before, torch.get_rng_state())


def test_schedule_is_addressable_and_reaches_declared_endpoint():
    assert learning_rate(0, 1., 2, 10) == .5
    assert learning_rate(1, 1., 2, 10) == 1.
    assert learning_rate(9, 1., 2, 10) == pytest.approx(.1)
    resumed = [learning_rate(i, 1., 2, 10) for i in range(5,10)]
    assert resumed == [learning_rate(i, 1., 2, 10) for i in range(10)][5:]


def test_restart_keeps_only_committed_log_rows_and_archives_tail(tmp_path):
    path = tmp_path/'metrics.jsonl'
    path.write_text(''.join(json.dumps(dict(step=i))+'\n' for i in range(1,6)))
    repair_log(path, 3)
    assert [json.loads(x)['step'] for x in path.read_text().splitlines()] == [1,2,3]
    assert len(list((tmp_path/'recovery').glob('*.jsonl'))) == 1
