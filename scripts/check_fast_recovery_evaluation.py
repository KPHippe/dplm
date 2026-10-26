"""GPU end-to-end evaluation probe on explicitly labeled diagnostic cache rows."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from fast_recovery_common import read, write, sha, released, build_model, Objective, Corpus
from fast_recovery_eval import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    bundle = read(root / 'bundle.json')
    for name, digest in bundle['files'].items():
        if sha(root / name) != digest:
            raise ValueError('integration probe changed')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group('gloo', init_method='file://' + temp + '/group', rank=0, world_size=1)
        api = released(root)
        model = build_model(api, read(root / 'config.json')).cuda()
        objective = Objective(model, api.criterion_class(label_smoothing=0., ignore_index=model.pad_id))
        corpus = Corpus(root / 'data/valid', model.tokenizer.all_tokens)
        evaluate(model, objective, api, corpus, root, 'probe', 0, sampling=True, small=True)
        dist.destroy_process_group()
    report = read(root / 'probe/evaluation/step_0000000/geometry.json')
    assert report['failures'] == 0
    for kind in ('conditional', 'joint', 'control'):
        assert report['groups'][kind]['summary']['scored'] == 2
    write(root / 'integration_check.json', dict(passed=True, diagnostic_validation_entries=32,
        conditional=2, joint=2, controls=2, steps=20, trained_updates=0,
        bundle_sha256=sha(root / 'bundle.json'),
        scope='End-to-end FAST evaluation plumbing only. Synthetic/untrained generator; no quality comparison. Fixture cache is not the production corpus.'))


if __name__ == '__main__':
    main()
