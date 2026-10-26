"""Freeze a separate evaluator overlay; never mutate or submit the training run."""
import argparse
import json
import shutil
from pathlib import Path

from fast_dplm.data import sha256
from fast_dplm.refolding import SETTINGS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if args.out.exists() or min(args.steps) < 0 or len(set(args.steps)) != len(args.steps):
        raise ValueError('fresh output and unique nonnegative steps required')
    training = args.training_bundle.resolve()
    for rel, digest in json.loads((training/'bundle.json').read_text())['files'].items():
        if sha256(training/rel) != digest:
            raise ValueError(f'training bundle integrity failure: {rel}')
    repo = Path(__file__).resolve().parents[1]
    fast = repo.parent/'fast-rewrite'
    args.out.mkdir(parents=True)
    # Reuse the exact frozen cache/provenance helpers; add the new isolated backend.
    shutil.copytree(training/'code/src/fast_dplm', args.out/'code/src/fast_dplm',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(repo/'src/fast_dplm/refolding.py', args.out/'code/src/fast_dplm/refolding.py')
    for name in ('evaluate_fast_samples_esmfold2.py', 'run_fast_esmfold2_panel.py'):
        shutil.copy2(repo/name, args.out/'code'/name)
    for name in ('prepare_fast_esmfold2.py', 'submit_fast_esmfold2.sbatch'):
        shutil.copy2(repo/'scripts'/name, args.out/name)
    write_json(args.out/'config.json', dict(schema='fast-dplm-esmfold2-overlay-v1',
        training_bundle=str(training), training_bundle_sha256=sha256(training/'bundle.json'),
        fold_settings=SETTINGS, steps=args.steps, decode_device='cpu',
        python=str(fast/'.venv/bin/python'), renderers=['selected', 'parent'],
        raw_api_cache='fold_cache', samples_per_step=len(json.loads((training/'panel.json').read_text())['samples'])))
    hashes = {str(p.relative_to(args.out)): sha256(p) for p in sorted(args.out.rglob('*')) if p.is_file()}
    write_json(args.out/'bundle.json', dict(schema='fast-dplm-esmfold2-overlay-v1', files=hashes))
    print(json.dumps(dict(bundle=str(args.out.resolve()), files=len(hashes), model=SETTINGS['model'])))


if __name__ == '__main__':
    main()
