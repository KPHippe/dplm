"""Prepare an immutable pilot bundle; never submits a job."""
import argparse
import json
import shutil
from pathlib import Path

from fast_dplm.data import TokenCache, sha256
from fast_dplm.evaluation import make_panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'val-cache', 'pretrained', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--canary', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    train, val = TokenCache(a.cache), TokenCache(a.val_cache)
    if train.manifest['fingerprint'] != val.manifest['fingerprint']:
        raise ValueError('train/validation tokenizer mismatch')
    if {(e['source'],e['entry_id']) for e in train.entries} & {(e['source'],e['entry_id']) for e in val.entries}:
        raise ValueError('train/validation overlap')
    a.out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / 'src/fast_dplm', a.out / 'code/src/fast_dplm', ignore=shutil.ignore_patterns('__pycache__'))
    for file in ('train_fast_production.py', 'run_fast_bundle.py', 'evaluate_fast_samples.py', 'run_fast_evaluation.py'):
        shutil.copy2(repo / file, a.out / 'code' / file)
    fast = Path('/hpc/mydata/kyle.hippe/github/fast-rewrite')
    shutil.copytree(fast / 'src', a.out / 'fast_code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (a.out / 'fast_code/scripts').mkdir()
    shutil.copy2(fast/'scripts/decode_dplm_samples.py', a.out/'fast_code/scripts/decode_dplm_samples.py')
    shutil.copy2(repo / 'requirements-fast.txt', a.out / 'requirements-fast.txt')
    panel = make_panel(val, per_source=2 if a.canary else 64,
        lengths=[64] if a.canary else [100,200,300,400,500],
        per_length=1 if a.canary else 4, sampling_steps=3 if a.canary else 100)
    (a.out / 'panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    cfg = dict(cache=str(a.cache.resolve()), val_cache=str(a.val_cache.resolve()),
        pretrained=str(a.pretrained.resolve()), out=str((a.out / 'run').resolve()),
        eval_panel=str((a.out / 'panel.json').resolve()), mode='components', seed=42,
        steps=8 if a.canary else 5000, schedule_steps=8 if a.canary else 5000,
        warmup=2 if a.canary else 500, lr=1e-4, batch_size=4, accumulate=4,
        checkpoint_every=2 if a.canary else 250, validate_every=2 if a.canary else 250,
        milestone_every=4 if a.canary else 1000, gradient_checkpointing=False,
        sample_length=100, sample_steps=100)
    (a.out / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    profile = dict(train=train.profile, validation_entries=len(panel['entries']),
        training_entries=len(train.entries), training_residues=sum(e['length'] for e in train.entries),
        cache_digest=train.digest, val_digest=val.digest, mode='canary' if a.canary else 'pilot',
        status='prepared; no job submitted')
    (a.out / 'preparation.json').write_text(json.dumps(profile, indent=2)+'\n')
    hashes = {str(f.relative_to(a.out)): sha256(f) for f in sorted(a.out.rglob('*')) if f.is_file()}
    (a.out / 'bundle.json').write_text(json.dumps(dict(schema='fast-dplm-run-bundle-v1',
        files=hashes, world_size=8, kind='canary' if a.canary else 'pilot'), indent=2)+'\n')
    print(json.dumps(profile, indent=2))


if __name__ == '__main__':
    main()
