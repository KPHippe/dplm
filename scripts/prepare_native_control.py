"""Freeze two matched native-tokenizer runs after exact corpus conversion succeeds."""
import argparse
import json
import shutil
from pathlib import Path

from fast_dplm.data import TokenCache, sha256
from fast_dplm.native import remap_panel


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    execution = root/'execution'
    settings = read(execution/'settings.json')
    old_root = Path(settings['fast_screen'])
    if not read(root/'corpus_verification.json')['complete']:
        raise ValueError('native corpus conversion incomplete')
    train, val = TokenCache(root/'cache/train'), TokenCache(root/'cache/val')
    original_train = TokenCache(Path(read(root/'plan.json')['source'])/'train')
    original_val = TokenCache(Path(read(root/'plan.json')['source'])/'val')
    if train.entries != original_train.entries or val.entries != original_val.entries:
        raise ValueError('unmatched corpus membership')
    panel = remap_panel(read(old_root/'panel.json'), val, original_val)
    cells = []
    for cell, seed, canary in [('canary', 42, True), ('control_seed42', 42, False), ('control_seed43', 43, False)]:
        target = root/'training'/cell
        target.mkdir(parents=True, exist_ok=False)
        baseline = old_root/f'control_seed{seed}'
        base_cfg = read(baseline/'config.json')
        shutil.copytree(baseline/'code', target/'code', ignore=shutil.ignore_patterns('__pycache__'))
        source = execution/'dplm_src/fast_dplm'
        allowed = {'data.py', 'provenance.py', 'native.py', 'generation_validation.py'}
        for file in sorted(source.glob('*.py')):
            original = target/'code/src/fast_dplm'/file.name
            if file.name not in allowed and (not original.exists() or sha256(file) != sha256(original)):
                raise ValueError(f'unsupported generator code change: {file.name}')
            shutil.copy2(file, original)
        for name in ('native_control_evaluate.py', 'verify_dplm2_recipe_checkpoint.py'):
            shutil.copy2(execution/name, target/'code'/name)
        shutil.copy2(baseline/'submit_dplm2_recipe.sbatch', target/'submit_dplm2_recipe.sbatch')
        cfg = base_cfg | dict(cache=str(root/'cache/train'), val_cache=str(root/'cache/val'),
                              out=str(target/'run'), eval_panel=str(target/'panel.json'), decoder_lineage=None)
        if canary:
            cfg.update(steps=20, checkpoint_every=10, validate_every=10, milestone_every=20)
        for key in set(base_cfg)-{'cache', 'val_cache', 'out', 'eval_panel', 'decoder_lineage'}:
            if canary and key in {'steps', 'checkpoint_every', 'validate_every', 'milestone_every'}:
                continue
            if cfg[key] != base_cfg[key]:
                raise ValueError(f'non-tokenizer setting changed: {key}')
        write(target/'config.json', cfg)
        write(target/'panel.json', panel)
        comparison = dict(fast_screen=str(old_root), fast_baseline_bundle_sha256=sha256(baseline/'bundle.json'),
                          fast_config_sha256=sha256(baseline/'config.json'), source_panel_sha256=sha256(old_root/'panel.json'),
                          tokenizer='released DPLM2 LFQ8192; frozen encoder/decoder',
                          differences=['LFQ8192 versus FAST4096; expanded shared embedding/head has more parameters',
                                       'Native coordinate-only encoder versus AA-aware FAST encoder',
                                       'Native backbone/CB output versus FAST full atoms; common chemistry N/CA/C/O only',
                                       'Vocabulary expansion consumes different model-initialization RNG; data/noise draws remain matched'],
                          frozen_trunk='same original DPLM650M', adapter='rank16 alpha32 dropout0.1',
                          objective='self_mixup=off; linear weights',
                          requested_evaluation_steps=[10000, 20000, 40000, 60000])
        write(target/'comparison.json', comparison)
        files = {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}
        write(target/'bundle.json', dict(schema='fast-dplm-run-bundle-v1', files=files, world_size=8,
                                         kind='recipe', representation='dplm2_native_lfq8192', purpose='canary' if canary else 'production'))
        cells.append(dict(cell=cell, seed=seed, steps=cfg['steps'], global_batch=192, bundle_sha256=sha256(target/'bundle.json')))
    write(root/'training_preparation.json', dict(complete=True, cells=cells, source_cache_digest=original_train.digest,
                                                native_cache_digest=train.digest, panel_sha256=sha256(root/'training/control_seed42/panel.json')))
    print(json.dumps(cells), flush=True)


if __name__ == '__main__':
    main()
