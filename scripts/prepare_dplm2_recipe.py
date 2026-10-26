"""Freeze the DPLM-2 recipe with FAST tokens; never submits training."""
import argparse
import json
import shutil
from pathlib import Path

from fast_dplm.data import TokenCache, sha256
from fast_dplm.evaluation import make_panel
from fast_dplm.provenance import decoder_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'val-cache', 'pretrained', 'decoder-lineage', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--canary', action='store_true')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--schedule-steps', type=int, default=100000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--world-size', type=int, choices=[1,2,4,8], default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=3)
    p.add_argument('--self-mixup', choices=['off', 'released'], default='off')
    p.add_argument('--noise-weighting', choices=['linear', 'constant_mass'], default='linear')
    p.add_argument('--eval-panel', type=Path,
                   help='Pre-frozen common validation panel; never silently regenerate its membership')
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.steps <= a.schedule_steps <= 100000 or a.schedule_steps <= 2000:
        raise ValueError('fresh output and valid endpoint required')
    global_batch = a.world_size*a.batch_size*a.accumulate
    if min(a.batch_size,a.accumulate)<1 or global_batch%4:
        raise ValueError('positive batch settings and balanced global task quarters required')
    train, val = TokenCache(a.cache), TokenCache(a.val_cache)
    if train.manifest['fingerprint'] != val.manifest['fingerprint']:
        raise ValueError('train/validation tokenizer mismatch')
    if {(e['source'],e['entry_id']) for e in train.entries} & {(e['source'],e['entry_id']) for e in val.entries}:
        raise ValueError('train/validation overlap')
    decoder = decoder_provenance(train.manifest, a.decoder_lineage)
    a.out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    fast = repo.parent/'fast-rewrite'
    shutil.copytree(repo/'src/fast_dplm', a.out/'code/src/fast_dplm', ignore=shutil.ignore_patterns('__pycache__'))
    for file in ('train_fast_production.py', 'run_fast_bundle.py', 'evaluate_fast_samples.py',
                 'run_fast_evaluation.py', 'generate_fast_dplm.py', 'evaluate_fast_checkpoint.py'):
        shutil.copy2(repo/file, a.out/'code'/file)
    shutil.copytree(fast/'src', a.out/'fast_code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (a.out/'fast_code/scripts').mkdir()
    shutil.copy2(fast/'scripts/decode_dplm_samples.py', a.out/'fast_code/scripts/decode_dplm_samples.py')
    shutil.copy2(repo/'requirements-fast.txt', a.out/'requirements-fast.txt')
    shutil.copy2(a.decoder_lineage, a.out/'decoder_lineage.json')
    shutil.copy2(__file__, a.out/'prepare_dplm2_recipe.py')
    for file in ('submit_dplm2_recipe.sbatch', 'submit_dplm2_recipe_panel.sbatch',
                 'verify_dplm2_recipe_checkpoint.py'):
        shutil.copy2(repo/'scripts'/file, a.out/file)
    panel = json.loads(a.eval_panel.read_text()) if a.eval_panel else make_panel(val, per_source=2 if a.canary else 64,
        lengths=[100] if a.canary else [100,200,300,400,500],
        per_length=1 if a.canary else 4, sampling_steps=5 if a.canary else 100)
    if panel['cache_digest'] != val.digest:
        raise ValueError('evaluation panel belongs to another validation cache')
    (a.out/'panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    cfg = dict(cache=str(a.cache.resolve()), val_cache=str(a.val_cache.resolve()),
        pretrained=str(a.pretrained.resolve()), out=str((a.out/'run').resolve()),
        decoder_lineage=str((a.out/'decoder_lineage.json').resolve()),
        eval_panel=str((a.out/'panel.json').resolve()), mode='index', architecture='dplm2_recipe', seed=a.seed,
        self_mixup=a.self_mixup, noise_weighting=a.noise_weighting,
        steps=a.steps, schedule_steps=a.schedule_steps, warmup=2000, warmup_init_lr=1e-7,
        lr=1e-4, batch_size=a.batch_size, accumulate=a.accumulate,
        checkpoint_every=20 if a.canary else 500, validate_every=50 if a.canary else 500,
        milestone_every=100 if a.canary else 5000, gradient_checkpointing=False,
        sample_length=100, sample_steps=100)
    (a.out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    # Measure the declared stateless sampler; do not equate sequences and tokens.
    sizes = [sum(train.entries[i]['length'] for i in train.draw_update(step, global_batch, 1234)) for step in range(100)]
    profile = dict(train=train.profile, validation_entries=len(panel['entries']),
        training_entries=len(train.entries), training_residues=sum(e['length'] for e in train.entries),
        cache_digest=train.digest, val_digest=val.digest, decoder=decoder,
        batch_sequences=global_batch, first100_mean_valid_residues=sum(sizes)/len(sizes),
        first100_mean_paired_tokens=2*(sum(sizes)/len(sizes)+2*global_batch),
        reference_recipe_sha256={name:sha256(repo/name) for name in (
            'configs/experiment/dplm2/dplm2_650m.yaml', 'src/byprot/models/utils.py',
            'src/byprot/models/dplm2/dplm2.py', 'src/byprot/models/dplm2/modules/dplm2_modeling_esm.py',
            'src/byprot/modules/cross_entropy.py')},
        task_mix='25% joint, 25% structure|AA, 25% AA|structure, 25% blocked cross-modality',
        objective=dict(self_mixup=a.self_mixup, noise_weighting=a.noise_weighting,
            normalization='global supervised-token means; constant_mass preserves weight mass per modality/task',
            mixup='released shared-vocabulary argmax, complemented masks, all-residue loss' if a.self_mixup=='released' else 'disabled'),
        status='prepared; no training submitted',
        limitations=['Existing whole-monomer data/splits, not the published confidence-filtered SwissProt/PDB corpus',
                     'Global masked-token means across accumulation; released trainer averages local batch means',
                     'Native-token AA conditioning uses an AA-aware FAST encoder, not sequence-blind inverse folding',
                     'Generation utility and remaining packing defects require measurement'])
    (a.out/'preparation.json').write_text(json.dumps(profile, indent=2)+'\n')
    hashes = {str(f.relative_to(a.out)): sha256(f) for f in sorted(a.out.rglob('*')) if f.is_file()}
    (a.out/'bundle.json').write_text(json.dumps(dict(schema='fast-dplm-run-bundle-v1', files=hashes,
        world_size=a.world_size, kind='canary' if a.canary else 'recipe'), indent=2)+'\n')
    print(json.dumps(profile, indent=2))


if __name__ == '__main__':
    main()
