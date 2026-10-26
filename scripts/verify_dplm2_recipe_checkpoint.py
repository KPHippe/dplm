"""Check real LoRA freeze boundaries, schedule/exposure and bundle integrity."""
import argparse
import json
from pathlib import Path

import torch

from fast_dplm.data import TokenCache, collate, sha256
from fast_dplm.model import noise_plan
from fast_dplm.objectives import loss_plan
from fast_dplm.recipe import model_from_spec
from fast_dplm.training import learning_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    bundle = json.loads((a.bundle/'bundle.json').read_text())
    for path, digest in bundle['files'].items():
        if sha256(a.bundle/path) != digest:
            raise ValueError(f'frozen bundle changed: {path}')
    cfg = json.loads((a.bundle/'config.json').read_text())
    run = Path(cfg['out'])
    state = torch.load(run/'last.ckpt', map_location='cpu', weights_only=False, mmap=True)
    model = model_from_spec(state['model_spec'])
    model.load_state_dict(state['model'], strict=True)
    parent = torch.load(Path(cfg['pretrained'])/'pytorch_model.bin', map_location='cpu',
                        weights_only=True, mmap=True)
    frozen = 0
    for name, value in model.named_parameters():
        if value.requires_grad:
            if not any(part in name for part in ('.a.weight', '.b.weight', '.embeddings.', '.lm_head.')):
                raise ValueError(f'unexpected trainable parameter: {name}')
            continue
        key = name.removeprefix('net.').replace('.base.', '.')
        if not torch.equal(value, parent[key]):
            raise ValueError(f'frozen original DPLM parameter changed: {name}')
        frozen += value.numel()
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1, state['step']+1)):
        raise ValueError('missing/duplicated logged update')
    cache = TokenCache(cfg['cache'])
    global_batch = bundle['world_size']*cfg['batch_size']*cfg['accumulate']
    if global_batch%4:
        raise ValueError('global batch cannot have balanced task quarters')
    exposure = {}
    for row in rows:
        step = row['step']
        lr = learning_rate(step-1, cfg['lr'], cfg['warmup'], cfg['schedule_steps'], cfg['warmup_init_lr'])
        if row['lr'] != lr or row['task_counts'] != [global_batch//4]*4:
            raise ValueError('schedule or task mixture changed')
        indices = cache.draw_update(step-1, global_batch, 1234)
        if 'objective' in row:
            batch = collate([cache[i] for i in indices], model.aa_map, model.special)
            generator = torch.Generator().manual_seed(cfg['seed']*1000003+step-1)
            noising = noise_plan(batch, generator, balanced=True)
            expected = loss_plan(batch, noising, self_mixup=cfg.get('self_mixup','off'),
                                 noise_weighting=cfg.get('noise_weighting','linear'))
            if row['objective']['self_mixup'] != cfg.get('self_mixup','off') or row['objective']['noise_weighting'] != cfg.get('noise_weighting','linear'):
                raise ValueError('wrong training objective logged')
            counts = [int(expected[k].sum()) for k in ('struct_mask','aa_mask')]
            mass = [float((expected[m].sum(-1)*expected[w]).sum()) for m,w in [('struct_mask','sw'),('aa_mask','aw')]]
            if counts != row['objective']['supervised_tokens'] or mass != row['objective']['weight_mass']:
                raise ValueError('objective masks/weight mass changed')
        for i in indices:
            e = cache.entries[i]
            exposure[e['source']] = exposure.get(e['source'], 0)+e['length']
        if exposure != row['exposure']:
            raise ValueError('training trace/exposure differs after restart')
    if exposure != state['exposure'] or state['step'] != cfg['steps']:
        raise ValueError('checkpoint endpoint/exposure mismatch')
    ordered = sorted(r['update_seconds'] for r in (rows[10:] or rows))
    result = dict(step=state['step'], strict_reload=True, frozen_original_dplm_elements=frozen,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        finite_training=all(torch.isfinite(torch.tensor([r[k] for k in ('loss','gradient_norm')])).all().item() for r in rows),
        trace_exposure_schedule_verified=True, exposure=exposure,
        median_update_seconds=ordered[len(ordered)//2], peak_gpu_gb=max(r['peak_memory_gb'] for r in rows),
        all_bundle_files_verified=True, verifier_sha256=sha256(__file__),
        objective=dict(self_mixup=cfg.get('self_mixup','off'), noise_weighting=cfg.get('noise_weighting','linear')),
        global_batch=global_batch, world_size=bundle['world_size'],
        limitation='Strict restart and exact draw/LR continuity; not a bitwise interrupted/uninterrupted comparison')
    if not result['finite_training']:
        raise ValueError('nonfinite training diagnostics')
    (a.bundle/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
