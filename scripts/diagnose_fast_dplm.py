"""Inference-only generation/conditional diagnostics on frozen trained checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fast_dplm.data import RESIDUE, TokenCache, collate, sha256
from fast_dplm.diagnostic_sampling import POLICIES, generate_diagnostic
from fast_dplm.recipe import model_from_spec
from fast_dplm.refolding import write_json
from fast_dplm.sampling import write_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.audit/'plan.json').read_text())
    for rel, digest in plan['files'].items():
        if sha256(args.audit/rel) != digest:
            raise ValueError('audit code changed: '+rel)
    task = plan['tasks'][args.task]
    bundle = Path(plan['training_bundle'])
    cfg = json.loads((bundle/'config.json').read_text())
    path = bundle/'run/milestones'/f'step_{task["step"]:07d}.pt'
    if sha256(path) != plan['checkpoints'][str(task['step'])]:
        raise ValueError('checkpoint changed')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['step'] == task['step']
    model = model_from_spec(ck['model_spec'])
    model.load_state_dict(ck['model'], strict=True)
    model.cuda().eval()
    out = args.audit/'tasks'/task['name']
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(fingerprint=ck['fast_provenance']['fingerprint'], levels=model.levels,
        decoder_provenance=ck['fast_provenance'], model_step=ck['step'],
        model_sha256=plan['checkpoints'][str(task['step'])],
        audit_plan_sha256=sha256(args.audit/'plan.json'), task=task)
    if task['kind'] == 'unconditional':
        samples = []
        for item in plan['samples']:
            n = item['length']
            records = [dict(aa=np.zeros(n, dtype=np.uint8), fast_id=np.zeros(n, dtype=np.uint16),
                            chain_end=np.arange(n) == n-1)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                samples += generate_diagnostic(model, records, seed=item['seed'], policy=task['policy'])
        write_samples(out/'samples', samples, provenance | dict(policy=POLICIES[task['policy']]))
    else:
        cache = TokenCache(cfg['val_cache'])
        if cache.digest != plan['val_cache_digest']:
            raise ValueError('validation cache changed')
        rows = []
        for item in plan['conditional_entries']:
            i = item['index']
            if any(cache.entries[i][k] != item[k] for k in ('source', 'entry_id')):
                raise ValueError('validation entry changed')
            record = cache[i]
            batch = collate([record], model.aa_map, model.special, 'cuda')
            real = batch['state'] == RESIDUE
            generator = torch.Generator(device='cuda').manual_seed(31919+i)
            # Same random ordering at every checkpoint/noise level; no generated context.
            u = torch.rand(real.shape, generator=generator, device='cuda')
            for fraction in (.15, .5, .85, 1.):
                sm = real & (u < fraction)
                if not sm.any():
                    continue
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(batch, sm, torch.zeros_like(real))
                logits = output['struct'][0][sm].float()
                target = batch['fast'][sm]
                rows.append(dict(**item, fraction=fraction, masked=int(sm.sum()),
                    ce=float(F.cross_entropy(logits, target)),
                    accuracy=float((logits.argmax(-1) == target).float().mean())))
            for policy in ('legacy', 'release_default'):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = generate_diagnostic(model, [record], seed=31919+i,
                                                    policy=policy, condition='aa')
                dest = out/policy/f'entry_{i:05d}'
                write_samples(dest/'samples', generated, provenance | dict(condition='aa', entry=item,
                    policy=POLICIES[policy], native_cache_index=i))
        write_json(out/'teacher_forced.json', rows)
    write_json(out/'complete.json', provenance)
    print(f'Completed {task["name"]}', flush=True)


if __name__ == '__main__':
    main()
