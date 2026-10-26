"""Fixed entry/mask and generation panels, isolated from training RNG."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .data import RESIDUE, collate
from .model import loss_sums
from .sampling import generate, write_samples


def make_panel(cache, per_source=64, lengths=(100, 200, 300, 400), per_length=4, sampling_steps=100):
    chosen = []
    for source in cache.sources:
        buckets = [[] for _ in range(4)]
        for i, e in enumerate(cache.entries):
            if e['source'] == source:
                buckets[max(0, min((e['length']-60)//114, 3))].append(i)
        for bucket in buckets:
            bucket.sort(key=lambda i: hashlib.sha256((source+cache.entries[i]['entry_id']+'919').encode()).digest())
        selected = []
        while len(selected) < per_source and any(buckets):
            for bucket in buckets:
                if bucket and len(selected) < per_source:
                    selected.append(bucket.pop())
        chosen.extend(selected)
    return dict(schema='fast-dplm-eval-panel-v1', cache_digest=cache.digest,
        entries=[dict(index=i, source=cache.entries[i]['source'], entry_id=cache.entries[i]['entry_id']) for i in chosen],
        tasks=['joint', 'structure_given_sequence', 'aa_given_fast', 'independent'], mask_seed=919,
        sampling_steps=sampling_steps,
        samples=[dict(length=n, seed=919+1000*n+j) for n in lengths for j in range(per_length)])


def fixed_plan(batch, entry, task, seed):
    key = f"{seed}:{entry['source']}:{entry['entry_id']}:{task}"
    number = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63)
    g = torch.Generator().manual_seed(number)
    t = (75, 250, 425)[number % 3]
    ts, ta = (t, 0) if task == 1 else (0, t) if task == 2 else (t, t)
    real = batch['state'].cpu() == RESIDUE
    return dict(struct_mask=(torch.rand(real.shape, generator=g) < ts/500) & real,
        aa_mask=(torch.rand(real.shape, generator=g) < ta/500) & real,
        independent=torch.tensor([task == 3]), sw=torch.tensor([(501-ts)/500]),
        aw=torch.tensor([(501-ta)/500]))


@torch.inference_mode()
def evaluate_panel(model, cache, panel, out, step, provenance, sampling=True):
    out = Path(out)
    identity = dict(step=step, panel=panel, fingerprint=provenance['fingerprint'], sampling=sampling,
                    training_contract=provenance.get('training_contract'))
    if panel.get('generation_validation') is not None:
        from .generation_validation import validate_generation_panel, verify_generation_outputs
        validate_generation_panel(cache, panel['generation_validation'])
        identity['checkpoint_sha256'] = provenance.get('checkpoint_sha256')
    if out.exists():
        old = json.loads((out / 'complete.json').read_text())
        if {k:v for k,v in old.items() if k != 'sampling'} != {k:v for k,v in identity.items() if k != 'sampling'}:
            raise ValueError('existing evaluation belongs to another contract')
        if sampling and not old['sampling']:
            raise ValueError('existing denoising-only evaluation cannot be silently upgraded')
        if panel.get('generation_validation') is not None:
            verify_generation_outputs(out, panel['generation_validation'], sampling)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name+'.partial-', dir=out.parent))
    device = next(model.parameters()).device
    was_training = model.training
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            tasks = []
            for task, name in enumerate(panel['tasks']):
                numer = torch.zeros(2, device=device)
                counts = torch.zeros(2, device=device, dtype=torch.long)
                correct = torch.zeros_like(counts)
                for entry in panel['entries']:
                    i = entry['index']
                    if any(entry[k] != cache.entries[i][k] for k in ('source', 'entry_id')):
                        raise ValueError('evaluation entry identity changed')
                    batch = collate([cache[i]], model.aa_map, model.special, device)
                    q = {k:v.to(device) for k,v in fixed_plan(batch, entry, task, panel['mask_seed']).items()}
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        output = model(batch, q['struct_mask'], q['aa_mask'], q['independent'])
                        n, c = loss_sums(model, output, batch, q)
                    pred = [x.argmax(-1) for x in output['struct']]
                    pred = model.pack(torch.stack(pred, -1)) if model.mode == 'components' else pred[0]
                    if 'struct_loss_logits' in output:
                        pred = output['struct_loss_logits'].argmax(-1) - output['struct_offset']
                    correct += torch.stack([((pred == batch['fast']) & q['struct_mask']).sum(),
                        ((output['aa'].argmax(-1) == batch['aa']) & q['aa_mask']).sum()])
                    numer += n
                    counts += c
                tasks.append(dict(task=name, weighted_loss_sums=numer.tolist(), masked_counts=counts.tolist(),
                    weighted_losses=[float(n/c) if int(c) else None for n,c in zip(numer,counts)],
                    accuracy=[float(n/c) if int(c) else None for n,c in zip(correct,counts)]))
            (tmp / 'denoising.json').write_text(json.dumps(dict(step=step, entries=len(panel['entries']), tasks=tasks), indent=2)+'\n')
            generation = panel.get('generation_validation')
            if generation is not None:
                from .generation_validation import full_mask_denoising, generate_validation, joint_sequence_report
                if generation['joint_samples'] != panel['samples']:
                    raise ValueError('joint requests disagree with generation-validation panel')
                report = full_mask_denoising(model, cache, generation)
                (tmp/'full_mask_denoising.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            if sampling:
                samples = []
                for item in panel['samples']:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        samples.extend(generate(model, item['length'], count=1, steps=panel['sampling_steps'], seed=item['seed']))
                write_samples(tmp / 'samples', samples, dict(fingerprint=provenance['fingerprint'], levels=model.levels,
                    decoder_provenance={k:v for k,v in provenance.items() if k != 'training_contract'},
                    model_step=step, panel=panel, training_contract=provenance.get('training_contract')))
                if generation is not None:
                    validation_provenance = dict(fingerprint=provenance['fingerprint'], levels=model.levels,
                        model_step=step, decoder_provenance=provenance,
                        validation_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob('*.py')})
                    generate_validation(model, cache, generation, tmp/'conditional_samples', validation_provenance)
                    sequence_report = joint_sequence_report(tmp/'samples', panel['samples'])
                    (tmp/'sequence_metrics.json').write_text(json.dumps(sequence_report, indent=2, allow_nan=False)+'\n')
            (tmp / 'complete.json').write_text(json.dumps(identity, indent=2)+'\n')
        os.rename(tmp, out)
    finally:
        # Incomplete directories are identifiable and never mistaken for completed output.
        model.train(was_training)
