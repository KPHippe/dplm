"""Opt-in held-out generation validation; separate from historical v1 evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import RESIDUE, collate
from .diagnostic_sampling import POLICIES, generate_diagnostic
from .generation_metrics import sequence_metrics, sequence_panel
from .sampling import write_samples

SCHEMA = 'fast-dplm-generation-validation-v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record_hash(record):
    h = hashlib.sha256()
    for key in ('aa', 'fast_id', 'residue_id', 'chain_end'):
        a = np.asarray(record[key])
        h.update(key.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def make_generation_panel(cache, per_source_bin=4, seed=30930, sampling_steps=100,
                          policy='legacy', joint_samples=(), exclude_identities=()):
    """Choose complete cached monomers from val only; no recropping or test reads."""
    if per_source_bin < 1 or sampling_steps < 1 or policy not in POLICIES:
        raise ValueError('invalid panel settings')
    excluded = set(map(tuple, exclude_identities))
    if any(e.get('split') != 'val' for e in cache.entries):
        raise ValueError('generation panel requires exclusively validation membership')
    entries = []
    for source in cache.sources:
        for lo, hi in ((60, 173), (174, 287), (288, 401), (402, 512)):
            candidates = [i for i, e in enumerate(cache.entries) if e['source'] == source
                and lo <= e['length'] <= hi and (source, e['entry_id']) not in excluded]
            candidates.sort(key=lambda i: digest_json([seed, source, cache.entries[i]['entry_id']]))
            if len(candidates) < per_source_bin:
                raise ValueError(f'insufficient validation entries: {source}/{lo}-{hi}')
            for i in candidates[:per_source_bin]:
                e = cache.entries[i]
                entries.append(dict(index=i, source=source, entry_id=e['entry_id'], length=e['length'],
                    split='val', cluster=e['cluster'], file_sha256=e['file_sha256'],
                    record_sha256=record_hash(cache[i]), seed=seed+i,
                    crop='complete cached monomer; source residue IDs and termini preserved'))
    return dict(schema=SCHEMA, cache_digest=cache.digest, entries=entries, sampling_steps=sampling_steps,
        policy=policy, policy_settings=POLICIES[policy], selection_seed=seed,
        excluded_identities=sorted(map(list, excluded)), joint_samples=list(joint_samples),
        full_mask_tasks=['structure_given_native_sequence', 'structure_without_sequence', 'sequence_without_structure'],
        geometry='proper-rotation CA RMSD and CA-distance lDDT; no optimized TM-score',
        chemistry='raw emitted atoms; diagnostic only; no readiness gate')


def validate_generation_panel(cache, panel):
    if panel.get('schema') != SCHEMA or panel.get('cache_digest') != cache.digest:
        raise ValueError('generation panel version/cache mismatch')
    if panel.get('policy') not in POLICIES or panel.get('policy_settings') != POLICIES[panel['policy']]:
        raise ValueError('sampler policy changed')
    if panel.get('sampling_steps', 0) < 1 or not panel.get('entries'):
        raise ValueError('incomplete generation panel')
    seen = set()
    for item in panel['entries']:
        e = cache.entries[item['index']]
        identity = (item['source'], item['entry_id'])
        if identity in seen or e.get('split') != 'val' or item.get('split') != 'val':
            raise ValueError('duplicate or non-validation membership')
        seen.add(identity)
        if any(item[k] != e[k] for k in ('source', 'entry_id', 'length', 'cluster', 'file_sha256')):
            raise ValueError('validation entry changed')
        if item['record_sha256'] != record_hash(cache[item['index']]):
            raise ValueError('cached tokens/sequence/crop changed')
    expected_tasks = ['structure_given_native_sequence', 'structure_without_sequence', 'sequence_without_structure']
    if panel.get('full_mask_tasks') != expected_tasks:
        raise ValueError('full-mask task contract changed')
    for sample in panel['joint_samples']:
        if not isinstance(sample['length'], int) or sample['length'] < 1 or not isinstance(sample['seed'], int):
            raise ValueError('invalid joint sample specification')


def sequence_condition(record):
    # Construct a whitelist from scratch: native FAST tokens/coordinates cannot enter sampler.
    aa = np.asarray(record['aa'], dtype=np.uint8).copy()
    return dict(aa=aa, fast_id=np.zeros(len(aa), dtype=np.uint16),
        residue_id=np.asarray(record['residue_id'], dtype=np.int32).copy(),
        chain_end=np.asarray(record['chain_end'], dtype=bool).copy())


@torch.inference_mode()
def full_mask_denoising(model, cache, panel):
    validate_generation_panel(cache, panel)
    device = next(model.parameters()).device
    rows = []
    for item in panel['entries']:
        native = cache[item['index']]
        batch = collate([native], model.aa_map, model.special, device)
        real = batch['state'] == RESIDUE
        for task in panel['full_mask_tasks']:
            sm = real
            am = torch.zeros_like(real) if task == 'structure_given_native_sequence' else real
            independent = torch.tensor([task != 'structure_given_native_sequence'], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                output = model(batch, sm, am, independent)
            target_aa = task == 'sequence_without_structure'
            logits = output['aa'][real] if target_aa else output['struct'][0][real]
            target = batch['aa'][real] if target_aa else batch['fast'][real]
            if model.mode != 'index':
                raise ValueError('full-mask validation currently requires categorical FAST indices')
            rows.append(dict(index=item['index'], task=task, residues=int(real.sum()),
                softmax_domain=(('full AA-output vocabulary (shared AA/native LFQ vocabulary in DPLM2Recipe)' if target_aa
                    else 'native LFQ structure vocabulary only') if panel.get('representation') == 'dplm2_native_lfq8192'
                    else ('full AA-output vocabulary (shared AA/FAST vocabulary in DPLM2Recipe)' if target_aa else 'FAST structure vocabulary only')),
                unweighted_ce=float(F.cross_entropy(logits.float(), target)),
                accuracy=float((logits.argmax(-1) == target).float().mean())))
    summary = {}
    for task in panel['full_mask_tasks']:
        selected = [row for row in rows if row['task'] == task]
        count = sum(row['residues'] for row in selected)
        summary[task] = dict(residues=count, softmax_domain=selected[0]['softmax_domain'], unweighted_ce=sum(row['unweighted_ce']*row['residues'] for row in selected)/count,
            accuracy=sum(row['accuracy']*row['residues'] for row in selected)/count)
    return dict(schema=SCHEMA, panel_sha256=digest_json(panel), rows=rows, summary=summary,
        description='One-step all-residue masked denoising; independent tasks hide both modalities. Structure CE normalizes over FAST tokens only, unlike the shared-vocabulary training loss. AA CE uses the full AA-output vocabulary. AA score is LM-retention diagnostic, not sequence generation quality.')


@torch.inference_mode()
def generate_validation(model, cache, panel, out, provenance):
    validate_generation_panel(cache, panel)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    rows = []
    for number, item in enumerate(panel['entries']):
        sample_id = f'entry_{number:04d}'
        try:
            record = sequence_condition(cache[item['index']])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                samples = generate_diagnostic(model, [record], steps=panel['sampling_steps'],
                    seed=item['seed'], policy=panel['policy'], condition='aa')
            write_samples(out/sample_id, samples, provenance | dict(validation_schema=SCHEMA,
                panel_sha256=digest_json(panel), entry=item, condition='native_aa_all_structure_masked'))
            rows.append(dict(id=sample_id, index=item['index'], status='ok'))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            rows.append(dict(id=sample_id, index=item['index'], status='failed', error=f'{type(exc).__name__}: {exc}'))
    result = dict(schema=SCHEMA, panel_sha256=digest_json(panel), provenance=provenance,
        requested=len(panel['entries']), generated=sum(r['status'] == 'ok' for r in rows), rows=rows)
    (out/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def joint_sequence_report(folder, expected):
    folder = Path(folder)
    from .data import sha256
    meta = json.loads((folder/'manifest.json').read_text())
    if len(meta['samples']) != len(expected):
        raise ValueError('joint sample count differs from frozen requests')
    rows = []
    for i, (item, request) in enumerate(zip(meta['samples'], expected, strict=True)):
        path = folder/item['file']
        if path.parent != folder or sha256(path) != item['sha256']:
            raise ValueError('joint sample integrity failure')
        with np.load(path, allow_pickle=False) as arrays:
            if len(arrays['aa']) != request['length']:
                raise ValueError('joint sample length changed')
            rows.append(dict(id=f'sample_{i:04d}', status='ok', seed=request['seed'], **sequence_metrics(arrays['aa'])))
    return dict(schema=SCHEMA, rows=rows, summary=sequence_panel(rows, len(expected)))


def verify_generation_outputs(out, panel, sampling):
    """Do not accept a completion marker when scored/generated artifacts are missing."""
    out = Path(out)
    report = json.loads((out/'full_mask_denoising.json').read_text())
    expected = {(entry['index'], task) for entry in panel['entries'] for task in panel['full_mask_tasks']}
    actual = [(row['index'], row['task']) for row in report['rows']]
    if report['panel_sha256'] != digest_json(panel) or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('full-mask output completeness failure')
    if not sampling:
        return
    manifest = json.loads((out/'conditional_samples/manifest.json').read_text())
    if manifest['panel_sha256'] != digest_json(panel) or manifest['requested'] != len(panel['entries']) or len(manifest['rows']) != len(panel['entries']):
        raise ValueError('conditional output completeness failure')
    from .data import sha256
    for number, (entry, row) in enumerate(zip(panel['entries'], manifest['rows'], strict=True)):
        name = f'entry_{number:04d}'
        if row['id'] != name or row['index'] != entry['index'] or row['status'] not in ('ok', 'failed'):
            raise ValueError('conditional output identity failure')
        if row['status'] == 'ok':
            folder = out/'conditional_samples'/name
            meta = json.loads((folder/'manifest.json').read_text())
            if meta['panel_sha256'] != digest_json(panel) or meta['entry'] != entry or len(meta['samples']) != 1:
                raise ValueError('conditional sample provenance failure')
            sample = meta['samples'][0]
            path = folder/sample['file']
            if path.parent != folder or sha256(path) != sample['sha256']:
                raise ValueError('conditional sample integrity failure')
    if manifest['generated'] != sum(row['status'] == 'ok' for row in manifest['rows']):
        raise ValueError('conditional generated count mismatch')
    observed = joint_sequence_report(out/'samples', panel['joint_samples'])
    if observed != json.loads((out/'sequence_metrics.json').read_text()):
        raise ValueError('joint sequence metric output changed')
