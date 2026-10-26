"""Freeze and retokenize the exact FAST corpus with the released native LFQ8k encoder."""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

DPLM = Path('/hpc/mydata/kyle.hippe/github/dplm')
FAST = DPLM.parent/'fast-rewrite'


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def prepare(args):
    from fast_dplm.data import sha256
    from fast_dplm.native import digest
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = root/'runtime'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git')
    for source, target in [(DPLM/'src', runtime/'dplm/src'),
                           (DPLM/'vendor/openfold', runtime/'dplm/vendor/openfold'),
                           (FAST/'outputs/dplm/recipe_screen_v1_20260930/control_seed42/fast_code/src', runtime/'fast/src'),
                           (DPLM/'outputs/native_tokenizer_audit/deps', runtime/'dplm/outputs/native_tokenizer_audit/deps')]:
        shutil.copytree(source, target, ignore=ignore)
    (runtime/'dplm/scripts').mkdir()
    for name in ('native_tokenizer_loader.py', 'native_control_corpus.py'):
        shutil.copy2(DPLM/'scripts'/name, runtime/'dplm/scripts'/name)
    asset = DPLM/'outputs/native_tokenizer_audit/esm_if1_gvp4_t16_142M_UR50.pt'
    (runtime/'dplm/outputs/native_tokenizer_audit'/asset.name).symlink_to(asset)
    previous = read(DPLM/'outputs/native_tokenizer_audit/loader_provenance.json')
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
    runtime_manifest = {'schema': 'native-dplm-runtime-v1', 'files': files}
    atomic(root/'runtime_manifest.json', runtime_manifest)
    contract = dict(representation='dplm2_native_lfq8192', vocabulary_size=8192,
                    checkpoint_sha256=previous['checkpoint_sha256'], config_sha256=previous['config_sha256'],
                    auxiliary_sha256=sha256(asset), runtime_manifest_sha256=sha256(root/'runtime_manifest.json'),
                    precision='fp32; TF32 disabled; deterministic algorithms required; one structure per encode',
                    encoder_inputs='observed N/CA/C coordinates, valid mask, length; no AA identities',
                    decoder_recycles=1, sequence_conditioning='none for tokenizer')
    provenance = {k: previous[k] for k in ('config', 'checkpoint', 'config_sha256', 'checkpoint_sha256')}
    provenance.update(representation='dplm2_native_lfq8192', levels=[2]*13,
                      fingerprint='native-lfq8k:'+digest(contract), tokenizer_contract=contract)
    jobs = []
    for split in ('train', 'val'):
        source = args.source.resolve()/split
        corpus = read(source/'manifest.json')
        for item in corpus['shards']:
            jobs.append(dict(index=len(jobs), split=split, source=str(source/item['path']),
                             source_manifest_sha256=item['sha256'], source_corpus_sha256=sha256(source/'manifest.json'),
                             target=str(root/'cache'/split/item['path'])))
    plan = dict(schema='native-dplm-matched-corpus-plan-v1', root=str(root), source=str(args.source.resolve()),
                runtime_manifest_sha256=sha256(root/'runtime_manifest.json'), provenance=provenance, jobs=jobs,
                source_panel=str((FAST/'outputs/dplm/recipe_screen_v1_20260930/panel.json').resolve()))
    plan['source_panel_sha256'] = sha256(plan['source_panel'])
    atomic(root/'plan.json', plan)
    print(json.dumps(dict(root=str(root), shards=len(jobs), fingerprint=provenance['fingerprint'])))


def setup(root):
    from fast_dplm.data import sha256
    plan = read(root/'plan.json')
    if sha256(root/'runtime_manifest.json') != plan['runtime_manifest_sha256']:
        raise ValueError('native runtime manifest changed')
    for relative, expected in read(root/'runtime_manifest.json')['files'].items():
        if sha256(root/relative) != expected:
            raise ValueError(f'changed runtime file: {relative}')
    sys.path.insert(0, str(root/'runtime/dplm/scripts'))
    from native_tokenizer_loader import load_native_tokenizer
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model, loaded = load_native_tokenizer()
    for key in ('checkpoint_sha256', 'config_sha256', 'auxiliary_sha256'):
        if loaded[key] != plan['provenance']['tokenizer_contract'][key]:
            raise ValueError(f'wrong native asset: {key}')
    if int(model.decoder.cfg.trunk.max_recycles) != 1:
        raise ValueError('decoder pass count changed')
    return plan, model.cuda().eval().requires_grad_(False), loaded


def encode(model, entry, record, repeat=False):
    import numpy as np
    import torch
    from fast.inference import build_batch
    from fast.models.frames import apply_frame
    from fast_dplm.data import sha256
    if sha256(entry['path']) != entry['file_sha256']:
        raise ValueError('source structure changed')
    batch = build_batch(entry['path'], 'cpu', max_length=None)
    for key, field in [('aa', 'aatype'), ('residue_id', 'residue_id'), ('chain_end', 'chain_end')]:
        if not np.array_equal(record[key], batch[field][0].numpy()):
            raise ValueError(f'parsed topology/sequence differs: {entry["entry_id"]}/{key}')
    if not batch['residue_mask'].all() or batch['chain_id'].unique().numel() != 1:
        raise ValueError('native control requires the complete valid monomer')
    if not (np.diff(record['residue_id']) == 1).all():
        raise ValueError('native decoder assumes contiguous residues')
    xyz = apply_frame(batch['frames_R'][:, :, None], batch['frames_t'][:, :, None], batch['coords']).cuda()
    mask = torch.ones(xyz.shape[:2], device='cuda')
    length = torch.tensor([xyz.shape[1]], device='cuda')
    with torch.inference_mode():
        tokens = model.tokenize(xyz, mask, seq_length=length)
        if tokens.shape != xyz.shape[:2] or tokens.min() < 0 or tokens.max() >= 8192:
            raise ValueError('invalid native token IDs')
        if repeat:
            if not torch.equal(tokens, model.tokenize(xyz, mask, seq_length=length)):
                raise ValueError('repeated deterministic encoding differs')
            pre, _ = model.encode(xyz, mask, seq_length=length)
            quant, _, (_, _, ids) = model.quantize(pre, mask=mask.bool())
            if not torch.equal(ids, tokens) or not torch.allclose(quant, model.quantize.get_codebook_entry(tokens), atol=1e-6, rtol=0):
                raise ValueError('native codebook lookup mismatch')
            decoded = model.detokenize(tokens, res_mask=mask)
            if not torch.isfinite(decoded['atom37_positions']).all():
                raise ValueError('nonfinite native decode')
    return tokens[0].cpu().numpy().astype(np.uint16)


def export_shard(job, plan, model):
    import numpy as np
    from fast_dplm.data import TokenCache, sha256
    out, source = Path(job['target']), Path(job['source'])
    if sha256(source/'manifest.json') != job['source_manifest_sha256']:
        raise ValueError('source shard changed')
    if out.exists():
        cached = TokenCache(out)
        if cached.manifest['source_manifest_sha256'] != job['source_manifest_sha256']:
            raise ValueError('existing native shard belongs to another source')
        if cached.manifest['fingerprint'] != plan['provenance']['fingerprint']:
            raise ValueError('existing native shard has wrong tokenizer')
        return
    cache = TokenCache(source)
    tokens, start = [], time.monotonic()
    for i, entry in enumerate(cache.entries):
        tokens.append(encode(model, entry, cache[i]))
        if (i+1) % 250 == 0:
            print(json.dumps(dict(shard=job['index'], encoded=i+1, total=len(cache.entries), seconds=time.monotonic()-start)), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name+'.partial-', dir=out.parent))
    for name in cache.manifest['files']:
        if name != 'fast_id.npy':
            shutil.copy2(source/name, tmp/name)
    # The legacy array column name is retained for trainer compatibility; the
    # new schema, fingerprint, contract and 13 binary LFQ levels identify it.
    np.save(tmp/'fast_id.npy', np.concatenate(tokens), allow_pickle=False)
    manifest = dict(plan['provenance'], schema='native-dplm-cache-v1', alphabet=cache.manifest['alphabet'],
                    entries=len(cache.entries), residues=sum(e['length'] for e in cache.entries),
                    source_manifest_sha256=job['source_manifest_sha256'],
                    source_corpus_sha256=job['source_corpus_sha256'],
                    files={p.name: sha256(p) for p in sorted(tmp.iterdir())})
    atomic(tmp/'manifest.json', manifest)
    TokenCache(tmp)
    if out.exists():
        raise FileExistsError(out)
    os.rename(tmp, out)
    print(json.dumps(dict(shard=job['index'], complete=True, seconds=time.monotonic()-start)), flush=True)


def finalize(root):
    from fast_dplm.data import TokenCache, sha256
    from fast_dplm.native import remap_panel
    plan = read(root/'plan.json')
    evidence = {}
    for split in ('train', 'val'):
        source = TokenCache(Path(plan['source'])/split)
        jobs = [j for j in plan['jobs'] if j['split'] == split]
        target = root/'cache'/split
        shards = []
        for job in jobs:
            leaf = TokenCache(job['target'])
            original = TokenCache(job['source'])
            if leaf.entries != original.entries:
                raise ValueError('native cache membership/order changed')
            for key in ('aa', 'residue_id', 'chain_end', 'offsets'):
                if sha256(Path(job['target'])/(key+'.npy')) != sha256(Path(job['source'])/(key+'.npy')):
                    raise ValueError('retokenization changed sequence/topology/order')
            shards.append(dict(path=str(Path(job['target']).relative_to(target)), sha256=sha256(Path(job['target'])/'manifest.json')))
        manifest = dict(plan['provenance'], schema='native-dplm-corpus-v1', shards=shards,
                        entries=len(source.entries), residues=sum(e['length'] for e in source.entries),
                        source_corpus_sha256=source.digest)
        if (target/'manifest.json').exists():
            raise FileExistsError(target/'manifest.json')
        atomic(target/'manifest.json', manifest)
        native = TokenCache(target)
        if native.entries != source.entries or native.profile != source.profile:
            raise ValueError('sampling membership/distribution changed')
        for step in range(100):
            if native.draw_update(step, 192, 1234) != source.draw_update(step, 192, 1234):
                raise ValueError('sampler plan changed')
        evidence[split] = dict(entries=len(native.entries), residues=manifest['residues'], cache_digest=native.digest,
                               source_digest=source.digest, identical_order_topology_sequence_sampler=True)
    source_val = TokenCache(Path(plan['source'])/'val')
    val = TokenCache(root/'cache/val')
    if sha256(plan['source_panel']) != plan['source_panel_sha256']:
        raise ValueError('source evaluation panel changed')
    panel = remap_panel(read(plan['source_panel']), val, source_val)
    atomic(root/'panel.json', panel)
    atomic(root/'corpus_verification.json', dict(complete=True, splits=evidence, panel_sha256=sha256(root/'panel.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'smoke', 'export', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(args)
        return
    sys.path[:0] = [str(root/'runtime/dplm/src'), str(root/'runtime/fast/src')]
    if args.command == 'finalize':
        finalize(root)
        return
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    import torch
    torch.cuda.set_device(rank)
    plan, model, loaded = setup(root)
    if args.command == 'smoke':
        from fast_dplm.data import TokenCache
        cache = TokenCache(Path(plan['source'])/'val')
        panel = read(plan['source_panel'])['generation_validation']['entries']
        checks = []
        for item in panel[::4]:
            i = item['index']
            ids = encode(model, cache.entries[i], cache[i], repeat=True)
            checks.append(dict(index=i, source=item['source'], length=len(ids), min=int(ids.min()), max=int(ids.max())))
        atomic(root/'tokenizer_smoke.json', dict(passed=True, checks=checks, loader=loaded))
    else:
        for job in plan['jobs'][rank::world]:
            export_shard(job, plan, model)
        atomic(root/f'export_status/rank{rank}.json', dict(complete=True, rank=rank, world=world))


if __name__ == '__main__':
    main()
