"""Native LFQ fold evaluation and matched backbone chemistry; never full-atom claims."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from fast_dplm.refolding import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def backbone_chemistry(xyz, mask, aa, residue_id):
    from fast.analysis.generated_chemistry import chemistry
    from native_tokenizer_loader import adjust_oxygen_pos
    import torch
    common = np.zeros_like(mask, dtype=bool)
    common[:, [0, 1, 2, 4]] = True
    if not mask[common].all():
        raise ValueError('incomplete N/CA/C/O correspondence')
    raw = chemistry(xyz, common, aa, residue_id, np.zeros(len(aa), dtype=np.int32))
    adjusted = adjust_oxygen_pos(torch.as_tensor(xyz).clone(), torch.ones(len(aa), dtype=torch.bool)).numpy()
    oxygen = chemistry(adjusted, common, aa, residue_id, np.zeros(len(aa), dtype=np.int32))
    return dict(raw=raw, matched_oxygen_adjustment=oxygen)


def verified_sample(folder, expected_fingerprint):
    from fast_dplm.data import sha256
    meta = read(folder/'manifest.json')
    if meta['fingerprint'] != expected_fingerprint or len(meta['samples']) != 1:
        raise ValueError('conditional token provenance mismatch')
    item = meta['samples'][0]
    path = folder/item['file']
    if path.parent != folder or sha256(path) != item['sha256']:
        raise ValueError('conditional sample changed')
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def decode_sample(model, sample, out, reference=None):
    import torch
    from fast_dplm.data import sha256
    from fast_dplm.generation_metrics import ca_geometry
    ids = np.asarray(sample['fast_id'])
    if ids.ndim != 1 or not len(ids) or (ids < 0).any() or (ids >= 8192).any():
        raise ValueError('invalid native token IDs')
    # Only IDs/mask reach the native decoder. Sequence is used after decoding to
    # define the common backbone chemistry graph, never for coordinate prediction.
    with torch.inference_mode():
        decoded = model.detokenize(torch.as_tensor(ids.astype(np.int64), device='cuda')[None])
    xyz = decoded['atom37_positions'][0].cpu().numpy()
    mask = decoded['atom37_mask'][0].bool().cpu().numpy()
    if not np.isfinite(xyz[mask]).all():
        raise ValueError('nonfinite native structure')
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out/'structure.npz', xyz=xyz, atom_mask=mask, aa=sample['aa'],
             residue_id=sample['residue_id'], chain_end=sample['chain_end'],
             auxiliary_identity_logits_argmax=decoded['aatype'][0].cpu().numpy())
    result = dict(status='ok', length=len(ids), structure_sha256=sha256(out/'structure.npz'),
                  backbone_chemistry=backbone_chemistry(xyz, mask, sample['aa'], sample['residue_id']))
    if reference is not None:
        result.update(ca_geometry(xyz[:, 1], reference[:, 1]))
    return result


def summarize(rows, requested, rmsd='ca_rmsd_a', fold_requested=True):
    ok = [r for r in rows if r['status'] == 'ok']
    folds = [r for r in ok if rmsd in r]
    return dict(requested=requested, scored=len(ok), failed_or_missing=requested-len(ok),
                median_ca_rmsd_a=float(np.median([r[rmsd] for r in folds])) if folds else None,
                below_2a=sum(r[rmsd] < 2 for r in folds) if fold_requested else None,
                success_denominator=requested if fold_requested else None, fold_scored=len(folds),
                mean_ca_lddt=float(np.mean([r['ca_lddt'] for r in folds])) if folds else None)


def decode(args, bundle):
    from fast.data.parsers import load_structure
    from fast_dplm.data import TokenCache, sha256
    from fast_dplm.generation_validation import validate_generation_panel, verify_generation_outputs
    from native_control_corpus import setup, encode
    _, model, loaded = setup(args.root)
    cfg = read(bundle/'config.json')
    cache = TokenCache(cfg['val_cache'])
    comparison = read(bundle/'comparison.json')
    out = args.root/'evaluation'/bundle.name
    for step in args.steps:
        folder = Path(cfg['out'])/'evaluation'/f'step_{step:07d}'
        complete = read(folder/'complete.json')
        panel = complete['panel']['generation_validation']
        if complete['step'] != step or complete['fingerprint'] != cache.manifest['fingerprint']:
            raise ValueError('evaluation/cache mismatch')
        validate_generation_panel(cache, panel)
        verify_generation_outputs(folder, panel, sampling=True)
        destination = out/f'step_{step:07d}'
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        conditional, controls, joint, fast = [], [], [], []
        original = Path(comparison['fast_screen'])/'native_scoring'/f'control_seed{cfg["seed"]}'/f'step_{step:07d}'
        fast_report = read(original/'report.json') if original.exists() else None
        generation = read(folder/'conditional_samples/manifest.json')
        for number, entry in enumerate(panel['entries']):
            name = f'entry_{number:04d}'
            source, record = cache.entries[entry['index']], cache[entry['index']]
            if sha256(source['path']) != entry['file_sha256']:
                raise ValueError('native reference changed')
            native = load_structure(source['path'])
            if not (np.array_equal(native.aatype, record['aa'])
                    and np.array_equal(native.residue_id, record['residue_id'])
                    and np.array_equal(native.chain_end_flags(), record['chain_end'])
                    and native.atom_mask[:, 1].all()):
                raise ValueError('reference sequence/topology differs')
            if not np.array_equal(encode(model, source, record), record['fast_id']):
                raise ValueError('deterministic native token re-encoding differs')
            for kind, rows, sample in [('controls', controls, record), ('conditional', conditional, None)]:
                try:
                    if kind == 'conditional':
                        if generation['rows'][number]['status'] != 'ok':
                            raise ValueError('generation failed')
                        sample = verified_sample(folder/'conditional_samples'/name, cache.manifest['fingerprint'])
                        for key in ('aa', 'residue_id', 'chain_end'):
                            if not np.array_equal(sample[key], record[key]):
                                raise ValueError('conditional generation changed sequence/topology')
                    row = decode_sample(model, sample, destination/kind/name, native.coords)
                    rows.append(dict(id=name, entry=entry, **row))
                except (RuntimeError, ValueError, OSError) as exc:
                    rows.append(dict(id=name, entry=entry, status='failed', error=f'{type(exc).__name__}: {exc}'))
                if step in (10000, 20000, 40000, 60000) and original.exists():
                    # Re-score saved FAST structures on the identical backbone atom set.
                    group = 'native_token_controls' if kind == 'controls' else kind
                    expected = fast_report[group]['rows'][number]
                    if (expected['entry']['entry_id'] != entry['entry_id']
                            or expected['entry']['source'] != entry['source']
                            or sha256(original/kind/name/'structure.npz') != expected['structure_sha256']):
                        raise ValueError('saved FAST comparison structure changed')
                    with np.load(original/kind/name/'structure.npz', allow_pickle=False) as data:
                        fast.append(dict(kind=kind, id=name, entry=entry,
                            structure_sha256=sha256(original/kind/name/'structure.npz'),
                            backbone_chemistry=backbone_chemistry(data['xyz'], data['atom_mask'], record['aa'], record['residue_id'])))
        sm = read(folder/'samples/manifest.json')
        if sm['fingerprint'] != cache.manifest['fingerprint']:
            raise ValueError('wrong tokenizer in joint samples')
        for number, item in enumerate(sm['samples']):
            name = f'sample_{number:04d}'
            path = folder/'samples'/item['file']
            if path.parent != folder/'samples' or sha256(path) != item['sha256']:
                raise ValueError('joint sample hash changed')
            with np.load(path, allow_pickle=False) as data:
                sample = dict(data)
            sequence = ''.join('ARNDCQEGHILKMFPSTWYV'[i] for i in sample['aa'])
            try:
                row = decode_sample(model, sample, destination/'joint'/name)
                joint.append(dict(id=name, sequence=sequence, **row))
            except (RuntimeError, ValueError, OSError) as exc:
                joint.append(dict(id=name, sequence=sequence, status='failed', error=f'{type(exc).__name__}: {exc}'))
            if step in (10000, 20000, 40000, 60000) and original.exists():
                if sha256(original/'joint'/name/'structure.npz') != fast_report['joint_raw_chemistry']['rows'][number]['structure_sha256']:
                    raise ValueError('saved FAST joint structure changed')
                fast_sample = (Path(comparison['fast_screen'])/f'control_seed{cfg["seed"]}'/'run/evaluation'/f'step_{step:07d}'/'samples'/item['file'])
                with np.load(fast_sample, allow_pickle=False) as source, np.load(original/'joint'/name/'structure.npz', allow_pickle=False) as data:
                    fast.append(dict(kind='joint', id=name, length=len(source['aa']),
                        structure_sha256=sha256(original/'joint'/name/'structure.npz'),
                        backbone_chemistry=backbone_chemistry(data['xyz'], data['atom_mask'], source['aa'], source['residue_id'])))
        failures = sum(r['status'] != 'ok' for r in conditional+controls+joint)
        report = dict(schema='native-dplm-control-geometry-v1', step=step, decoder=loaded,
                      decoder_fingerprint=cache.manifest['fingerprint'],
                      evaluation_complete_sha256=sha256(folder/'complete.json'),
                      conditional=dict(rows=conditional, summary=summarize(conditional, len(panel['entries']))),
                      native_token_controls=dict(rows=controls, summary=summarize(controls, len(panel['entries']))),
                      joint=dict(rows=joint, summary=summarize(joint, len(panel['joint_samples']), fold_requested=False)),
                      fast_backbone_comparison=fast, failures=failures,
                      limitations='Native decoder emits alanine-template N/CA/C/O/CB, not full side chains. Shared chemistry uses only N/CA/C/O; identity/backbone combined gate not applicable. Both raw and matched O-adjustment scores retained.')
        write(destination/'report.json', report)
        if failures:
            raise RuntimeError(f'{failures} failures retained in requested denominators')
        write(destination/'decode_complete.json', dict(report_sha256=sha256(destination/'report.json')))
        print(json.dumps(dict(step=step, conditional=report['conditional']['summary'], controls=report['native_token_controls']['summary'])), flush=True)


def refold(args, bundle):
    from fast_dplm.data import sha256
    from fast_dplm.generation_metrics import ca_geometry
    from fast_dplm.refolding import FoldCache, parse_fold, SETTINGS
    cache = FoldCache(args.root/'esmfold2_cache')
    for step in args.steps:
        folder = args.root/'evaluation'/bundle.name/f'step_{step:07d}'
        report = read(folder/'report.json')
        if sha256(folder/'report.json') != read(folder/'decode_complete.json')['report_sha256']:
            raise ValueError('decoded report changed')
        rows = []
        for item in report['joint']['rows']:
            row = dict(id=item['id'], sequence=item['sequence'], status='failed')
            try:
                if item['status'] != 'ok':
                    raise ValueError('decode failed')
                path = folder/'joint'/item['id']/'structure.npz'
                if sha256(path) != item['structure_sha256']:
                    raise ValueError('decoded coordinates changed')
                response, receipt = cache.get(item['sequence'])
                xyz, mask, confidence = parse_fold(response, item['sequence'])
                with np.load(path, allow_pickle=False) as data:
                    row.update(ca_geometry(data['xyz'][:, 1], xyz[:, 1]))
                row.update(status='ok', fold_receipt=receipt, mean_plddt=float(confidence.mean()),
                           structure_sha256=item['structure_sha256'])
                np.savez(folder/'joint'/item['id']/'refold.npz', xyz=xyz, atom_mask=mask, plddt=confidence)
            except (RuntimeError, ValueError, OSError) as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            write(folder/'refold_progress.json', dict(rows=rows, requested=report['joint']['summary']['requested']))
        result = dict(schema='native-dplm-control-esmfold2-v1', settings=SETTINGS,
                      decode_report_sha256=sha256(folder/'report.json'), rows=rows,
                      summary=summarize(rows, report['joint']['summary']['requested']),
                      sequence='generated AA only; one cached API prediction per sequence/settings; no API seed',
                      comparison='proper-rotation CA RMSD and CA-distance lDDT; not full-atom readiness')
        write(folder/'refold_report.json', result)
        if result['summary']['failed_or_missing']:
            raise RuntimeError('refolding failures retained in denominator')
        print(json.dumps(dict(step=step, summary=result['summary'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['decode', 'refold'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[10000, 20000, 40000, 60000])
    args = parser.parse_args()
    args.root, args.bundle = args.root.resolve(), args.bundle.resolve()
    sys.path[:0] = [str(args.bundle/'code/src'), str(args.root/'runtime/fast/src'), str(args.root/'runtime/dplm/scripts')]
    from fast_dplm.data import sha256
    for relative, expected in read(args.bundle/'bundle.json')['files'].items():
        if sha256(args.bundle/relative) != expected:
            raise ValueError('training/evaluation bundle changed')
    if args.command == 'decode':
        decode(args, args.bundle)
    else:
        refold(args, args.bundle)


if __name__ == '__main__':
    main()
