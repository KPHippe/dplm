"""Target-free FAST rendering and paired backbone/full-atom scoring."""
from pathlib import Path
from contextlib import chdir
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fast_recovery_common import Corpus, read, write, sha


def load_frozen_renderer(root, lineage, device, loader):
    """Resolve saved relative asset paths against the candidate's frozen FAST.

    Keep the original configuration and lineage hashes. Each decoder process
    gets a private compatibility directory; caller CWD is restored on failure.
    """
    source = (root / 'code/fast_src/fast').resolve(strict=True)
    with TemporaryDirectory(prefix='fast-render-assets-') as temporary:
        work = Path(temporary)
        (work / 'src').mkdir()
        (work / 'src/fast').symlink_to(source, target_is_directory=True)
        with chdir(work):
            return loader(lineage, device)


def backbone_scorer(root):
    """Run the released oxygen function without importing legacy training code."""
    from released_dplm2_loader import extract
    from fast.analysis.generated_chemistry import chemistry
    scope = dict(torch=torch)
    extract(root / 'code/src/byprot/datamodules/pdb_dataset/all_atom.py', ['adjust_oxygen_pos'], scope)
    def score(xyz, mask, aa, residue_id):
        common = np.zeros_like(mask, dtype=bool)
        common[:, [0, 1, 2, 4]] = True
        if not mask[common].all():
            raise ValueError('incomplete N/CA/C/O correspondence')
        chain = np.zeros(len(aa), dtype=np.int32)
        raw = chemistry(xyz, common, aa, residue_id, chain)
        adjusted = scope['adjust_oxygen_pos'](torch.as_tensor(xyz).clone(), torch.ones(len(aa), dtype=torch.bool)).numpy()
        return dict(raw=raw, matched_oxygen_adjustment=chemistry(adjusted, common, aa, residue_id, chain))
    return score


def decode(args):
    from encode_fast_recovery import load_verified, render
    from fast_recovery_eval import record
    from fast.analysis.generated_chemistry import chemistry
    from fast_dplm.generation_metrics import ca_geometry
    root = args.root
    backbone_chemistry = backbone_scorer(root)
    cfg = read(root / 'config.json')
    lineage = read(root / 'decoder_lineage.json')
    if sha(root / 'decoder_lineage.json') != cfg['decoder_lineage_sha256']:
        raise ValueError('FAST decoder lineage changed')
    if args.controls:
        requests = [dict(kind='control', **x) for x in read(root / 'panel.json')['conditional']]
        generated = {}
    else:
        manifest = read(args.folder / 'generation.json')
        requests = manifest['requests'] + [x | {'kind': 'control'} for x in manifest['requests'] if x['kind'] == 'conditional']
        generated = {(r['request']['kind'], r['request']['id']): r for r in manifest['rows']}
    tasks = [r for i, r in enumerate(requests) if i % args.world == args.rank]
    args.folder.mkdir(parents=True, exist_ok=True)
    if not tasks:
        write(args.folder / f'decode_rank_{args.rank}.json', [])
        return
    torch.cuda.set_device(args.rank)
    model = load_frozen_renderer(root, lineage, 'cuda', load_verified)
    vocab = (Path(cfg['vocab_directory']) / 'vocab.txt').read_text().splitlines()
    corpus = Corpus(root / 'data/valid', vocab)
    rows = []
    for item in tasks:
        row = dict(id=item['id'], kind=item['kind'], request=item, status='failed')
        try:
            ref = None
            if item['kind'] != 'joint':
                ref = dict(np.load(root / 'references' / (item['id'] + '.npz')))
            if item['kind'] == 'control':
                sample = record(root, corpus, item, True)
                lo, hi = corpus.offsets[item['index']:item['index']+2]
                valid = np.array(corpus.valid[lo:hi])
            else:
                source = generated[item['kind'], item['id']]
                if source['status'] != 'ok':
                    raise ValueError('generation failed')
                tokenpath = args.folder / 'tokens' / item['kind'] / (item['id'] + '.npz')
                if sha(tokenpath) != source['tokens_sha256']:
                    raise ValueError('generated tokens changed')
                sample = dict(np.load(tokenpath))
                valid = np.ones(len(sample['aa']), dtype=bool)
                row.update(sequence=source['sequence'], sequence_metrics=source['sequence_metrics'])
                if item['kind'] == 'conditional':
                    expected = record(root, corpus, item, True)
                    for k in ('aa', 'residue_id', 'chain_end'):
                        if not np.array_equal(sample[k], expected[k]):
                            raise ValueError('conditioning/topology changed')
            dest = args.folder / 'decoded' / item['kind'] / item['id']
            decoded = render(model, sample['fast_id'], valid, sample['residue_id'],
                             sample['chain_end'], dest, lineage['fingerprint'])
            agrees = decoded['aa'][valid] == sample['aa'][valid]
            row.update(status='ok', length=len(valid), structure_sha256=sha(dest / 'structure.npz'),
                aa_agreement=float(agrees.mean()), exact_sequence_agreement=bool(agrees.all()),
                missing_structure_slots=int((~valid).sum()),
                full_atom_chemistry=chemistry(decoded['xyz'], decoded['atom_mask'], decoded['aa'],
                    sample['residue_id'], decoded['chain_id']),
                backbone_chemistry=backbone_chemistry(decoded['xyz'][valid], decoded['atom_mask'][valid],
                    sample['aa'][valid], sample['residue_id'][valid]))
            if ref is not None:
                observed = ref['atom_mask'][:, 1] & decoded['atom_mask'][:, 1]
                row.update(ca_geometry(decoded['xyz'][observed, 1], ref['xyz'][observed, 1]))
                row.update(native_ca_scored=int(observed.sum()), native_ca_requested=len(observed))
        except (ValueError, RuntimeError, OSError) as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    write(args.folder / f'decode_rank_{args.rank}.json', rows)
    write(args.folder / f'decoder_provenance_rank_{args.rank}.json', lineage)
