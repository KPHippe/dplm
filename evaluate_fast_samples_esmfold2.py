"""Score untouched FAST decodes against shared sequence-only ESMFold2 predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from fast.analysis import generated_chemistry
from fast.analysis.generated_chemistry import aligned_ca_rmsd, chemistry
from fast.data.constants import AA_ALPHABET
from fast_dplm import refolding
from fast_dplm.data import sha256
from fast_dplm.refolding import FoldCache, parse_fold, write_json


def evaluate(decoded, out, cache):
    decoded, out = Path(decoded), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema='fast-dplm-generated-eval-esmfold2-v1',
        evaluator_sha256=sha256(__file__), client_sha256=sha256(refolding.__file__),
        endpoint=refolding.ENDPOINT, settings=refolding.SETTINGS,
        chemistry_sha256=sha256(generated_chemistry.__file__),
        conformer_sha256=sha256(Path(generated_chemistry.__file__).parents[1]/'data/canonical_conformers.pt'),
        decoded_manifest_sha256=sha256(decoded/'report.json'),
        sequence='DPLM-generated AA; sequence-only; no MSA or coordinate conditioning',
        plddt_scale='0-100; arithmetic mean of API per-residue 0-1 confidence times 100',
        comparison='proper-rotation CA Kabsch RMSD; no optimized TM score',
        chemistry='heavy-atom emitted pairs; exclude covalent graph distance <=2; no relaxation',
        stochasticity='one cached API prediction per sequence/settings shared across renderers; API has no seed',
        correspondence='API positional atom37 ordering; response may not echo sequence',
        model_provenance='versioned API name; provider weight hash unavailable')
    expected = json.loads((decoded/'report.json').read_text())
    names = [r['name'] for r in expected]
    if not names or len(names) != len(set(names)):
        raise ValueError('empty or duplicate decoded sample panel')
    rows = []
    for declared in expected:
        if Path(declared['name']).name != declared['name'] or declared['name'] in ('.', '..'):
            raise ValueError('invalid sample name')
        sample = decoded/declared['name']
        diagnostics = json.loads((sample/'diagnostics.json').read_text())
        if diagnostics != declared or sha256(sample/'structure.npz') != diagnostics['structure_sha256']:
            raise ValueError('decoded structure/diagnostics integrity failure')
        sequence = ''.join((sample/'sequence_dplm.fasta').read_text().splitlines()[1:])
        with np.load(sample/'structure.npz', allow_pickle=False) as data:
            arrays = dict(data)
        fast_sequence = ''.join(AA_ALPHABET[int(i)] for i in arrays['aa'])
        if (len(sequence) != len(fast_sequence) or len(sequence) != diagnostics['length']
                or not np.all(arrays['chain_id'] == arrays['chain_id'][0])
                or not np.all(np.diff(arrays['residue_id']) == 1)):
            raise ValueError('decoded sequence/topology correspondence incomplete')
        agreement = sum(a == b for a, b in zip(sequence, fast_sequence, strict=True))/len(sequence)
        exact = sequence == fast_sequence
        if not np.isclose(agreement, diagnostics['aa_agreement']) or exact != diagnostics['exact_sequence_agreement']:
            raise ValueError('decoded sequence/identity report mismatch')
        identity = dict(protocol=protocol, structure_sha256=diagnostics['structure_sha256'], sequence=sequence)
        result = out/f'{sample.name}.json'
        if result.exists() and json.loads(result.read_text())['identity'] != identity:
            raise ValueError('evaluation output belongs to another input/protocol')
        # Recompute cheap metrics on resume; successful folding is cached independently.
        row = dict(name=sample.name, identity=identity, aa_agreement=agreement,
            exact_sequence_agreement=exact,
            chemistry=chemistry(arrays['xyz'], arrays['atom_mask'], arrays['aa'],
                                arrays['residue_id'], arrays['chain_id']))
        try:
            if not arrays['atom_mask'][:, 1].all():
                raise ValueError('FAST CA correspondence incomplete')
            response, receipt = cache.get(sequence)
            xyz, present, confidence = parse_fold(response, sequence)
            score = aligned_ca_rmsd(arrays['xyz'][:, 1], xyz[:, 1])
            row.update(status='ok', refold_ca_rmsd_a=score, refold_mean_plddt=float(confidence.mean()),
                refold_ptm=response.get('ptm'), fold_receipt=receipt,
                warning_messages=response.get('warning_messages'),
                backbone_consistent_2a=score < 2., identity_and_backbone_consistent=score < 2. and exact)
            # Full precision coordinates with actual returned occupancy; FAST files are never changed.
            np.savez(out/f'{sample.name}.refold.npz', xyz=xyz, atom_mask=present,
                     sequence=np.asarray(sequence), plddt=confidence)
        except (RuntimeError, ValueError) as error:
            row.update(status='refold_failed', error=f'{type(error).__name__}: {error}')
        write_json(result, row)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('identity', 'chemistry')}), flush=True)
    summary = dict(protocol=protocol, requested=len(expected), refolded=sum(r['status'] == 'ok' for r in rows),
        exact_sequence_agreement=sum(r['exact_sequence_agreement'] for r in rows),
        identity_and_backbone_consistent=sum(r.get('identity_and_backbone_consistent', False) for r in rows), rows=rows)
    write_json(out/'report.json', summary)
    if any(r['status'] != 'ok' for r in rows):
        raise RuntimeError('incomplete ESMFold2 panel; failures retained in report and denominator')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoded', 'out', 'fold-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.decoded, args.out, FoldCache(args.fold_cache))


if __name__ == '__main__':
    main()
