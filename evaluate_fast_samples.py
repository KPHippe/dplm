"""Independent ESMFold and corrected heavy-atom diagnostics on decoded FAST samples.

Run in Slurm using the DPLM overlay and FAST/src on PYTHONPATH. Refolds the DPLM
sequence; FAST's predicted identities remain untouched and agreement is reported.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import EsmForProteinFolding
from transformers.models.esm.openfold_utils import atom14_to_atom37, residue_constants

from fast.analysis.generated_chemistry import aligned_ca_rmsd, chemistry
from fast.analysis import generated_chemistry
from fast_dplm.data import sha256


def write_json(path, obj):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2)+'\n')
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoded', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recycles', type=int, default=3)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model, loading = EsmForProteinFolding.from_pretrained(a.pretrained, local_files_only=True, output_loading_info=True)
    unused = {'esm.contact_head.regression.bias', 'esm.contact_head.regression.weight'}
    if set(loading['missing_keys']) - unused or loading['unexpected_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'ESMFold checkpoint mismatch: {loading}')
    model.esm.contact_head = None
    model.cuda().eval()
    model.trunk.set_chunk_size(64)
    protocol = dict(schema='fast-dplm-generated-eval-v1', evaluator_sha256=sha256(__file__),
        esmfold_path=str(a.pretrained.resolve()), esmfold_config_sha256=sha256(a.pretrained/'config.json'),
        esmfold_weights_sha256=sha256(a.pretrained/'pytorch_model.bin'), recycles=a.recycles,
        chemistry_sha256=sha256(generated_chemistry.__file__),
        conformer_sha256=sha256(Path(generated_chemistry.__file__).parents[1]/'data/canonical_conformers.pt'),
        decoded_manifest_sha256=sha256(a.decoded/'report.json'), plddt_scale='0-100',
        sequence='DPLM', comparison='proper-rotation CA Kabsch RMSD; no optimized TM score',
        chemistry='heavy-atom emitted pairs; exclude covalent graph distance <=2; no relaxation')
    rows = []
    expected = json.loads((a.decoded/'report.json').read_text())[:a.limit]
    for declared in expected:
        sample = a.decoded / declared['name']
        if sample.parent != a.decoded:
            raise ValueError('invalid sample name')
        diagnostics = json.loads((sample/'diagnostics.json').read_text())
        if sha256(sample/'structure.npz') != diagnostics['structure_sha256']:
            raise ValueError('decoded structure integrity failure')
        seq = ''.join((sample/'sequence_dplm.fasta').read_text().splitlines()[1:])
        identity = dict(protocol=protocol, structure_sha256=diagnostics['structure_sha256'], sequence=seq)
        result = a.out / f'{sample.name}.json'
        if result.exists():
            old = json.loads(result.read_text())
            if old['identity'] != identity:
                raise ValueError('evaluation output belongs to another input/protocol')
            rows.append(old)
            continue
        with np.load(sample/'structure.npz', allow_pickle=False) as arr:
            arrays = {k:arr[k] for k in arr.files}
        row = dict(name=sample.name, identity=identity, aa_agreement=diagnostics['aa_agreement'],
            exact_sequence_agreement=diagnostics['exact_sequence_agreement'],
            chemistry=chemistry(arrays['xyz'], arrays['atom_mask'], arrays['aa'], arrays['residue_id'], arrays['chain_id']))
        try:
            with torch.inference_mode():
                ids = torch.tensor([[residue_constants.restype_order[c] for c in seq]], device='cuda')
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), num_recycles=a.recycles)
                xyz = atom14_to_atom37(output['positions'][-1], output)[0].float().cpu().numpy()
                present = output['atom37_atom_exists'][0].bool().cpu().numpy()
                confidence = 100 * output['plddt'][0].float().cpu().numpy()
                pdb = model.output_to_pdb(dict(output, plddt=100*output['plddt']))[0]
            if len(xyz) != len(arrays['xyz']) or not arrays['atom_mask'][:,1].all() or not present[:,1].all():
                raise ValueError('CA correspondence incomplete')
            score = aligned_ca_rmsd(arrays['xyz'][:,1], xyz[:,1])
            row.update(status='ok', refold_ca_rmsd_a=score, refold_mean_plddt=float(confidence[present].mean()),
                       backbone_consistent_2a=score<2., identity_and_backbone_consistent=score<2. and diagnostics['exact_sequence_agreement'])
            (a.out / f'{sample.name}.refold.pdb').write_text(pdb)
            del output
        except (RuntimeError, ValueError) as error:
            row.update(status='refold_failed', error=f'{type(error).__name__}: {error}')
            torch.cuda.empty_cache()
        write_json(result, row)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('identity','chemistry')}), flush=True)
    summary = dict(protocol=protocol, requested=len(rows), refolded=sum(r['status']=='ok' for r in rows),
        exact_sequence_agreement=sum(r['exact_sequence_agreement'] for r in rows),
        identity_and_backbone_consistent=sum(r.get('identity_and_backbone_consistent',False) for r in rows), rows=rows)
    write_json(a.out/'report.json', summary)
    if not rows or any(r['status']!='ok' for r in rows):
        raise RuntimeError('incomplete refolding panel; failures retained in report and denominator')


if __name__ == '__main__':
    main()
