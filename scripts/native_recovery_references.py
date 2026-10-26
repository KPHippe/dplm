"""Freeze sequence-matched CAMEO coordinates for a preselected panel."""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from fast.data.constants import ATOM37_INDEX
from fast.dplm_bridge import sha256


def extract_reference(item,path):
    from biotite.structure.io.pdbx import CIFFile,get_structure
    from biotite.structure.info import one_letter_code
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal,SubstitutionMatrix
    cif=CIFFile.read(path);scheme=cif.block['pdbx_poly_seq_scheme']
    atoms=get_structure(cif,model=1,use_author_fields=False)
    fields={k:scheme[k].as_array(str) for k in ('asym_id','seq_id','mon_id','pdb_strand_id')}
    author_chain=item['name'].rsplit('_',1)[1]
    chains=list(dict.fromkeys(fields['asym_id'][fields['pdb_strand_id']==author_chain]))
    if not chains:raise ValueError('author chain absent in polymer sequence scheme')
    matches=[]
    for chain in chains:
        rows=np.flatnonzero(fields['asym_id']==chain)
        ids=fields['seq_id'][rows].astype(int)
        sequence=''.join(one_letter_code(x) or 'X' for x in fields['mon_id'][rows])
        target=item['sequence'];start=sequence.find(target)
        overrides=[]
        # The author's 7u04_H starts E, while deposited residue1 is PCA and
        # CCD maps it to Q. Its backbone position is unambiguous. Record this
        # one explicit identity convention; never infer full-side-chain parity.
        if (item['name']=='7u04_H' and target.startswith('E') and sequence.startswith('Q')
                and any(str(x.res_name)=='PCA' for x in atoms[(atoms.chain_id==chain)&(atoms.res_id==ids[0])])):
            overrides.append(dict(label_seq_id=int(ids[0]),component='PCA',ccd_letter='Q',author_letter='E',scope='backbone coordinate correspondence only'))
            sequence='E'+sequence[1:];start=sequence.find(target)
        ambiguity=False
        if start>=0:
            mapping=np.arange(start,start+len(target))
        else:
            s1,s2=ProteinSequence(target),ProteinSequence(sequence)
            alphabet=s1.get_alphabet()
            matrix=SubstitutionMatrix(alphabet,alphabet,np.where(np.eye(len(alphabet)),2,-1000).astype(int))
            alignments=align_optimal(s1,s2,matrix,gap_penalty=(-10,-1),terminal_penalty=False,max_number=2)
            maps=[]
            for alignment in alignments:
                trace=alignment.trace;trace=trace[trace[:,0]>=0]
                if len(trace)!=len(target) or (trace[:,1]<0).any():continue
                if any(target[i]!=sequence[j] for i,j in trace):continue
                maps.append(trace[:,1])
            if not maps:continue
            ambiguity=len(maps)>1 and not np.array_equal(maps[0],maps[1])
            if ambiguity:raise ValueError('ambiguous residue correspondence; explicit review required')
            mapping=maps[0]
        label_ids=ids[mapping]
        xyz=np.zeros((len(target),37,3),dtype=np.float32);mask=np.zeros((len(target),37),dtype=bool)
        lookup={int(n):i for i,n in enumerate(label_ids)}
        for atom in atoms[atoms.chain_id==chain]:
            i=lookup.get(int(atom.res_id));j=ATOM37_INDEX.get(str(atom.atom_name))
            if i is not None and j is not None:
                xyz[i,j]=atom.coord;mask[i,j]=True
        if mask[:,1].mean()<.9:raise ValueError('less than90% observed CA coverage')
        alphabet_string='ARNDCQEGHILKMFPSTWYV'
        data=dict(xyz=xyz,atom_mask=mask,aa=np.array([alphabet_string.index(c) for c in target]),
                  residue_id=label_ids,chain_end=label_ids==ids[-1])
        meta=dict(author_chain=author_chain,label_chain=str(chain),polymer_label_seq_ids=label_ids.tolist(),
                  observed_ca=int(mask[:,1].sum()),requested_ca=len(target),modified_residues=[str(x) for x in fields['mon_id'][rows[mapping]] if str(x) not in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL')],alignment_ambiguous=ambiguity)
        meta['identity_convention_overrides']=overrides
        matches.append((data,meta))
    if len(matches)!=1:raise ValueError('polymer chain/sequence matching not unique')
    return matches[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root;folder=root/'references';folder.mkdir(exist_ok=False)
    panel=json.loads((root/'panel.json').read_text());rows=[]
    for item in panel['conditional']:
        pdb=item['name'].split('_')[0].lower()
        result=dict(id=item['id'],name=item['name'],status='failed')
        try:
            path=folder/(pdb+'.cif')
            if not path.exists():
                cached=root/'preparation/references_failed_975'/(pdb+'.cif')
                if not cached.exists():cached=root/'preparation/references_failed_974'/(pdb+'.cif')
                if cached.exists():path.write_bytes(cached.read_bytes())
                else:urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pdb.upper()}.cif',path)
            data,metadata=extract_reference(item,path)
            target=folder/(item['id']+'.npz');np.savez(target,**data)
            result.update(status='ok',source=str(path),source_sha256=sha256(path),
                          reference_sha256=sha256(target),**metadata)
        except (OSError,ValueError,IndexError) as exc:
            result['error']=str(exc)
        rows.append(result)
        print(json.dumps(result),flush=True)
    report=dict(requested=len(panel['conditional']),matched=sum(r['status']=='ok' for r in rows),rows=rows,
        policy='Preselected panel; author chain and polymer sequence scheme. CCD parent one-letter identities include modified residues. Exact matching target letters, unique alignment if noncontiguous. Missing coordinate masks retained, >=90% observed CA required. No coordinate imputation.')
    (root/'reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['matched']!=report['requested']:
        raise RuntimeError('reference correspondence incomplete; do not silently drop panel requests')


if __name__=='__main__':main()
