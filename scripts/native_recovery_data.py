"""Prepare a new, split-safe projection of the pinned author token dataset.

Author token provenance is explicit: the original encoder execution/checkpoint
is not attested by these parquet files. They are not relabeled FAST/native caches.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity(name):
    name=str(name).split('::')[-1].split('/')[-1]
    if name.startswith('AF-'):
        return 'af:'+name.split('-')[1]
    return 'pdb:'+name[:4].lower()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--joint',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=root/'data'
    out.mkdir(exist_ok=False)
    vocab=a.vocab.read_text().splitlines();vmap={v:i for i,v in enumerate(vocab)}
    assert len(vocab)==8229 and all(vmap[f'{i:04d}']==i+36 for i in range(8192))
    original=json.loads((a.joint/'splits/joint_core_v1_foldsplit.json').read_text())
    pruned=json.loads((a.joint/'splits/joint_core_v1_pruned_foldsplit.json').read_text())
    known={identity(x) for group in original.values() for x in group if not x.startswith('afdb_multimer::')}
    safe={identity(x) for x in pruned['train'] if not x.startswith('afdb_multimer::')}
    protected={identity(x) for group in ('val','test') for x in original[group] if not x.startswith('afdb_multimer::')}
    protected |= known-safe
    protected |= {identity(x) for x in (a.joint/'frozen/tcr3d_blacklist.txt').read_text().splitlines()}
    columns=['pdb_name','aa_seq','struct_seq','length','split','cluster','avg_plddt','plddt_std']
    tables={}
    for split in ('train','valid'):
        available=pq.read_schema(next((root/'author_dataset'/split).glob('*.parquet'))).names
        table=pq.read_table(root/'author_dataset'/split,columns=[c for c in columns if c in available]).to_pydict()
        for c in columns:
            table.setdefault(c,[None]*len(table['pdb_name']))
        tables[split]=table
    val=tables['valid'];tr=tables['train']
    val_ids={identity(x) for x in val['pdb_name']}
    val_sequences=set(val['aa_seq'])
    # Protect all existing known held-out entry identities, and whole author
    # clusters touching them. This is explicitly a projection, not exact replay.
    excluded_clusters=set()
    reasons={}
    for i,name in enumerate(tr['pdb_name']):
        why=[]
        if identity(name) in protected: why.append('existing_split_or_tcr3d_protection')
        if identity(name) in val_ids: why.append('author_validation_entry')
        if tr['aa_seq'][i] in val_sequences: why.append('author_validation_exact_sequence')
        if why:
            reasons[i]=why
            if tr['cluster'][i] is not None: excluded_clusters.add(str(tr['cluster'][i]))
    for i,c in enumerate(tr['cluster']):
        if str(c) in excluded_clusters:
            reasons.setdefault(i,[]).append('author_cluster_propagation')
    manifests={}
    for split,table in tables.items():
        folder=out/split;folder.mkdir()
        metadata=[];codes=[];aas=[];offsets=[0]
        count=collections.Counter()
        for i,name in enumerate(table['pdb_name']):
            sequence=table['aa_seq'][i]
            words=table['struct_seq'][i].split(',')
            if len(sequence)!=len(words) or len(sequence)!=table['length'][i]:
                raise ValueError(f'unaligned released row {split}:{i}')
            if any(len(w)!=4 or not w.isdecimal() or not 0<=int(w)<8192 for w in words):
                raise ValueError(f'invalid released codes {split}:{i}')
            if any(c not in vmap or vmap[c]>=33 for c in sequence):
                raise ValueError(f'unknown released AA {split}:{i}')
            if split=='train' and i in reasons:
                continue
            if split=='train' and table['split'][i]=='afdb_swissprot':
                assert table['avg_plddt'][i]>85 and table['plddt_std'][i]<15
            codes.extend(map(int,words));aas.extend(vmap[c] for c in sequence);offsets.append(len(codes))
            row=dict(id=name,author_index=i,length=len(sequence),source=table['split'][i],
                     cluster=str(table['cluster'][i]),sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
            metadata.append(row);count[table['split'][i]]+=1
        np.save(folder/'codes.npy',np.asarray(codes,dtype=np.uint16))
        np.save(folder/'aa.npy',np.asarray(aas,dtype=np.uint8))
        np.save(folder/'offsets.npy',np.asarray(offsets,dtype=np.int64))
        (folder/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
        manifest=dict(schema='released-dplm-token-projection-v1',split=split,entries=len(metadata),
            residues=len(codes),sources=dict(count),vocab_sha256=sha(a.vocab),vocabulary_size=8192,
            author_dataset_revision='eafc13c5b50f51543dae8d4d2895332d4f8e5261',
            author_encoder_checkpoint='not provided in parquet; decoder compatibility assessed separately',
            files={f.name:sha(f) for f in sorted(folder.iterdir())})
        write(folder/'manifest.json',manifest);manifests[split]=manifest
        print(json.dumps(dict(split=split,entries=len(metadata),residues=len(codes),sources=dict(count))),flush=True)
    excluded=[dict(author_index=i,id=tr['pdb_name'][i],cluster=str(tr['cluster'][i]),reasons=why) for i,why in sorted(reasons.items())]
    write(out/'exclusions.json',excluded)
    # Panel selection is independent of structure availability or model scores.
    eligible=[i for i,s in enumerate(val['aa_seq']) if len(s)<=512 and set(s)<=set('ARNDCQEGHILKMFPSTWYV')]
    eligible=sorted(eligible,key=lambda i:(len(val['aa_seq'][i]),val['pdb_name'][i]))
    selected=[];rng=np.random.default_rng(20261008)
    for group in np.array_split(eligible,4):
        selected.extend(sorted(rng.choice(group,size=8,replace=False).tolist()))
    panel=dict(schema='native-recovery-generation-panel-v1',selection_seed=20261008,
        conditional=[dict(id=f'entry_{j:04d}',index=i,name=val['pdb_name'][i],sequence=val['aa_seq'][i],
                          seed=810000+j,length=len(val['aa_seq'][i])) for j,i in enumerate(selected)],
        joint=[dict(id=f'sample_{j:04d}',length=n,seed=820000+j) for j,n in enumerate([100,200,300,400,500]*4)],
        sampling=dict(conditional_steps=100,joint_steps=500,conditional_strategy='argmax',
                      joint_strategy='annealing@2.0:0.1',joint_unmasking='stochastic1.0',top_p=.95))
    write(root/'panel.json',panel)
    report=dict(schema='native-recovery-data-audit-v1',passed=True,original_train=len(tr['pdb_name']),
        excluded_train=len(reasons),retained_train=manifests['train']['entries'],validation=len(val['pdb_name']),
        exclusion_counts=dict(collections.Counter(x for r in reasons.values() for x in r)),
        protected_id_count=len(protected),protected_author_clusters=len(excluded_clusters),
        train_validation_exact_entry_overlap=0,train_validation_exact_sequence_overlap=0,
        note='Existing held-out/pruned-out identities and touching author clusters protected. This is a new dataset projection. Remote homology and structural fold separation are not established; do not claim unseen-fold generalization.',
        inputs={str(f):sha(f) for f in [a.vocab,a.joint/'splits/joint_core_v1_foldsplit.json',
            a.joint/'splits/joint_core_v1_pruned_foldsplit.json',a.joint/'frozen/tcr3d_blacklist.txt',
            *sorted((root/'author_dataset').rglob('*.parquet'))]},
        train_manifest_sha256=sha(out/'train/manifest.json'),val_manifest_sha256=sha(out/'valid/manifest.json'),
        panel_sha256=sha(root/'panel.json'))
    # Recheck assertions on actual retained membership, not just assigned zeros.
    keep=[i for i in range(len(tr['pdb_name'])) if i not in reasons]
    assert not ({identity(tr['pdb_name'][i]) for i in keep}&(protected|val_ids))
    assert not ({tr['aa_seq'][i] for i in keep}&val_sequences)
    write(root/'data_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'}),flush=True)


if __name__=='__main__':
    main()
