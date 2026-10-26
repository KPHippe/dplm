import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module(name):
    path=Path(__file__).parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def vocab():
    aa=['<cls_aa>','<pad>','<eos_aa>','<unk_aa>']+list('LAGVSERTIDPKQNFYMHWCXB UZO'.replace(' ',''))+['.','-','<null_1>','<mask_aa>']
    assert len(aa)==33
    return aa+['<cls_struct>','<eos_struct>','<unk_struct>']+[f'{i:04d}' for i in range(8192)]+['<mask_struct>']


def test_vocabulary_permutation_preserves_semantics_in_both_directions():
    v=vocab();mapping=module('released_dplm2_loader').vocabulary_permutation(v)
    for i in range(8192):
        assert v[mapping[33+i]]==f'{i:04d}'
    assert [v[i] for i in mapping[-4:]]==['<cls_struct>','<eos_struct>','<mask_struct>','<unk_struct>']
    inverse=np.argsort(mapping)
    assert np.array_equal(np.array(mapping)[inverse],np.arange(8229))
    with pytest.raises(ValueError):
        module('released_dplm2_loader').vocabulary_permutation(v[:-1]+[v[0]])


def test_native_unpack_keeps_conditioning_and_rejects_wrong_domains():
    c=module('released_dplm2_control');v=vocab()
    model=SimpleNamespace(device='cpu',struct_bos_id=33,struct_eos_id=34,aa_bos_id=0,aa_eos_id=2,
        tokenizer=SimpleNamespace(all_tokens=v,_token_to_id={s:i for i,s in enumerate(v)}))
    rec=dict(aa=np.array([0,1,2],dtype=np.uint8),residue_id=np.array([5,6,7]),chain_end=np.array([0,0,1]))
    ids=c.input_tokens(model,rec)
    result=c.unpack(model,ids,rec,True)
    assert np.array_equal(result['fast_id'],[0,0,0]) and np.array_equal(result['aa'],rec['aa'])
    bad=ids.clone();bad[0,1]=0
    with pytest.raises(ValueError,match='structure'):
        c.unpack(model,bad,rec,True)
    bad=ids.clone();bad[0,6]=v.index('G')
    with pytest.raises(ValueError,match='conditioning'):
        c.unpack(model,bad,rec,True)
    assert torch.equal(ids,c.input_tokens(model,rec))


def test_request_counts_and_native_sequence_whitelist():
    c=module('released_dplm2_control')
    panel=dict(generation_validation=dict(entries=[dict(seed=i,length=60,index=i) for i in range(32)]),
               samples=[dict(seed=i,length=100) for i in range(20)])
    rs=c.requests(panel)
    assert len(rs)==124
    assert sum(r['steps']==500 for r in rs)==20
    class Cache:
        def __getitem__(self,i):
            return dict(aa=np.array([0,1]),fast_id=np.array([7999,7999]),residue_id=np.array([1,2]),chain_end=np.array([0,1]))
    rec=c.get_record(rs[0],Cache())
    assert not rec['fast_id'].any()
