"""Check released data/crop alignment and exact cursor restoration without a GPU."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from native_recovery_common import read,write,released,Corpus,epoch_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();cfg=read(root/'config.json')
    torch.set_num_threads(2)
    api=released(root)
    tokenizer=api.tokenizer_class.from_pretrained(cfg['vocab_directory'])
    corpus=Corpus(root/'data/train',tokenizer.all_tokens)
    valid=Corpus(root/'data/valid',tokenizer.all_tokens)
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group('gloo',init_method='file://'+temp+'/group',rank=0,world_size=1)
        dataset,batches,collater=epoch_data(api,corpus,tokenizer,0,cfg['max_tokens'])
        assert all(len(b)*min(512,max(dataset.data.lengths[b]))<=cfg['max_tokens'] for b in batches)
        assert sorted(i for b in batches for i in b)==list(range(len(corpus.rows)))
        # Independent correspondence check: reconstruct the crop start from its
        # saved RNG, then compare every residue/code with the original mmap row.
        selected=np.argsort(corpus.lengths)[-16:].tolist()+[b[0] for b in batches[:16]]
        for index in selected:
            original=corpus.record(index);n=min(512,dataset.data.lengths[index])
            before=np.random.get_state()
            start=int(np.random.choice(original['length']-n)) if original['length']>n else 0
            np.random.set_state(before);item=dataset[index]
            assert item['aatype_tokens']=='<cls_aa>'+original['aa_seq'][start:start+n]+'<eos_aa>'
            assert item['struct_tokens']=='<cls_struct>'+''.join(original['struct_seq'].split(',')[start:start+n])+'<eos_struct>'
        # Rebuilding the epoch must reproduce the full batch index stream, and
        # restoring RNG/cursor must reproduce the next collated crop exactly.
        saved=np.random.get_state();cursor=7
        expected=collater([dataset[i] for i in batches[cursor]])
        restored,again,collate_again=epoch_data(api,corpus,tokenizer,0,cfg['max_tokens'])
        assert batches==again
        np.random.set_state(saved);actual=collate_again([restored[i] for i in again[cursor]])
        for key in expected:
            if isinstance(expected[key],dict):
                for field in expected[key]:
                    assert torch.equal(expected[key][field],actual[key][field]),(key,field)
            else:
                assert expected[key]==actual[key],key
        dist.destroy_process_group()
    panel=read(root/'panel.json');coverage=[]
    for item in panel['conditional']:
        ref=dict(np.load(root/'references'/f'{item["id"]}.npz'))
        sequence=''.join('ARNDCQEGHILKMFPSTWYV'[x] for x in ref['aa'])
        assert sequence==valid.record(item['index'])['aa_seq']
        assert ref['xyz'].shape==(len(sequence),37,3)
        assert np.isfinite(ref['xyz'][ref['atom_mask']]).all()
        assert (np.diff(ref['residue_id'])>0).all()
        assert ref['atom_mask'][:,1].mean()>=.9
        coverage.append(dict(id=item['id'],observed_ca=int(ref['atom_mask'][:,1].sum()),requested_ca=len(sequence)))
    result=dict(passed=True,entries=len(corpus.rows),batches=len(batches),
        released_cropped_rows_checked=len(selected),epoch_membership_and_budget=True,
        rebuilt_batches_and_resumed_crop='identical',reference_coverage=coverage)
    write(root/'cpu_verification.json',result);print(result)


if __name__=='__main__':main()
