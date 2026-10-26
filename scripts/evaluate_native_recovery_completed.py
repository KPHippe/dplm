"""Finish saved recovery refolds and evaluate released weights on the same panel.

Write only a new evaluation overlay; frozen training code/artifacts are inputs.
The existing native evaluator supplies generation, decoding and refold semantics.
"""
import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from native_recovery_common import read,write,sha,verify,Corpus,Objective
from native_recovery_eval import evaluate,refold
from released_dplm2_loader import load,extract


def verify_overlay(out):
    manifest=read(out/'bundle.json')
    for name,digest in manifest['files'].items():
        if sha(out/name)!=digest:raise ValueError(f'evaluation code changed: {name}')
    plan=read(out/'plan.json')
    verify(Path(plan['training_root']),plan['training_bundle_sha256'])
    return plan


def released_generation(root,out,plan):
    import os
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==8
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    if rank==0:verify_overlay(out)
    dist.barrier();torch.manual_seed(42);np.random.seed(42)
    model,_,proof=load(root/'code/src',plan['released_checkpoint'],with_port=False)
    for key in ('checkpoint_sha256','config_sha256','vocab_sha256'):
        if proof[key]!=plan['released_'+key]:raise ValueError(f'released asset changed: {key}')
    write(out/'released/proof'/f'rank_{rank}.json',proof)
    model.cuda()
    # Only evaluator dependencies are extracted here. The released loader above
    # strictly checks every tensor of the actual released generator checkpoint.
    scope=dict(torch=torch,nn=nn,F=F,np=np)
    extract(root/'code/src/byprot/datamodules/dataset/tokenized_protein.py',['DPLM2Collater'],scope)
    criterion_scope=dict(scope,Tensor=torch.Tensor)
    extract(root/'code/src/byprot/modules/cross_entropy.py',
            ['label_smoothed_nll_loss','StructAARDMCrossEntropyLoss'],criterion_scope)
    criterion=criterion_scope['StructAARDMCrossEntropyLoss'](label_smoothing=0.,ignore_index=model.pad_id)
    objective=Objective(model,criterion)
    api=SimpleNamespace(collater=scope['DPLM2Collater'])
    corpus=Corpus(root/'data/valid',model.tokenizer.all_tokens)
    evaluate(model,objective,api,corpus,root,str(out/'released'),0,sampling=True)
    dist.barrier();dist.destroy_process_group()


def finish_refolds(root,out,variant):
    plan=verify_overlay(out)
    steps=plan['milestones'] if variant=='recovery' else [0]
    results=[]
    for step in steps:
        source=(root/'production' if variant=='recovery' else out/'released')/'evaluation'/f'step_{step:07d}'
        geometry=read(source/'geometry.json')
        if read(source/'complete.json')['geometry_sha256']!=sha(source/'geometry.json'):
            raise ValueError('source geometry changed')
        if (source/'refold.json').exists():
            report=read(source/'refold.json')
            if report['geometry_sha256']!=sha(source/'geometry.json'):raise ValueError('refold geometry changed')
            if report['summary']['scored']!=20:raise ValueError('prior incomplete refold needs an explicit retry overlay')
            destination=source
            reused=True
        else:
            destination=out/'refolding'/variant/f'step_{step:07d}'
            destination.mkdir(parents=True,exist_ok=False)
            for name in ('geometry.json','complete.json'):shutil.copy2(source/name,destination/name)
            for row in geometry['groups']['joint']['rows']:
                path=source/'decoded/joint'/row['id']/'structure.npz'
                if sha(path)!=row['structure_sha256']:raise ValueError('decoded structure changed')
                target=destination/'decoded/joint'/row['id']/'structure.npz'
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            refold(out,destination)
            reused=False
        record=dict(step=step,source=str(source),refold=str(destination/'refold.json'),
            refold_sha256=sha(destination/'refold.json'),geometry_sha256=sha(source/'geometry.json'),reused=reused)
        results.append(record)
        write(out/f'{variant}_refold_index.json',dict(complete=False,rows=results))
        print(record,flush=True)
    write(out/f'{variant}_refold_index.json',dict(complete=True,rows=results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['released','refold'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--variant',choices=['recovery','released'])
    args=parser.parse_args();out=args.out.resolve();plan=read(out/'plan.json');root=Path(plan['training_root'])
    if args.stage=='released':released_generation(root,out,plan)
    else:finish_refolds(root,out,args.variant)


if __name__=='__main__':main()
