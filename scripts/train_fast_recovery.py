"""Resumable eight-GPU runner around the frozen released DPLM-2 recipe."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fast_recovery_common import (read,write,sha,save,verify,released,build_model,Objective,
    Corpus,epoch_data,to_device,rng_state,restore_rng,adapter_state,load_adapter)
from fast_recovery_eval import evaluate


def stress(objective,model,api):
    """Full 8192-residue local batch at max length, backward but no update."""
    state=rng_state()
    raw=[dict(struct_tokens='<cls_struct>'+model.tokenizer.all_tokens[36]*512+'<eos_struct>',
              aatype_tokens='<cls_aa>'+'A'*512+'<eos_aa>') for _ in range(16)]
    batch=to_device(api.collater(model.tokenizer)(raw),model.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=objective(batch)
    loss.backward();objective.zero_grad(set_to_none=True)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1e9
    restore_rng(state);torch.cuda.empty_cache()
    return peak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--run',required=True);p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--canary',action='store_true')
    p.add_argument('--stress',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if Path(a.run).name!=a.run:raise ValueError('run must be one directory name')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    config_message=[verify(root,a.bundle_sha256) if rank==0 else None]
    dist.broadcast_object_list(config_message,src=0)
    cfg=config_message[0]
    if world!=8:raise ValueError('matched FAST swap uses eight GPUs')
    if not a.canary:
        verification=read(root/'verification.json')
        if not verification['production_ready'] or verification['bundle_sha256']!=a.bundle_sha256:
            raise ValueError('production requires verified canary and renderer for this bundle')
    endpoint=cfg['steps'] if a.stop_after is None else a.stop_after
    if not 1<=endpoint<=cfg['steps'] or (a.canary and endpoint>500):raise ValueError('invalid endpoint')
    out=root/a.run;out.mkdir(exist_ok=True)
    lock=None
    if rank==0:
        lock=(out/'.writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'run.json').exists() and not a.resume:raise FileExistsError('run already exists')
    dist.barrier()
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    api=released(root);model=build_model(api,cfg).cuda()
    initial=root/'initial.pt';initial_receipt=root/'initial.json'
    if initial_receipt.exists():
        receipt=read(initial_receipt)
        if rank==0 and sha(initial)!=receipt['sha256']:raise ValueError('initial state changed')
        initial_state=torch.load(initial,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(initial_state,strict=True);del initial_state
    else:
        if rank==0:
            save(initial,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            write(initial_receipt,dict(sha256=sha(initial),seed=cfg['seed'],bundle_sha256=a.bundle_sha256,
                initialized_from=cfg['pretrained'],initialization='unchanged released get_net_dplm2; direct original checkpoint state reader'))
        dist.barrier();receipt=read(initial_receipt)
    if receipt['bundle_sha256']!=a.bundle_sha256:raise ValueError('initialization bundle changed')
    criterion=api.criterion_class(label_smoothing=0.,ignore_index=model.pad_id)
    objective=Objective(model,criterion)
    # Match the released strategy: unused rotary position table remains in the
    # parameter set; DDP handles it without manufacturing gradients.
    wrapped=DDP(objective,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
    params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],betas=(.9,.98),weight_decay=.01)
    vocab=model.tokenizer.all_tokens
    train=Corpus(root/'data/train',vocab);val=Corpus(root/'data/valid',vocab)
    start=epoch=cursor=0;exposure={};checkpoint=None
    if a.resume:
        checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['bundle_sha256']!=a.bundle_sha256 or checkpoint['initial_sha256']!=receipt['sha256']:
            raise ValueError('resume contract changed')
        load_adapter(model,checkpoint['adapter']);optimizer.load_state_dict(checkpoint['optimizer'])
        start,epoch,cursor=checkpoint['step'],checkpoint['epoch'],checkpoint['cursor']
        exposure=checkpoint['exposure']
        if start>endpoint:raise ValueError('endpoint before saved step')
    dataset,batches,collater=epoch_data(api,train,model.tokenizer,epoch,cfg['max_tokens'])
    if checkpoint:restore_rng(checkpoint['rng'][rank])
    else:torch.manual_seed(cfg['seed']+rank)
    if rank==0:
        write(out/'run.json',dict(config=cfg,bundle_sha256=a.bundle_sha256,initial=receipt,
            trainable_elements=sum(x.numel() for x in params),canary=a.canary,
            data_contract='released crop/sortish/token-budget batches; mmap row adapter; synchronous crop fetching for exact resume',
            compatibility=api.compatibility))
        if a.resume:
            log=out/'metrics.jsonl'
            all_rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
            uncommitted=[x for x in all_rows if x['step']>start]
            if uncommitted:write(out/f'uncommitted_{time.time_ns()}.json',uncommitted)
            log.write_text(''.join(json.dumps(x)+'\n' for x in all_rows if x['step']<=start))
    if a.stress:
        peak=stress(wrapped,model,api)
        write(out/f'stress_rank_{rank}.json',dict(passed=True,local_sequences=16,length=512,
            local_residues=8192,peak_memory_gb=peak,backward_without_update=True))
    if start==0 and (not a.canary or a.run=='canary_reference'):
        evaluate(model,objective,api,val,root,a.run,0,sampling=True,small=a.canary)
    for step in range(start,endpoint):
        if cursor==len(batches):
            epoch+=1;cursor=0
            dataset,batches,collater=epoch_data(api,train,model.tokenizer,epoch,cfg['max_tokens'])
        started=time.monotonic();model.train();indices=batches[cursor]
        raw=[dataset[i] for i in indices];batch=to_device(collater(raw),model.device)
        # LambdaLR in the released runner starts at schedule address 0.
        lr=api.lr(step,cfg['steps'],cfg['warmup'],1e-7,cfg['lr'],1e-5,1.)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapped(batch)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,.5,error_if_nonfinite=True)
        optimizer.step();cursor+=1
        local_sources={}
        for index,row in zip(indices,raw,strict=True):
            key=train.rows[index]['source'];local_sources[key]=local_sources.get(key,0)+row['length']-2
        sources=[None]*world;dist.all_gather_object(sources,local_sources)
        if rank==0:
            for counts in sources:
                for k,n in counts.items():exposure[k]=exposure.get(k,0)+n
        values=torch.tensor([float(loss),len(indices),sum(r['length']-2 for r in raw)],device=model.device)
        dist.all_reduce(values)
        torch.cuda.synchronize()
        if rank==0:
            row=dict(step=step+1,epoch=epoch,cursor=cursor,loss=float(values[0]/world),lr=lr,
                sequences=int(values[1]),valid_residues=int(values[2]),exposure=exposure.copy(),
                gradient_norm=float(norm),update_seconds=time.monotonic()-started,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                local_metrics={k:float(v) for k,v in metrics.items()},
                rank0_batch_identity=[train.rows[i]['id'] for i in indices],
                rank0_batch_tokens_sha256=__import__('hashlib').sha256(batch['struct_tokens']['targets'].cpu().numpy().tobytes()+batch['aatype_tokens']['targets'].cpu().numpy().tobytes()).hexdigest())
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if (step+1)%10==0 or a.canary:print(json.dumps({k:v for k,v in row.items() if k not in ('local_metrics','rank0_batch_identity')}),flush=True)
        milestone=(step+1 in cfg['sampling_steps']) and not a.canary
        validate=(step+1)%cfg['validate_every']==0 or (step+1==endpoint and not a.canary)
        if validate or milestone:evaluate(model,objective,api,val,root,a.run,step+1,sampling=milestone)
        if (step+1)%cfg['checkpoint_every']==0 or step+1==endpoint or milestone:
            states=[None]*world;dist.all_gather_object(states,rng_state())
            if rank==0:
                ckpt=dict(step=step+1,epoch=epoch,cursor=cursor,adapter=adapter_state(model),
                    optimizer=optimizer.state_dict(),rng=states,exposure=exposure,
                    bundle_sha256=a.bundle_sha256,initial_sha256=receipt['sha256'])
                save(out/'last.pt',ckpt)
                if milestone:save(out/'milestones'/f'step_{step+1:07d}.pt',{k:v for k,v in ckpt.items() if k not in ('optimizer','rng')})
            dist.barrier()
    if rank==0:write(out/'complete.json',dict(step=endpoint,final=endpoint==cfg['steps'],last_sha256=sha(out/'last.pt')))
    dist.barrier();dist.destroy_process_group()
    if lock is not None:lock.close()


if __name__=='__main__':main()
