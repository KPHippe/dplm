"""Frozen CAMEO/token controls and sequence-only ESMFold2 recovery diagnostics."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from native_recovery_common import read,write,sha,Corpus,to_device,rng_state,restore_rng

ALPHABET='ARNDCQEGHILKMFPSTWYV'


def record(root,corpus,item,conditional):
    n=item['length']
    if conditional:
        ref=dict(np.load(root/'references'/f'{item["id"]}.npz'))
        lo,hi=corpus.offsets[item['index']:item['index']+2]
        return dict(aa=ref['aa'].astype(np.uint8),fast_id=np.array(corpus.codes[lo:hi]),
                    residue_id=ref['residue_id'],chain_end=ref['chain_end'])
    return dict(aa=np.zeros(n,dtype=np.uint8),fast_id=np.zeros(n,dtype=np.uint16),
                residue_id=np.arange(1,n+1,dtype=np.int32),chain_end=np.arange(n)==n-1)


@torch.no_grad()
def evaluate(model,objective,api,corpus,root,run,step,sampling=False,small=False):
    from released_dplm2_control import official
    from fast_dplm.generation_metrics import sequence_metrics,sequence_panel
    rank=dist.get_rank();world=dist.get_world_size();device=model.device
    state=rng_state();was_training=model.training;model.eval()
    dest=root/run/'evaluation'/f'step_{step:07d}'
    dest.mkdir(parents=True,exist_ok=True)
    collater=api.collater(model.tokenizer);rows=[]
    indices=list(range(len(corpus.rows)))[rank::world]
    for start in range(0,len(indices),4):
        selected=indices[start:start+4];raw=[]
        for index in selected:
            r=corpus.record(index);n=min(512,r['length'])
            codes=r['struct_seq'].split(',')[:n];aa=r['aa_seq'][:n]
            raw.append(dict(struct_tokens='<cls_struct>'+''.join(codes)+'<eos_struct>',
                            aatype_tokens='<cls_aa>'+aa+'<eos_aa>'))
        batch=to_device(collater(raw),device);torch.manual_seed(730000+rank*1000+start)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,logs=objective(batch)
            st=batch['struct_tokens']['targets'];aa=batch['aatype_tokens']['targets']
            sm=model.get_non_special_symbol_mask(st)
            ids=torch.cat([st.masked_fill(sm,model.struct_mask_id),aa],1)
            logits=model(ids)['logits'][:,:st.shape[1]]
            full=torch.nn.functional.cross_entropy(logits[sm].float(),st[sm],reduction='sum')
        rows.append(dict(indices=selected,loss=float(loss),full_mask_structure_ce_sum=float(full),
            full_mask_structure_count=int(sm.sum()),full_mask_structure_correct=int((logits.argmax(-1)[sm]==st[sm]).sum()),
            metrics={k:float(v) for k,v in logs.items()}))
    write(dest/f'validation_rank_{rank}.json',rows)
    panel=read(root/'panel.json');generated=[]
    if sampling:
        requests=[dict(kind=k,**s) for k in ('conditional','joint') for s in (panel[k][:2] if small else panel[k])]
        for number,item in enumerate(requests):
            if number%world!=rank:continue
            conditional=item['kind']=='conditional';r=record(root,corpus,item,conditional)
            row=dict(request=item,status='failed')
            try:
                steps=20 if small else (100 if conditional else 500)
                result=official(model,r,conditional,steps,item['seed'])
                out=dest/'tokens'/item['kind'];out.mkdir(parents=True,exist_ok=True)
                path=out/(item['id']+'.npz');np.savez(path,**result)
                row.update(status='ok',tokens_sha256=sha(path),sequence=''.join(ALPHABET[int(x)] for x in result['aa']),
                           sequence_metrics=sequence_metrics(result['aa']))
            except (ValueError,RuntimeError,OSError) as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            generated.append(row)
        write(dest/f'generation_rank_{rank}.json',generated)
    dist.barrier()
    if rank==0:
        all_rows=[r for i in range(world) for r in read(dest/f'validation_rank_{i}.json')]
        count=sum(r['full_mask_structure_count'] for r in all_rows)
        summary=dict(step=step,validation_entries=sum(len(r['indices']) for r in all_rows),
            full_mask_structure_ce=sum(r['full_mask_structure_ce_sum'] for r in all_rows)/count,
            full_mask_structure_accuracy=sum(r['full_mask_structure_correct'] for r in all_rows)/count)
        if sampling:
            gen=[r for i in range(world) for r in read(dest/f'generation_rank_{i}.json')]
            assert len(gen)==len(requests)
            write(dest/'generation.json',dict(step=step,small=small,requests=requests,rows=gen,
                  panel_sha256=sha(root/'panel.json')))
            joint=[dict(status=r['status'],**r.get('sequence_metrics',{})) for r in gen if r['request']['kind']=='joint']
            summary['joint_sequence']=sequence_panel(joint,len(joint))
        write(dest/'validation.json',summary);print('VALIDATION',summary,flush=True)
    dist.barrier()
    if sampling:
        torch.cuda.empty_cache()
        # The audited tokenizer loader requires a fresh process. Keep it separate
        # from native generator imports; child uses this rank's allocated GPU.
        command=[sys.executable,str(root/'code/native_recovery_eval.py'),'decode','--root',str(root),
                 '--folder',str(dest),'--rank',str(rank),'--world',str(world)]
        subprocess.run(command,check=True)
        dist.barrier()
        if rank==0:aggregate(root,dest,world)
        dist.barrier()
    restore_rng(state);model.train(was_training)


def decode(args):
    cfg=read(args.root/'config.json');root=args.root
    if args.controls:
        panel=read(root/'panel.json')
        requests=[dict(kind='control',**x) for x in panel['conditional']]
        gen={}
    else:
        manifest=read(args.folder/'generation.json')
        requests=manifest['requests']+[dict(**(x|{'kind':'control'})) for x in manifest['requests'] if x['kind']=='conditional']
        gen={(r['request']['kind'],r['request']['id']):r for r in manifest['rows']}
    tasks=[r for i,r in enumerate(requests) if i%args.world==args.rank]
    args.folder.mkdir(parents=True,exist_ok=True)
    if not tasks:
        write(args.folder/f'decode_rank_{args.rank}.json',[]);return
    torch.cuda.set_device(args.rank)
    old=Path(cfg['native_tokenizer_root'])
    sys.path[:0]=[str(old/'runtime/dplm/scripts'),str(old/'runtime/fast/src')]
    from native_control_corpus import setup
    from native_control_evaluate import decode_sample
    from fast_dplm.generation_metrics import ca_geometry
    _,decoder,provenance=setup(old)
    vocab=(Path(cfg['vocab_directory'])/'vocab.txt').read_text().splitlines()
    corpus=Corpus(root/'data/valid',vocab)
    rows=[]
    for item in tasks:
        row=dict(id=item['id'],kind=item['kind'],request=item,status='failed')
        try:
            native=item['kind']!='joint';reference=None
            if native:
                refpath=root/'references'/f'{item["id"]}.npz'
                ref=dict(np.load(refpath));reference=ref['xyz']
            if item['kind']=='control':sample=record(root,corpus,item,True)
            else:
                original=gen[(item['kind'],item['id'])]
                if original['status']!='ok':raise ValueError('generation failed')
                tokenpath=args.folder/'tokens'/item['kind']/(item['id']+'.npz')
                if sha(tokenpath)!=original['tokens_sha256']:raise ValueError('token file changed')
                sample=dict(np.load(tokenpath));row['sequence']=original['sequence'];row['sequence_metrics']=original['sequence_metrics']
                if native:
                    expected=record(root,corpus,item,True)
                    for k in ('aa','residue_id','chain_end'):
                        if not np.array_equal(sample[k],expected[k]):raise ValueError('conditioning/topology changed')
            out=args.folder/'decoded'/item['kind']/item['id']
            row.update(decode_sample(decoder,sample,out,None))
            if reference is not None:
                decoded=dict(np.load(out/'structure.npz'))
                observed=ref['atom_mask'][:,1]&decoded['atom_mask'][:,1]
                row.update(ca_geometry(decoded['xyz'][observed,1],reference[observed,1]))
                row.update(native_ca_scored=int(observed.sum()),native_ca_requested=len(observed))
        except (ValueError,RuntimeError,OSError) as exc:
            row['status']='failed'
            row['error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
    write(args.folder/f'decode_rank_{args.rank}.json',rows)
    write(args.folder/f'decoder_provenance_rank_{args.rank}.json',provenance)


def aggregate(root,folder,world):
    from native_control_evaluate import summarize
    rows=[r for rank in range(world) for r in read(folder/f'decode_rank_{rank}.json')]
    groups={k:[r for r in rows if r['kind']==k] for k in ('conditional','joint','control')}
    report=dict(groups={k:dict(rows=v,summary=summarize(v,len(v),fold_requested=k!='joint')) for k,v in groups.items()},
        failures=sum(r['status']!='ok' for r in rows),panel_sha256=sha(root/'panel.json'),
        note='Author token cache decoded with pinned released LFQ8k renderer; no claim of author encoder execution equivalence. Native backbone/CB only; raw and matched-oxygen chemistry separate.')
    write(folder/'geometry.json',report)
    if report['failures']:raise RuntimeError('evaluation failures retained; inspect geometry report')
    write(folder/'complete.json',dict(geometry_sha256=sha(folder/'geometry.json')))
    return report


def refold(root,folder):
    from fast_dplm.refolding import FoldCache,parse_fold,SETTINGS
    from fast_dplm.generation_metrics import ca_geometry
    from native_control_evaluate import summarize
    report=read(folder/'geometry.json')
    if sha(folder/'geometry.json')!=read(folder/'complete.json')['geometry_sha256']:raise ValueError('geometry changed')
    cache=FoldCache(root/'esmfold2_cache');rows=[]
    for item in report['groups']['joint']['rows']:
        row=dict(id=item['id'],status='failed')
        try:
            if item['status']!='ok':raise ValueError('generation/decoding failed')
            response,receipt=cache.get(item['sequence']);xyz,mask,confidence=parse_fold(response,item['sequence'])
            path=folder/'decoded/joint'/item['id']/'structure.npz'
            if sha(path)!=item['structure_sha256']:raise ValueError('decoded coordinates changed')
            sample=dict(np.load(path));row.update(ca_geometry(sample['xyz'][:,1],xyz[:,1]))
            row.update(status='ok',sequence=item['sequence'],sequence_metrics=item['sequence_metrics'],
                       mean_plddt=float(confidence.mean()),fold_receipt=receipt)
            np.savez(path.parent/'refold.npz',xyz=xyz,atom_mask=mask,plddt=confidence)
        except (ValueError,RuntimeError,OSError) as exc:row['error']=str(exc)
        rows.append(row);write(folder/'refold_progress.json',rows)
    write(folder/'refold.json',dict(rows=rows,summary=summarize(rows,len(rows)),settings=SETTINGS,
        nonrepetitive_below_2a=sum(r['status']=='ok' and r['ca_rmsd_a']<2 and r['sequence_metrics']['repeat4_fraction']<=.5 for r in rows),
        geometry_sha256=sha(folder/'geometry.json')))
    if any(r['status']!='ok' for r in rows):raise RuntimeError('refolding failures retained')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['decode','aggregate','refold','watch'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--controls',action='store_true');p.add_argument('--train-job');p.add_argument('--run',default='production')
    a=p.parse_args();a.root=a.root.resolve()
    if a.stage=='decode':decode(a)
    elif a.stage=='aggregate':aggregate(a.root,a.folder,a.world)
    elif a.stage=='refold':refold(a.root,a.folder)
    else:
        cfg=read(a.root/'config.json')
        for step in [0,*cfg['sampling_steps']]:
            folder=a.root/a.run/'evaluation'/f'step_{step:07d}'
            while not (folder/'complete.json').exists():
                try:
                    status=subprocess.check_output(
                        ['sacct','-j',a.train_job,'--format=JobID,State','-n','-P'],
                        text=True,stderr=subprocess.PIPE,timeout=30)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                    # Slurm accounting outages must not discard later panels.
                    # The worker's Slurm time limit bounds this retry loop.
                    print('Slurm accounting unavailable; retrying in 30 seconds',flush=True)
                    time.sleep(30)
                    continue
                state=next((line.split('|')[1] for line in status.splitlines() if line.split('|')[0]==a.train_job),'UNKNOWN')
                if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'):
                    raise RuntimeError(f'training ended {state} before evaluation step{step}')
                time.sleep(30)
            refold(a.root,folder)


if __name__=='__main__':main()
