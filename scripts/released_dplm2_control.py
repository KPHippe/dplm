"""Frozen released DPLM-2 positive control, official and existing pipeline sampling."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ALPHABET = 'ARNDCQEGHILKMFPSTWYV'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from fast_dplm.refolding import write_json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(path), value)


def input_tokens(model, record):
    aa = [model.tokenizer._token_to_id[ALPHABET[int(x)]] for x in record['aa']]
    n = len(aa)
    struct = [model.struct_bos_id] + [model.tokenizer._token_to_id['0000']]*n + [model.struct_eos_id]
    return torch.tensor([struct + [model.aa_bos_id] + aa + [model.aa_eos_id]], device=model.device)


def unpack(model, tokens, record, conditional):
    n = len(record['aa'])
    if tokens.shape != (1, 2*(n+2)):
        raise ValueError('wrong released output shape')
    st, aa = tokens[0].chunk(2)
    if [int(st[0]), int(st[-1]), int(aa[0]), int(aa[-1])] != [model.struct_bos_id,model.struct_eos_id,model.aa_bos_id,model.aa_eos_id]:
        raise ValueError('released output changed boundary tokens')
    vocabulary = model.tokenizer.all_tokens
    codes = [vocabulary[int(x)] for x in st[1:-1]]
    sequence = [vocabulary[int(x)] for x in aa[1:-1]]
    if any(s not in ALPHABET or len(s) != 1 for s in sequence):
        raise ValueError('released output contains noncanonical sequence symbols')
    if any(not s.isdecimal() or not 0 <= int(s) < 8192 for s in codes):
        raise ValueError('released output contains invalid structure symbols')
    result = dict(aa=np.array([ALPHABET.index(x) for x in sequence],dtype=np.uint8),
                  fast_id=np.array([int(s) for s in codes],dtype=np.uint16),
                  residue_id=record['residue_id'],chain_end=record['chain_end'])
    if conditional and not np.array_equal(result['aa'],record['aa']):
        raise ValueError('released generation changed conditioning sequence')
    return result


def official(model, record, conditional, steps, seed):
    ids = input_tokens(model, record)
    partial = model.get_modality_type(ids).eq(model.aa_type) if conditional else None
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        output = model.generate(ids,max_iter=steps,partial_masks=partial,
            sampling_strategy='argmax' if conditional else 'annealing@2.0:0.1',
            unmasking_strategy='deterministic' if conditional else 'stochastic1.0')['output_tokens']
    return unpack(model,output,record,conditional)


def requests(panel):
    out = []
    for arm in ('official_default','official_100','pipeline_100'):
        if arm != 'official_100':
            for number,e in enumerate(panel['generation_validation']['entries']):
                out.append(dict(arm=arm,kind='conditional',id=f'entry_{number:04d}',seed=e['seed'],length=e['length'],entry=e,steps=100))
        for number,s in enumerate(panel['samples']):
            out.append(dict(arm=arm,kind='joint',id=f'sample_{number:04d}',**s,steps=500 if arm=='official_default' else 100))
    return out


def get_record(item, cache):
    from fast_dplm.generation_validation import sequence_condition
    if item['kind']=='conditional':
        return sequence_condition(cache[item['entry']['index']])
    n=item['length']
    return dict(aa=np.zeros(n,dtype=np.uint8),fast_id=np.zeros(n,dtype=np.uint16),
                residue_id=np.arange(1,n+1,dtype=np.int32),chain_end=np.arange(n)==n-1)


def generate(args, plan, smoke=False):
    from released_dplm2_loader import load
    from fast_dplm.data import TokenCache, collate, RESIDUE, sha256
    from fast_dplm.generation_validation import validate_generation_panel
    from fast_dplm.diagnostic_sampling import generate_diagnostic
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    model, port, proof = load(args.root/'code/src',plan['checkpoint'],with_port=True)
    for key in ('checkpoint_sha256','config_sha256','vocab_sha256'):
        if proof[key] != plan[key]:
            raise ValueError(f'released asset changed: {key}')
    model.cuda();port.cuda()
    cache=TokenCache(plan['val_cache']);panel=read(args.root/'panel.json')
    validate_generation_panel(cache,panel['generation_validation'])
    if smoke:
        records=[get_record(x,cache) for x in requests(panel)[:2]]
        batch=collate(records,port.aa_map,port.special,'cuda')
        real=batch['state']==RESIDUE
        mapping=torch.tensor(proof['pipeline_vocabulary_permutation'],device='cuda')
        checks=[]
        for label,sm,am in [('conditional',real,torch.zeros_like(real)),('joint',real,real),('partial',real & (torch.arange(real.shape[1],device='cuda')%2==0),real & (torch.arange(real.shape[1],device='cuda')%3==0))]:
            ids=port.input_ids(batch,sm,am)
            with torch.inference_mode():
                official_logits=model(mapping[ids])['logits'][...,mapping]
                po=port(batch,sm,am)
                port_logits=torch.cat([po['struct_loss_logits'],po['aa']],1)
            difference=(official_logits-port_logits).abs()
            torch.testing.assert_close(official_logits,port_logits,rtol=1e-4,atol=1e-3)
            checks.append(dict(case=label,max_absolute_logit_difference=float(difference.max()),
                argmax_agreement=float((official_logits.argmax(-1)==port_logits.argmax(-1)).float().mean())))
        example=requests(panel)[0];record=get_record(example,cache)
        official(model,record,True,20,example['seed'])
        j=dict(kind='joint',length=100);record=get_record(j,cache)
        official(model,record,False,20,100919)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            generate_diagnostic(port,[record],steps=20,seed=100919,policy='legacy')
        write(args.root/'smoke.json',dict(passed=True,proof=proof,logit_comparison=checks,
             canonical_outputs=True,conditioning_preserved=True,limitation='Infrastructure/forward comparison; 20-step samples are not quality evidence'))
        return
    if not read(args.root/'smoke.json')['passed']:
        raise ValueError('generation requires passing smoke')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    write(args.root/'loader_proof'/f'rank_{rank}.json',proof)
    for index,item in enumerate(requests(panel)):
        if index%world!=rank:
            continue
        folder=args.root/'samples'/item['arm']/item['kind']/item['id']
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        record=get_record(item,cache)
        row=dict(request=item,status='failed')
        try:
            if item['arm']=='pipeline_100':
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    sample=generate_diagnostic(port,[record],steps=item['steps'],seed=item['seed'],
                        policy='legacy',condition='aa' if item['kind']=='conditional' else None)[0]
            else:
                sample=official(model,record,item['kind']=='conditional',item['steps'],item['seed'])
            np.savez(folder/'tokens.npz',**sample)
            row.update(status='ok',tokens_sha256=sha256(folder/'tokens.npz'))
        except (RuntimeError,ValueError) as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
        write(folder/'result.json',row)
        print(json.dumps(dict(rank=rank,**row)),flush=True)
    write(args.root/'generation_complete'/f'rank_{rank}.json',dict(rank=rank,world_size=world,complete=True))


def decoder_setup(root,plan):
    native=Path(plan['native_root'])
    sys.path[:0]=[str(native/'runtime/dplm/scripts'),str(native/'runtime/fast/src')]
    from native_control_corpus import setup
    from native_control_evaluate import decode_sample, summarize
    return setup(native)[1],decode_sample,summarize


def decode(args,plan):
    from fast_dplm.data import TokenCache,sha256
    from fast_dplm.generation_metrics import sequence_metrics,sequence_panel
    model,decode_sample,summarize=decoder_setup(args.root,plan)
    from fast.data.parsers import load_structure
    from native_control_corpus import encode
    cache=TokenCache(plan['val_cache']);panel=read(args.root/'panel.json')
    for rank in range(plan['world_size']):
        assert read(args.root/'generation_complete'/f'rank_{rank}.json')['complete']
    controls=[]
    for i,e in enumerate(panel['generation_validation']['entries']):
        record=cache[e['index']];source=cache.entries[e['index']]
        assert sha256(source['path'])==e['file_sha256']
        native=load_structure(source['path'])
        assert np.array_equal(encode(model,source,record),record['fast_id'])
        row=decode_sample(model,record,args.root/'geometry/controls'/f'entry_{i:04d}',native.coords)
        controls.append(dict(id=f'entry_{i:04d}',entry=e,**row))
    arms={name:dict(conditional=[],joint=[]) for name in ('official_default','official_100','pipeline_100')}
    for item in requests(panel):
        folder=args.root/'samples'/item['arm']/item['kind']/item['id']
        result=read(folder/'result.json');assert result['request']==item
        row=dict(id=item['id'],request=item,status='failed',generation_result_sha256=sha256(folder/'result.json'))
        try:
            if result['status']!='ok':
                raise ValueError(result['error'])
            assert sha256(folder/'tokens.npz')==result['tokens_sha256']
            with np.load(folder/'tokens.npz',allow_pickle=False) as z:
                sample=dict(z)
            record=get_record(item,cache)
            for key in ('residue_id','chain_end'):
                assert np.array_equal(sample[key],record[key])
            reference=None
            if item['kind']=='conditional':
                assert np.array_equal(sample['aa'],record['aa'])
                e=item['entry'];source=cache.entries[e['index']]
                assert sha256(source['path'])==e['file_sha256']
                reference=load_structure(source['path']).coords
            row.update(decode_sample(model,sample,args.root/'geometry'/item['arm']/item['kind']/item['id'],reference))
            row['sequence']=''.join(ALPHABET[int(i)] for i in sample['aa'])
            row['sequence_metrics']=sequence_metrics(sample['aa'])
        except (ValueError,RuntimeError,OSError) as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
        arms[item['arm']][item['kind']].append(row)
    report=dict(controls=dict(rows=controls,summary=summarize(controls,32)),arms={})
    for arm,groups in arms.items():
        seq=[dict(id=r['id'],status=r['status'],**r.get('sequence_metrics',{})) for r in groups['joint']]
        report['arms'][arm]=dict(conditional=dict(rows=groups['conditional'],summary=summarize(groups['conditional'],len(groups['conditional']))),
            joint=dict(rows=groups['joint'],sequence_summary=sequence_panel(seq,20)))
    write(args.root/'geometry/report.json',report)
    write(args.root/'decode_complete.json',dict(report_sha256=sha256(args.root/'geometry/report.json')))


def refold(args,plan):
    from fast_dplm.data import sha256
    from fast_dplm.refolding import FoldCache,parse_fold,SETTINGS
    from fast_dplm.generation_metrics import ca_geometry
    from native_control_evaluate import summarize
    path=args.root/'geometry/report.json';report=read(path)
    assert sha256(path)==read(args.root/'decode_complete.json')['report_sha256']
    cache=FoldCache(args.root/'esmfold2_cache');groups={}
    for arm,data in report['arms'].items():
        rows=[]
        for item in data['joint']['rows']:
            row=dict(id=item['id'],status='failed')
            try:
                if item['status']!='ok':
                    raise ValueError('generation/decoding failed')
                path=args.root/'geometry'/arm/'joint'/item['id']/'structure.npz'
                assert sha256(path)==item['structure_sha256']
                response,receipt=cache.get(item['sequence'])
                xyz,mask,confidence=parse_fold(response,item['sequence'])
                with np.load(path,allow_pickle=False) as z:
                    row.update(ca_geometry(z['xyz'][:,1],xyz[:,1]))
                row.update(status='ok',sequence=item['sequence'],fold_receipt=receipt,
                           mean_plddt=float(confidence.mean()),structure_sha256=item['structure_sha256'],
                           sequence_metrics=item['sequence_metrics'])
                np.savez(path.parent/'refold.npz',xyz=xyz,atom_mask=mask,plddt=confidence)
            except (RuntimeError,ValueError,OSError) as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row)
            write(args.root/'refold_progress.json',dict(arm=arm,rows=rows,requested=20))
        groups[arm]=dict(rows=rows,summary=summarize(rows,20))
        print(json.dumps(dict(arm=arm,summary=groups[arm]['summary'])),flush=True)
    write(args.root/'refold_report.json',dict(settings=SETTINGS,arms=groups,
        geometry_report_sha256=sha256(args.root/'geometry/report.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['smoke','generate','decode','refold'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle-sha256',required=True)
    a=p.parse_args();a.root=a.root.resolve()
    sys.path.insert(0,str(a.root/'code/src'))
    from fast_dplm.data import sha256
    if sha256(a.root/'bundle.json')!=a.bundle_sha256:
        raise ValueError('control bundle changed')
    for relative,digest in read(a.root/'bundle.json')['files'].items():
        if sha256(a.root/relative)!=digest:
            raise ValueError(f'control file changed: {relative}')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    plan=read(a.root/'plan.json')
    for path,digest in plan['external_bindings'].items():
        if sha256(path)!=digest:
            raise ValueError(f'external immutable input changed: {path}')
    if a.stage in ('smoke','generate'):
        generate(a,plan,smoke=a.stage=='smoke')
    else:
        globals()[a.stage](a,plan)


if __name__=='__main__':
    main()
