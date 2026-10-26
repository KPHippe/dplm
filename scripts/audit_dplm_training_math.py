"""Compare frozen adaptation training against released model, PEFT and criterion.

Audit-only model copies. Fixed shared noising plans isolate implementation from
the deliberately different global/local batch reductions. No weights are saved.
"""
import argparse
import contextlib
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from released_dplm2_loader import bootstrap, extract, vocabulary_permutation
from fast_dplm.data import collate, sha256
from fast_dplm.model import noise_plan, loss_sums
from fast_dplm.recipe import model_from_spec, DPLM2Recipe
from transformers import EsmConfig


def semantic_key(name):
    name = name.removeprefix('base_model.model.')
    name = name.replace('.modules_to_save.default.', '.')
    name = name.replace('.base_layer.', '.base.')
    name = name.replace('.lora_A.default.', '.a.').replace('.lora_B.default.', '.b.')
    return 'net.'+name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--native-vocab',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--rotary-fp32',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    device='cpu' if args.tiny else 'cuda'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(901)
    if args.rotary_fp32:
        # Audit-only intervention; do not edit frozen training code or live model.
        import fast_dplm.model as port_module
        text=textwrap.dedent(inspect.getsource(port_module.PairedAttention.forward))
        old='cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)'
        assert text.count(old)==1
        scope=dict(port_module.__dict__)
        exec(compile(text.replace(old,'cos, sin = phase.cos(), phase.sin()'),'<audit-only fp32 rotary>', 'exec'),scope)
        port_module.PairedAttention.forward=scope['forward']
    vocab=(args.native_vocab/'vocab.txt').read_text().splitlines()
    if args.tiny:
        port=DPLM2Recipe(EsmConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
            intermediate_size=64,position_embedding_type='rotary',mask_token_id=32,pad_token_id=1,
            token_dropout=True),[x.replace('_aa>','>') for x in vocab[:33]],levels=[2]*13)
    else:
        state=torch.load(args.initial,map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==0
        port=model_from_spec(state['model_spec'])
        port.load_state_dict(state['model'],strict=True)
        del state
    module,compat=bootstrap(args.source)
    from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2
    from peft import LoraConfig,TaskType,get_peft_model
    mapping=torch.tensor(vocabulary_permutation(vocab))
    inverse=torch.argsort(mapping)
    native_config=EsmConfig.from_dict(port.net.config.to_dict())
    native_config.dplm_type='index'  # Wrapper metadata absent in original sequence checkpoint.
    net=EsmForDPLM2(native_config)
    native_state=net.state_dict()
    row_keys=('esm.embeddings.word_embeddings.weight','lm_head.decoder.weight','lm_head.bias')
    copied=[]
    for k,v in port.net.state_dict().items():
        if '.a.weight' in k or '.b.weight' in k:
            continue
        k=k.replace('.base.','.')
        native_state[k]=v[inverse] if k in row_keys else v
        copied.append(k)
    assert set(native_state)-set(copied)=={'esm.contact_head.regression.weight','esm.contact_head.regression.bias'}
    net.load_state_dict(native_state,strict=True)
    del native_state
    net=get_peft_model(net,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=r'(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)',
        modules_to_save=['lm_head','esm.embeddings'],inference_mode=False,r=16,lora_alpha=32,lora_dropout=.1))
    port_params=dict(port.named_parameters())
    for k,param in net.named_parameters():
        if param.requires_grad:
            key=semantic_key(k)
            if key=='net.esm.embeddings.position_embeddings.weight':
                param.requires_grad_(False)  # Unused rotary table; frozen in port.
                continue
            v=port_params[key]
            if key.removeprefix('net.') in row_keys:
                v=v[inverse]
            param.data.copy_(v)
    official=module.MultimodalDiffusionProteinLanguageModel(
        {'tokenizer':{'vocab_file':str(args.native_vocab)}},net=net)
    scope=dict(torch=torch,Tensor=torch.Tensor,nn=nn,F=F)
    extract(args.source/'byprot/modules/cross_entropy.py',
            ['label_smoothed_nll_loss','StructAARDMCrossEntropyLoss'],scope)
    criterion=scope['StructAARDMCrossEntropyLoss'](label_smoothing=0,ignore_index=1)
    port.to(device);official.to(device);mapping=mapping.to(device)
    pparams={k:v for k,v in port.named_parameters() if v.requires_grad}
    nparams={semantic_key(k):v for k,v in net.named_parameters() if v.requires_grad}
    assert set(pparams)==set(nparams)
    rng=np.random.default_rng(510)
    records=[dict(aa=rng.integers(0,20,n,dtype=np.uint8),fast_id=rng.integers(0,8192,n,dtype=np.uint16),
                  chain_end=np.arange(n)==n-1) for n in (17,31,43,59)]
    batch=collate(records,port.aa_map,port.special)
    plan=noise_plan(batch,torch.Generator().manual_seed(225),balanced=True)
    assert sorted(plan['task'].tolist())==[0,1,2,3]
    batch={k:v.to(device) for k,v in batch.items()};plan={k:v.to(device) for k,v in plan.items()}
    zeros=torch.zeros_like(plan['aa_mask'])
    st_target,aa_target=mapping[port.input_ids(batch,zeros,zeros)].chunk(2,1)
    inputs=mapping[port.input_ids(batch,plan['struct_mask'],plan['aa_mask'])]
    optimizers=[torch.optim.AdamW(params.values(),lr=1e-4,betas=(.9,.98),weight_decay=.01) for params in (pparams,nparams)]
    checks=[]
    cases=[('train_bf16',True)] if args.bf16 else [('eval_backward',False),('train_backward',True),('train_after_update',True)]
    for name,training in cases:
        port.train(training);official.train(training)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        torch.manual_seed(551)
        with torch.autocast(device,dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
            output=port(batch,plan['struct_mask'],plan['aa_mask'],plan['independent'])
            logits=torch.cat([output['struct_loss_logits'],output['aa']],1)
            nums,dens=loss_sums(port,output,batch,plan)
            ploss=.5*(nums/dens.clamp_min(1)).sum()
        ploss.backward()
        torch.manual_seed(551)
        with torch.autocast(device,dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext():
            nlogits=official(inputs,single_modality=plan['independent'])['logits']
            ns,na=nlogits.chunk(2,1)
            nloss,_=criterion(dict(struct=ns,aatype=na),dict(struct=st_target,aatype=aa_target),
                dict(struct=plan['struct_mask'],aatype=plan['aa_mask']),
                dict(struct=plan['sw'][:,None].expand_as(st_target),aatype=plan['aw'][:,None].expand_as(aa_target)))
        nloss.backward()
        logit_error=float((logits-nlogits[...,mapping]).abs().max())
        if not args.bf16:
            torch.testing.assert_close(logits,nlogits[...,mapping],rtol=2e-4,atol=2e-4)
            torch.testing.assert_close(ploss,nloss,rtol=1e-5,atol=1e-6)
        errors=[]
        diff2=port2=native2=0.
        for key,pvalue in pparams.items():
            ng=nparams[key].grad
            assert pvalue.grad is not None and ng is not None, key
            if key.removeprefix('net.') in row_keys:
                ng=ng[mapping]
            errors.append(float((ng-pvalue.grad).abs().max()))
            diff2+=float((ng-pvalue.grad).square().sum())
            port2+=float(pvalue.grad.square().sum())
            native2+=float(ng.square().sum())
            if not args.bf16:
                torch.testing.assert_close(pvalue.grad,ng,rtol=2e-3,atol=2e-5,msg=lambda msg:key+' '+msg)
        norms=[]
        for opt,params in zip(optimizers,(pparams,nparams),strict=True):
            norms.append(float(torch.nn.utils.clip_grad_norm_(params.values(),.5,error_if_nonfinite=True)))
            opt.step()
        update_error=0.
        for key,pvalue in pparams.items():
            nv=nparams[key]
            if key.removeprefix('net.') in row_keys:
                nv=nv[mapping]
            update_error=max(update_error,float((nv-pvalue).abs().max()))
            # Adam is sensitive to tiny sign errors for near-zero gradients.
            if not args.bf16:
                torch.testing.assert_close(pvalue,nv,rtol=1e-3,atol=2e-4)
        row=dict(case=name,port_loss=float(ploss),native_loss=float(nloss),
                 max_logit_error=logit_error,max_gradient_error=max(errors),
                 gradient_norms=norms,max_parameter_error_after_adam=update_error,
                 relative_gradient_l2=(diff2/native2)**.5,
                 gradient_cosine=(port2+native2-diff2)/(2*(port2*native2)**.5),
                 mean_absolute_logit_error=float((logits-nlogits[...,mapping]).abs().float().mean()))
        checks.append(row)
        print(json.dumps(row),flush=True)
        # Realign audit copies to prevent rounding accumulation obscuring the next comparison.
        with torch.no_grad():
            for key,pvalue in pparams.items():
                value=pvalue[inverse.to(device)] if key.removeprefix('net.') in row_keys else pvalue
                nparams[key].copy_(value)
        del output,logits,nlogits,ns,na,ploss,nloss
    result=dict(passed=True if not args.bf16 else None,measurement_completed=True,
        tiny=args.tiny,bf16=args.bf16,audit_only_rotary_fp32=args.rotary_fp32,checks=checks,parameter_tensors=len(pparams),
        trainable_elements=sum(v.numel() for v in pparams.values()),
        initial_sha256=None if args.tiny else sha256(args.initial),source=str(args.source),compatibility=compat,
        versions=dict(torch=torch.__version__,peft=__import__('peft').__version__,transformers=__import__('transformers').__version__),
        exclusions=['unused contact head','unused position table'],
        limitations='Fixed-batch, shared noising; not DDP trajectory equivalence, initialization distribution or data audit; bf16 mode measures discrepancies without a parity pass criterion')
    filename='math_bf16'+('_rotary_fp32' if args.rotary_fp32 else '')+'.json' if args.bf16 else ('math_tiny.json' if args.tiny else 'math_full.json')
    (args.root/filename).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
