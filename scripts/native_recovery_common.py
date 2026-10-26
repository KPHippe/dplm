"""Released DPLM-2 implementation with narrow import/data-loading compatibility.

Model forward, noising, loss, PEFT construction, embedding resize, crop function,
dataset item/collation, sampler and scheduler bodies are executed from frozen
upstream source. The runner supplies mmap data and resumable orchestration.
"""
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler, BatchSampler

from released_dplm2_loader import bootstrap, extract, namespace


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');torch.save(value,temp);os.replace(temp,path)


def verify(root,bundle_sha):
    if sha(root/'bundle.json')!=bundle_sha:
        raise ValueError('bundle changed')
    bundle=read(root/'bundle.json')
    for rel,expected in bundle['files'].items():
        if sha(root/rel)!=expected:
            raise ValueError(f'frozen input changed: {rel}')
    for path,expected in bundle['external_files'].items():
        if sha(path)!=expected:
            raise ValueError(f'external frozen input changed: {path}')
    cfg=read(root/'config.json')
    return cfg


def released(root):
    source=root/'code/src'
    module,changes=bootstrap(source)
    import peft
    from transformers import AutoConfig
    utils=sys.modules['byprot.models.utils']
    utils.__dict__.update(os=os,nn=nn,AutoConfig=AutoConfig,LoraConfig=peft.LoraConfig,
        TaskType=peft.TaskType,get_peft_model=peft.get_peft_model,PeftModel=peft.PeftModel)
    extract(source/'byprot/models/utils.py',['get_net_dplm2'],utils.__dict__)
    module.get_net_dplm2=utils.get_net_dplm2
    # Load the original sequence checkpoint state directly. All initialization,
    # strict loading, resizing and adapter construction still run in upstream.
    namespace('byprot.models.dplm')
    shim=namespace('byprot.models.dplm.dplm')
    class SequenceCheckpoint:
        @staticmethod
        def from_pretrained(path):
            state=torch.load(Path(path)/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
            return types.SimpleNamespace(net=types.SimpleNamespace(state_dict=lambda:state))
    shim.DiffusionProteinLanguageModel=SequenceCheckpoint
    tok=sys.modules['byprot.datamodules.dataset.tokenized_protein']
    tok.__dict__.update(np=np,math=math,torch=torch,dist=dist,Iterable=Iterable,
                        Dataset=Dataset,Sampler=Sampler,BatchSampler=BatchSampler)
    extract(source/'byprot/datamodules/dataset/tokenized_protein.py',
            ['SortishSampler','ApproxBatchSampler','TokenizedProteinDataset','DPLM2Collater'],tok.__dict__)
    scope=dict(torch=torch,Tensor=torch.Tensor,nn=nn,F=F,np=np,pd=pd)
    extract(source/'byprot/modules/cross_entropy.py',
            ['label_smoothed_nll_loss','StructAARDMCrossEntropyLoss'],scope)
    extract(source/'byprot/datamodules/tokenized_protein_datamodule.py',['length_cropping'],scope)
    extract(source/'byprot/utils/lr_scheduler.py',['polynomial_lr_schedule'],scope)
    return types.SimpleNamespace(model_class=module.MultimodalDiffusionProteinLanguageModel,
        tokenizer_class=tok.DPLM2Tokenizer,dataset_class=tok.TokenizedProteinDataset,
        collater=tok.DPLM2Collater,sortish=tok.SortishSampler,batcher=tok.ApproxBatchSampler,
        criterion_class=scope['StructAARDMCrossEntropyLoss'],crop=scope['length_cropping'],
        lr=scope['polynomial_lr_schedule'],compatibility=changes)


def build_model(api,cfg):
    return api.model_class(dict(net=dict(arch_type='esm',name=cfg['pretrained'],dropout=.1,
        pretrain=True,pretrained_model_name_or_path=cfg['pretrained']),
        tokenizer=dict(vocab_file=cfg['vocab_directory']),training_stage='train_from_dplm',
        num_diffusion_timesteps=500,gradient_ckpt=cfg['gradient_checkpointing'],
        self_mixup=dict(enable=False,with_original_loss=False),
        lora=dict(enable=True,lora_rank=16,lora_dropout=.1,
            lora_target_module=r'(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)',
            modules_to_save='lm_head,esm.embeddings')))


class Objective(nn.Module):
    def __init__(self,model,criterion):
        super().__init__();self.model=model;self.criterion=criterion

    def forward(self,batch):
        logits,targets,masks,weights=self.model.compute_loss(batch,weighting='linear')
        loss,logs=self.criterion(logits,targets,masks,weights,cal_constant_loss=True,watch_t1_t2_loss=False)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite native objective')
        return loss,logs


class Corpus:
    def __init__(self,path,vocab,verify_files=True):
        self.path=Path(path);self.manifest=read(self.path/'manifest.json');self.vocab=vocab
        if self.manifest['schema']!='released-dplm-token-projection-v1':
            raise ValueError('wrong corpus schema')
        if verify_files:
            for name,expected in self.manifest['files'].items():
                if sha(self.path/name)!=expected: raise ValueError(f'corpus changed: {name}')
        self.rows=[json.loads(line) for line in (self.path/'rows.jsonl').read_text().splitlines()]
        self.codes=np.load(self.path/'codes.npy',mmap_mode='r')
        self.aa=np.load(self.path/'aa.npy',mmap_mode='r')
        self.offsets=np.load(self.path/'offsets.npy',mmap_mode='r')
        self.lengths=np.diff(self.offsets)
        assert len(self.rows)==self.manifest['entries'] and self.offsets[-1]==len(self.codes)==len(self.aa)
        assert np.array_equal(self.lengths,[r['length'] for r in self.rows])

    def record(self,index):
        lo,hi=self.offsets[index:index+2]
        return dict(pdb_name=self.rows[index]['id'],length=int(hi-lo),
            aa_seq=''.join(self.vocab[int(x)] for x in self.aa[lo:hi]),
            struct_seq=','.join(f'{int(x):04d}' for x in self.codes[lo:hi]))


class ReleasedRows:
    def __init__(self,corpus,lengths):
        self.corpus,self.lengths=corpus,lengths
    def __len__(self): return len(self.lengths)
    def __getitem__(self,index):
        if isinstance(index,str):
            if index!='length': raise KeyError(index)
            return self.lengths
        row=self.corpus.record(index);row['length']=int(self.lengths[index]);return row


def epoch_data(api,corpus,tokenizer,epoch,max_tokens):
    lengths=api.crop(pd.DataFrame({'length':corpus.lengths.copy()}),epoch)['length'].to_numpy()
    dataset=api.dataset_class.__new__(api.dataset_class)
    dataset.data=ReleasedRows(corpus,lengths);dataset.max_len=512;dataset.tokenizer=tokenizer
    sampler=api.sortish(lengths,bucket_size=1000,epoch=epoch)
    batches=api.batcher(sampler,max_tokens,800,lengths,max_len=512).batches
    return dataset,batches,api.collater(tokenizer)


def to_device(batch,device):
    return {k:({n:t.to(device) for n,t in v.items()} if isinstance(v,dict) else v) for k,v in batch.items()}


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda']);np.random.set_state(state['numpy'])


def adapter_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}


def load_adapter(model,state):
    params={k:v for k,v in model.named_parameters() if v.requires_grad}
    if params.keys()!=state.keys(): raise ValueError('adapter parameter set mismatch')
    with torch.no_grad():
        for k,v in params.items(): v.copy_(state[k])
