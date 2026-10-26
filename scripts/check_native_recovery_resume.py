"""Require identical data traces, adapters, optimizer and RNG after a resume fork."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(f'tensor differs: {path}; max={float((a-b).abs().max())}')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],f'{path}/{k}')
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b,strict=True)):equal(x,y,f'{path}/{i}')
    else:assert a==b,path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    x=torch.load(a.root/'canary_reference/last.pt',map_location='cpu',weights_only=False,mmap=True)
    y=torch.load(a.root/'canary_split/last.pt',map_location='cpu',weights_only=False,mmap=True)
    equal(x,y)
    traces=[]
    for folder in ('canary_reference','canary_split'):
        rows=[json.loads(l) for l in (a.root/folder/'metrics.jsonl').read_text().splitlines()]
        traces.append([{k:v for k,v in r.items() if k not in ('update_seconds','peak_memory_gb')} for r in rows])
    equal(*traces)
    assert x['step']==20
    result=dict(passed=True,updates=20,split=10,adapter_optimizer_rng_and_data_trace='bitwise identical',
                bundle_sha256=x['bundle_sha256'],initial_sha256=x['initial_sha256'],world_size=8)
    (a.root/'canary_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
