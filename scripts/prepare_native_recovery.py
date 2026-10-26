"""Freeze a candidate native recovery runner; never modify an existing bundle."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from native_recovery_common import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',required=True)
    a=p.parse_args();outer=a.root.resolve();root=outer/'candidates'/a.candidate
    root.mkdir(parents=True,exist_ok=False)
    dplm=Path(__file__).resolve().parents[1];fast=dplm.parent/'fast-rewrite'
    assert read(outer/'data_audit.json')['passed']
    assert read(outer/'reference_audit.json')['matched']==32
    for name in ('data','references','dependencies'):(root/name).symlink_to(outer/name,target_is_directory=True)
    for name in ('panel.json','data_audit.json','reference_audit.json'):
        shutil.copy2(outer/name,root/name)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(dplm/'src',root/'code/src',ignore=ignore)
    shutil.copytree(fast/'outputs/dplm/native_control_v1_20261005/runtime/fast/src',root/'code/fast_src',ignore=ignore)
    for name in ('native_recovery_common.py','native_recovery_eval.py','train_native_recovery.py',
                 'check_native_recovery_resume.py','check_native_recovery_cpu.py',
                 'released_dplm2_loader.py','released_dplm2_control.py','native_control_evaluate.py'):
        shutil.copy2(dplm/'scripts'/name,root/'code'/name)
    pretrained=Path('/hpc/mydata/kyle.hippe/.cache/huggingface/hub/models--airkingbd--dplm_650m/snapshots/7a7e651baa667d094aba05e9dc1cf52a3332110a')
    vocab=Path('/hpc/mydata/kyle.hippe/.cache/huggingface/hub/models--airkingbd--dplm2_650m/snapshots/0bc69b644976c6680ab7e26669854d1979e8876e')
    cfg=dict(schema='released-native-recovery-v1',seed=42,steps=100000,max_tokens=8192,warmup=2000,lr=1e-4,
        gradient_checkpointing=False,validate_every=500,checkpoint_every=500,
        sampling_steps=[10000,25000,50000,75000,100000],pretrained=str(pretrained),vocab_directory=str(vocab),
        native_tokenizer_root=str(fast/'outputs/dplm/native_control_v1_20261005'),
        self_mixup=False,noise_weighting='linear',loss_reduction='released rank-local masked means; DDP mean',
        precision='bf16 autocast, fp32 weights, TF32 off, deterministic algorithms',
        optimizer=dict(name='AdamW',betas=[.9,.98],weight_decay=.01,clip_norm=.5),
        renderer_gate=dict(requested=32,failures=0,median_ca_rmsd_below_a=4.,mean_ca_lddt_at_least=.75,
            meaning='Cached-token renderer compatibility only; not a full-atom or generation quality gate'),
        data_projection=read(root/'data_audit.json'),
        reference_protocol=read(root/'reference_audit.json')['policy'])
    write(root/'config.json',cfg)
    script='''#!/bin/bash
#SBATCH --job-name=native-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_recovery_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_recovery_%j.err
set -euo pipefail
REC_ROOT=$(realpath "$1")
REC_SHA="$2"
REC_STAGE="$3"
export PYTHONPATH="$REC_ROOT/dependencies:$REC_ROOT/code/src:$REC_ROOT/code/fast_src:$REC_ROOT/code"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
REC_PY=/hpc/mydata/kyle.hippe/github/dplm/.venv/bin/python
cd /hpc/mydata/kyle.hippe/github/dplm
"$REC_PY" -c 'import sys; from pathlib import Path; from native_recovery_common import verify; verify(Path(sys.argv[1]), sys.argv[2])' "$REC_ROOT" "$REC_SHA"
if [[ "$REC_STAGE" == canary ]]; then
    "$REC_PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$REC_ROOT/code/train_native_recovery.py" --root "$REC_ROOT" --bundle-sha256 "$REC_SHA" --run canary_reference --canary --stress --stop-after 20
    "$REC_PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$REC_ROOT/code/train_native_recovery.py" --root "$REC_ROOT" --bundle-sha256 "$REC_SHA" --run canary_split --canary --stress --stop-after 10
    "$REC_PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$REC_ROOT/code/train_native_recovery.py" --root "$REC_ROOT" --bundle-sha256 "$REC_SHA" --run canary_split --canary --resume --stop-after 20
    "$REC_PY" "$REC_ROOT/code/check_native_recovery_resume.py" --root "$REC_ROOT"
elif [[ "$REC_STAGE" == renderer ]]; then
    "$REC_PY" "$REC_ROOT/code/native_recovery_eval.py" decode --root "$REC_ROOT" --folder "$REC_ROOT/renderer_check" --controls
    "$REC_PY" "$REC_ROOT/code/native_recovery_eval.py" aggregate --root "$REC_ROOT" --folder "$REC_ROOT/renderer_check"
elif [[ "$REC_STAGE" == cpu ]]; then
    exec "$REC_PY" "$REC_ROOT/code/check_native_recovery_cpu.py" --root "$REC_ROOT"
elif [[ "$REC_STAGE" == train ]]; then
    exec "$REC_PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$REC_ROOT/code/train_native_recovery.py" --root "$REC_ROOT" --bundle-sha256 "$REC_SHA" --run production
elif [[ "$REC_STAGE" == canary_refold ]]; then
    exec "$REC_PY" "$REC_ROOT/code/native_recovery_eval.py" refold --root "$REC_ROOT" --folder "$REC_ROOT/canary_reference/evaluation/step_0000000"
elif [[ "$REC_STAGE" == watch ]]; then
    exec "$REC_PY" "$REC_ROOT/code/native_recovery_eval.py" watch --root "$REC_ROOT" --train-job "$4"
else
    exit 2
fi
'''
    (root/'submit.sbatch').write_text(script)
    files={str(f.relative_to(root)):sha(f) for f in sorted((root/'code').rglob('*')) if f.is_file()}
    for name in ('config.json','panel.json','data_audit.json','reference_audit.json','submit.sbatch'):files[name]=sha(root/name)
    external={}
    for folder in (outer/'dependencies',outer/'references',outer/'data'):
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.suffix not in ('.pyc',):external[str(f)]=sha(f)
    for folder in (pretrained,vocab):
        for name in ('config.json','vocab.txt','tokenizer_config.json','special_tokens_map.json'):
            if (folder/name).exists():external[str(folder/name)]=sha(folder/name)
    external[str(pretrained/'pytorch_model.bin')]=sha(pretrained/'pytorch_model.bin')
    for name in ('native_recovery_data.py','native_recovery_references.py','prepare_native_recovery.py'):
        shutil.copy2(dplm/'scripts'/name,root/'code'/name)
        files['code/'+name]=sha(root/'code'/name)
    write(root/'bundle.json',dict(files=files,external_files=external,root=str(root),
        source_git_commit=subprocess.check_output(['git','-C',str(dplm),'rev-parse','HEAD'],text=True).strip()))
    print(json.dumps(dict(root=str(root),bundle_sha256=sha(root/'bundle.json'),files=len(files),external_files=len(external))))


if __name__=='__main__':main()
