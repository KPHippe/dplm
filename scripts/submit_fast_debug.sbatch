#!/bin/bash
#SBATCH --job-name=dplm-fast-debug
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_debug_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_debug_%j.err
set -euo pipefail
DPLM_ROOT=/hpc/mydata/kyle.hippe/github/dplm
FAST_ROOT=/hpc/mydata/kyle.hippe/github/fast-rewrite
PRETRAINED=/hpc/mydata/kyle.hippe/.cache/huggingface/hub/models--airkingbd--dplm_650m/snapshots/7a7e651baa667d094aba05e9dc1cf52a3332110a
export PYTHONPATH="$DPLM_ROOT/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DPLM_ROOT"
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
    train_fast_dplm.py --cache "$FAST_ROOT/outputs/dplm/debug_v1/train" \
    --val-cache "$FAST_ROOT/outputs/dplm/debug_v1/val" --pretrained "$PRETRAINED" "$@"
