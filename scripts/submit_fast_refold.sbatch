#!/bin/bash
#SBATCH --job-name=fast-dplm-refold
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_refold_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_refold_%j.err
set -euo pipefail
DPLM_ROOT=/hpc/mydata/kyle.hippe/github/dplm
export PYTHONPATH="$DPLM_ROOT/src:/hpc/mydata/kyle.hippe/github/fast-rewrite/src" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DPLM_ROOT"
exec .venv/bin/python evaluate_fast_samples.py \
    --pretrained /hpc/mydata/kyle.hippe/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a "$@"
