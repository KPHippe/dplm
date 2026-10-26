#!/bin/bash
#SBATCH --job-name=dplm-token-audit
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_token_audit_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_token_audit_%j.err
set -euo pipefail
cd /hpc/mydata/kyle.hippe/github/dplm
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec .venv/bin/python scripts/audit_native_tokenizer.py "$@"
