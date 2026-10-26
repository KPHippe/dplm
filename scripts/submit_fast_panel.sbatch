#!/bin/bash
#SBATCH --job-name=fast-dplm-panel
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_panel_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_panel_%j.err
set -euo pipefail
DPLM_ROOT=/hpc/mydata/kyle.hippe/github/dplm
BUNDLE=$(realpath "$1")
STEP=${2:?Supply a completed milestone step}
export PYTHONPATH="$BUNDLE/code/src" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
exec "$DPLM_ROOT/.venv/bin/python" "$BUNDLE/code/run_fast_evaluation.py" --bundle "$BUNDLE" --step "$STEP"
