#!/bin/bash
#SBATCH --job-name=fast-dplm-pilot
#SBATCH --partition=gpu
#SBATCH --gpus=h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_pilot_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_pilot_%j.err
set -euo pipefail
DPLM_ROOT=/hpc/mydata/kyle.hippe/github/dplm
BUNDLE=$(realpath "$1")
shift
export PYTHONPATH="$BUNDLE/code/src" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DPLM_ROOT"
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$BUNDLE/code/run_fast_bundle.py" --bundle "$BUNDLE" "$@"
