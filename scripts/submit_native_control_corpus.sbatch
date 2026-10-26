#!/bin/bash
#SBATCH --job-name=native-dplm-corpus
#SBATCH --partition=gpu
#SBATCH --gpus=h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_corpus_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_corpus_%j.err
set -euo pipefail
NATIVE_ROOT=$(realpath "$1")
NATIVE_COMMAND=${2:-export}
NATIVE_WORLD=${3:-8}
export PYTHONPATH="$NATIVE_ROOT/runtime/dplm/src:$NATIVE_ROOT/runtime/fast/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /hpc/mydata/kyle.hippe/github/dplm
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NATIVE_WORLD" \
    "$NATIVE_ROOT/runtime/dplm/scripts/native_control_corpus.py" "$NATIVE_COMMAND" --root "$NATIVE_ROOT"
