#!/bin/bash
#SBATCH --job-name=dplm-esmfold2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_esmfold2_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_esmfold2_%j.err
set -euo pipefail
EVAL_BUNDLE=$(realpath "${1:?Supply the ESMFold2 evaluator bundle}")
STEP=${2:?Supply a completed milestone step}
FAST_ROOT=/hpc/mydata/kyle.hippe/github/fast-rewrite
TRAINING_BUNDLE=$("$FAST_ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_bundle"])' "$EVAL_BUNDLE/config.json")
export PYTHONPATH="$EVAL_BUNDLE/code/src:$TRAINING_BUNDLE/fast_code/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
"$FAST_ROOT/.venv/bin/python" "$EVAL_BUNDLE/code/run_fast_esmfold2_panel.py" --bundle "$EVAL_BUNDLE" --step "$STEP"
