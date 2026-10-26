#!/bin/bash
#SBATCH --job-name=dplm-objective-canary
#SBATCH --partition=gpu
#SBATCH --gpus=h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_objective_canary_%A_%a.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_objective_canary_%A_%a.err
set -euo pipefail
CANARY_ROOT=$(realpath "$1")
CANARY_ARMS=(control mixup constant both)
CANARY_BUNDLE="$CANARY_ROOT/${CANARY_ARMS[$SLURM_ARRAY_TASK_ID]}"
bash "$CANARY_BUNDLE/submit_dplm2_recipe.sbatch" "$CANARY_BUNDLE" --stop-after 10
bash "$CANARY_BUNDLE/submit_dplm2_recipe.sbatch" "$CANARY_BUNDLE" --resume
PYTHONPATH="$CANARY_BUNDLE/code/src" OMP_NUM_THREADS=2 \
    /hpc/mydata/kyle.hippe/github/dplm/.venv/bin/python \
    "$CANARY_ROOT/verify_recipe_checkpoint.py" --bundle "$CANARY_BUNDLE"
