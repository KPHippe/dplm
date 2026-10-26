#!/bin/bash
#SBATCH --job-name=dplm2-fast-recipe
#SBATCH --partition=gpu
#SBATCH --gpus=h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=14-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_recipe_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/dplm_recipe_%j.err
set -euo pipefail
DPLM_ROOT=/hpc/mydata/kyle.hippe/github/dplm
BUNDLE=$(realpath "$1")
shift
export PYTHONPATH="$BUNDLE/code/src" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$DPLM_ROOT"
RECIPE_WORLD_SIZE=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "$BUNDLE/bundle.json")
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$RECIPE_WORLD_SIZE" \
    "$BUNDLE/code/run_fast_bundle.py" --bundle "$BUNDLE" "$@"
