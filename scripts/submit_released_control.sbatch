#!/bin/bash
#SBATCH --job-name=released-dplm2-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/released_control_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/released_control_%j.err
set -euo pipefail
RELEASE_ROOT=$(realpath "$1")
RELEASE_STAGE="$2"
RELEASE_SHA="$3"
export PYTHONPATH="$RELEASE_ROOT/code/src:$RELEASE_ROOT/code"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /hpc/mydata/kyle.hippe/github/dplm
if [[ "$RELEASE_STAGE" == generate ]]; then
    exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
        "$RELEASE_ROOT/code/released_dplm2_control.py" "$RELEASE_STAGE" --root "$RELEASE_ROOT" --bundle-sha256 "$RELEASE_SHA"
else
    exec .venv/bin/python "$RELEASE_ROOT/code/released_dplm2_control.py" "$RELEASE_STAGE" --root "$RELEASE_ROOT" --bundle-sha256 "$RELEASE_SHA"
fi
