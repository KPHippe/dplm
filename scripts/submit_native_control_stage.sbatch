#!/bin/bash
#SBATCH --job-name=native-dplm-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_control_%j.out
#SBATCH --error=/hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/slurm/native_control_%j.err
set -euo pipefail
NATIVE_ROOT=$(realpath "$1")
NATIVE_STAGE="$2"
NATIVE_SHA="$3"
NATIVE_CELL="${4:-canary}"
export PYTHONPATH="$NATIVE_ROOT/execution/dplm_src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd /hpc/mydata/kyle.hippe/github/dplm
exec .venv/bin/python "$NATIVE_ROOT/execution/run_native_control_stage.py" "$NATIVE_STAGE" \
    --root "$NATIVE_ROOT" --execution-sha256 "$NATIVE_SHA" --cell "$NATIVE_CELL"
