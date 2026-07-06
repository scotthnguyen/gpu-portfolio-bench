#!/bin/bash
#SBATCH --job-name=gpu-bench-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm_array_%A_%a.log

# Splits the sweep into 4 parallel jobs, one per asset-count bucket.
# Submit: sbatch slurm/array_sweep.sh
# Each task runs paths sweep for one asset count.

ASSET_COUNTS=(10 50 100 500)
N_ASSETS=${ASSET_COUNTS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID: n_assets=$N_ASSETS on $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$(dirname "$0")/.."
python -m src.bench.run_sweep --device both --assets "$N_ASSETS"
