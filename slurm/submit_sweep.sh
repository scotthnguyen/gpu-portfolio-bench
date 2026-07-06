#!/bin/bash
#SBATCH --job-name=gpu-portfolio-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.err

# Submit: sbatch slurm/submit_sweep.sh
# Monitor: squeue -u $USER  |  tail -f results/slurm_<jobid>.log
# Array job (run across multiple configs): sbatch --array=0-3 slurm/submit_sweep.sh

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Activate environment — adjust path to match your cluster setup
# Option A: conda
# source activate gpu-portfolio-bench
# Option B: venv
# source /scratch/$USER/gpu-portfolio-bench/.venv/bin/activate

cd "$(dirname "$0")/.."

# Confirm CUDA is visible inside the job
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Full sweep: all asset counts × path counts × CPU and GPU
python -m src.bench.run_sweep --device both

echo "Sweep complete. Results in results/benchmark_sweep.csv"
