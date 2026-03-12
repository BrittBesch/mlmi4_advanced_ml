#!/bin/bash
#
# SLURM batch script for CUB-200-2011 zero-shot experiment (Table 3 replication)
# Follows the conventions described in README.md:
#   - Uses configs/cub_config.yaml
#   - Saves checkpoints to experiments/checkpoints
#   - Saves results / metrics to experiments/results
#   - Uses experiments/logs for SLURM/stdout logs
#
# Edit the lines marked TODO to match your cluster (partition, account, modules, conda env).

# ===== SLURM RESOURCES =====
#SBATCH --job-name=cub-zeroshot
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=ampere
#SBATCH --account=mlmi-bb720-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --gres=gpu:nvidia_a100:1

# Optional / cluster-specific:
# If your site uses constraints, set them here (otherwise leave commented out).
##SBATCH --constraint=TODO_CONSTRAINT

set -euo pipefail

echo "Running on host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"

# ===== ENVIRONMENT SETUP =====
# Adjust this section to your environment.

# Example: load modules
# module purge
# module load cuda/12.1
# module load anaconda3

# Example: activate conda env
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate mlmi4_protonet

cd /home/bb720/rds/hpc-work/mlmi4_advanced_ml

# Activate local virtual environment
source .venv/bin/activate

# Ensure experiment directories exist (as suggested in README)
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/results

# ===== CONFIG =====
CONFIG="configs/cub_config.yaml"
echo "Using config: $CONFIG"

# ===== RUN CUB ZERO-SHOT TRAINING =====
# Per README:
#   python src/training/train_zeroshot.py --config configs/cub_config.yaml
# Best model will be saved to:
#   experiments/checkpoints/cub_zeroshot_best.pt

python src/training/train_zeroshot.py --config "$CONFIG"

echo "End time: $(date)"
