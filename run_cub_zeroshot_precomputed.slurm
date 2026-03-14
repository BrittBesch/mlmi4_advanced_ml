#!/bin/bash
#
# SLURM batch script for CUB zero-shot experiment using precomputed features (Table 3).
# Uses cvpr2016_cub GoogLeNet features + 312-dim CUB attributes per Snell et al. (2017).

# ===== SLURM RESOURCES =====
#SBATCH --job-name=cub-zeroshot-precomputed
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=ampere
#SBATCH --account=mlmi-bb720-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --gres=gpu:nvidia_a100:1

set -euo pipefail

echo "Running on host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"

# ===== ENVIRONMENT SETUP =====
cd /home/bb720/rds/hpc-work/mlmi4_advanced_ml

source .venv/bin/activate

export PYTHONPATH="/home/bb720/rds/hpc-work:${PYTHONPATH:-}"

mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/results

# ===== RUN =====
python src/training/train_zeroshot_precomputed.py \
    --data_root data/cvpr2016_cub \
    --cub_root  data/CUB_200_2011 \
    --config    configs/cub_precomputed_config.yaml

echo "End time: $(date)"
