#!/bin/bash
#
# SLURM array job: sweep n_episodes in {50, 100, 200} to find best setting.
# Each array task runs one value of n_episodes.
#
# Submit with:
#   sbatch run_cub_sweep_episodes.slurm
#
# Array index → n_episodes mapping:
#   0 → 50
#   1 → 100
#   2 → 200

#SBATCH --job-name=cub-sweep-episodes
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=6:00:00
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
echo "SLURM job id: ${SLURM_JOB_ID:-N/A}  array task: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"

cd /home/bb720/rds/hpc-work/mlmi4_advanced_ml

source .venv/bin/activate

export PYTHONPATH="/home/bb720/rds/hpc-work:${PYTHONPATH:-}"

mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/results

# Map array index to n_episodes value
N_EPISODES_VALUES=(50 100 200)
N_EPISODES=${N_EPISODES_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "n_episodes = ${N_EPISODES}"

python src/training/train_zeroshot_precomputed.py \
    --data_root  data/cvpr2016_cub \
    --cub_root   data/CUB_200_2011 \
    --config     configs/cub_precomputed_config.yaml \
    --n_episodes ${N_EPISODES} \
    --exp_name   "cub_zeroshot_precomputed_ep${N_EPISODES}"

echo "End time: $(date)"
