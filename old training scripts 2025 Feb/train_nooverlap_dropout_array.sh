#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-3
#SBATCH -o logs/train_nooverlap_drop_%A_%a.out
#SBATCH -e logs/train_nooverlap_drop_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

declare -a configs=(
  "exp_lnso_nooverlap_d30_e32"
  "exp_lnso_nooverlap_d30_e24"
  "exp_lnso_nooverlap_d40_e32"
  "exp_lnso_nooverlap_d40_e24"
)

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

CONFIG_NAME="${configs[$SLURM_ARRAY_TASK_ID]}"
echo "=== Running config: $CONFIG_NAME ==="

python train.py --config "$CONFIG_NAME" --load

deactivate
