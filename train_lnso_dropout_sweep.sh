#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-9
#SBATCH -o logs/train_dropout_sweep_%A_%a.out
#SBATCH -e logs/train_dropout_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

module purge
module load cuda/11.2.67
module load python/3.9.0

declare -a configs=(
  "exp_lnso_dropout_0.1"
  "exp_lnso_dropout_0.15"
  "exp_lnso_dropout_0.2"
  "exp_lnso_dropout_0.25"
  "exp_lnso_dropout_0.3"
  "exp_lnso_dropout_0.35"
  "exp_lnso_dropout_0.4"
  "exp_lnso_dropout_0.45"
  "exp_lnso_dropout_0.5"
  "exp_lnso_dropout_0.55"
)

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID
CONFIG_NAME="${configs[$IDX]}"

python train.py --config "$CONFIG_NAME" --load

deactivate
