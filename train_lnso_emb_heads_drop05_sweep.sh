#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-5
#SBATCH -o logs/train_emb_heads_drop05_sweep_%A_%a.out
#SBATCH -e logs/train_emb_heads_drop05_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

module purge
module load cuda/11.2.67
module load python/3.9.0

declare -a configs=(
  "exp_lnso_emb32_heads4_drop0.5"
  "exp_lnso_emb32_heads8_drop0.5"
  "exp_lnso_emb64_heads4_drop0.5"
  "exp_lnso_emb64_heads8_drop0.5"
  "exp_lnso_emb128_heads4_drop0.5"
  "exp_lnso_emb128_heads8_drop0.5"
)

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID
CONFIG_NAME="${configs[$IDX]}"

python train.py --config "$CONFIG_NAME" --load

deactivate
