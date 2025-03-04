#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-4
#SBATCH -o logs/train_sched_lr_%A_%a.out
#SBATCH -e logs/train_sched_lr_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

declare -a configs=(
  "exp_lnso_conf_scheduler_lr7"   # 0.0007
  "exp_lnso_conf_scheduler_lr8"   # 0.0008
  "exp_lnso_conf_scheduler_lr9"   # 0.0009
  "exp_lnso_conf_scheduler_lr10"  # 0.0010
  "exp_lnso_conf_scheduler_lr11"  # 0.0011
)

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

# Optional debug
python -c "
import torch
print('Torch version:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

CONFIG_NAME="${configs[$SLURM_ARRAY_TASK_ID]}"
echo "=== Training with config: $CONFIG_NAME ==="

python train.py --config "$CONFIG_NAME" --load

deactivate
