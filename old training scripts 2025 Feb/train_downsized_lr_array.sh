#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-4
#SBATCH -o logs/downsized_lr_%A_%a.out
#SBATCH -e logs/downsized_lr_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# We'll test 5 LRs (0.0007, 0.0008, 0.0009, 0.0010, 0.0011)
declare -a configs=(
  "exp_downsized_lr7"
  "exp_downsized_lr8"
  "exp_downsized_lr9"
  "exp_downsized_lr10"
  "exp_downsized_lr11"
)

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

# Optional debug
python -c "
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

CONFIG_NAME="${configs[$SLURM_ARRAY_TASK_ID]}"
echo "=== Training downsized model with config: $CONFIG_NAME ==="

python train.py --config "$CONFIG_NAME" --load

deactivate
