#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1     # Request 1 GPU (any type available in 'nvidia' partition)
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/train_anygpu_%A_%a.out
#SBATCH -e logs/train_anygpu_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "exp_lnso_conformer_overlap_lr1"  # LR=0.001 in YAML
    "exp_lnso_conformer_overlap_lr2"  # LR=0.0005 in YAML
    "exp_lnso_conformer_overlap_lr3"  # LR=0.0001 in YAML
)

# 1) Load required modules
module purge
module load cuda/11.2.67
module load python/3.9.0

# 2) Move to project directory
cd /scratch/dt2307/machine-learning

# 3) Activate virtual environment
source /scratch/dt2307/venv/bin/activate

# 4) Optional: Debug GPU setup
python -c "
import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# 5) Run training for each config in the array
CONFIG_NAME="${configs[$SLURM_ARRAY_TASK_ID]}"
echo "=== Running config: $CONFIG_NAME on ANY GPU ==="

python train.py --config "$CONFIG_NAME" --load

# 6) Deactivate environment
deactivate