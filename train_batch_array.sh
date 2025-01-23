#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-6
#SBATCH -o logs/train_batch_%A_%a.out
#SBATCH -e logs/train_batch_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "exp_batch_0"  # batch=128
    "exp_batch_1"  # batch=256
    "exp_batch_2"  # batch=512
    "exp_batch_3"  # batch=1024 (original)
    "exp_batch_4"  # batch=2048
    "exp_batch_5"  # batch=4096
    "exp_batch_6"  # batch=8192
)

# Load required modules
module purge
module load cuda/11.2.67
module load python/3.9.0

# Change to project directory
cd /scratch/dt2307/machine-learning

# Activate virtual environment
source /scratch/dt2307/venv/bin/activate

# Debug GPU setup
python -c "
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Execute training script with config from array
python train.py --config ${configs[$SLURM_ARRAY_TASK_ID]} --load

# Deactivate environment
deactivate