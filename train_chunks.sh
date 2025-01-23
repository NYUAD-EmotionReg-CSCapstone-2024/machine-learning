#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-8
#SBATCH -o logs/train_chunk_%A_%a.out
#SBATCH -e logs/train_chunk_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "exp_chunk_0"  # 2s chunks
    "exp_chunk_1"  # 3s chunks
    "exp_chunk_2"  # 4s chunks
    "exp_chunk_3"  # 5s chunks
    "exp_chunk_4"  # 6s chunks
    "exp_chunk_5"  # 7s chunks
    "exp_chunk_6"  # 8s chunks
    "exp_chunk_7"  # 9s chunks
    "exp_chunk_8"  # 10s chunks
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