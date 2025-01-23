#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 4-00:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Load required modules
module purge
module load cuda/11.2.67
module load python/3.9.0

# Change to the project directory
cd /scratch/dt2307/machine-learning

# Activate preconfigured virtual environment
source /scratch/dt2307/venv/bin/activate

# Debug GPU setup
python -c "
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Execute training script
python train.py --config exp_14 --load

# Deactivate environment
deactivate
