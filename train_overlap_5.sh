#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=256G      # Increased memory
#SBATCH -t 4-00:00:00
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o logs/train_overlap_5_%j.out
#SBATCH -e logs/train_overlap_5_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

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

# Execute training script for just overlap_5
python train.py --config exp_overlap_5 --load

# Deactivate environment
deactivate