#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1  # Request any GPU
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o logs/train_loso_%j.out
#SBATCH -e logs/train_loso_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning

source /scratch/dt2307/venv/bin/activate

# Debug GPU setup
python -c "
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Run training
python train.py --config exp_loso --load

deactivate