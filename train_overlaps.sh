#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH --array=0-5
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o logs/train_overlap_%A_%a.out
#SBATCH -e logs/train_overlap_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "exp_overlap_0"  # overlap=0.25
    "exp_overlap_1"  # overlap=0.375
    "exp_overlap_2"  # overlap=0.5
    "exp_overlap_3"  # overlap=0.625
    "exp_overlap_4"  # overlap=0.75
    "exp_overlap_5"  # overlap=0.875
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

# Execute training script
python train.py --config ${configs[$SLURM_ARRAY_TASK_ID]} --load

# Deactivate environment
deactivate