#!/bin/bash

SBATCH -p nvidia
SBATCH --gres=gpu:a100:1
SBATCH -c 4
SBATCH -N 1
SBATCH --mem=64G
SBATCH -t 4-00:00:00
SBATCH -o logs/train_%j.out
SBATCH -e logs/train_%j.err
SBATCH --mail-type=END,FAIL,TIME_LIMIT
SBATCH --mail-user=ap7146@nyu.edu
SBATCH --array=1-10

# Load required modules
module purge
module load cuda/12.6
module load python/3.12.5

# Change to the project directory
cd /scratch/ap7146/machine-learning

# Activate preconfigured virtual environment
source /scratch/ap7146/venv/bin/activate

# Debug GPU setup
python -c "
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Execute script
# python build.py --config exp_${SLURM_ARRAY_TASK_ID}
srun $(head -n $SLURM_ARRAY_TASK_ID jobs.txt | tail -n 1)


# Deactivate environment
deactivate