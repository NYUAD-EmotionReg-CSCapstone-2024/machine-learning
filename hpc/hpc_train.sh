#!/bin/bash

#SBATCH --job-name=train_model         
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 4-00:00:00
#SBATCH --output=/scratch/ap7146/machine-learning/logs/train_%A_%a.out   
#SBATCH --error=/scratch/ap7146/machine-learning/logs/train_%A_%a.err    
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=ap7146@nyu.edu
#SBATCH --array=0-1

# Initialize the modules system
source /etc/profile.d/modules.sh

# Load required modules
module purge
module load cuda/11.8.0
module load python/3.11.3

# Activate the virtual environment
source /scratch/ap7146/machine-learning/venv/bin/activate

# Navigate to the directory containing build.py and config files
cd /scratch/ap7146/machine-learning/

# Debug GPU setup
python -c "
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Array of configuration file names
config_files=("ATCNet_2p_0o" "ERTNet_2p_0o")

# Get the config file corresponding to the current task ID
config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

# Execute the train.py script with the appropriate config file
python train.py --config "$config_file"