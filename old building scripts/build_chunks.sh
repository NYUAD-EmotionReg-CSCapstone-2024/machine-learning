#!/bin/bash

#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-8
#SBATCH -o logs/build_chunk_%A_%a.out
#SBATCH -e logs/build_chunk_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "build_chunk_0"  # 2s chunks
    "build_chunk_1"  # 3s chunks
    "build_chunk_2"  # 4s chunks
    "build_chunk_3"  # 5s chunks
    "build_chunk_4"  # 6s chunks
    "build_chunk_5"  # 7s chunks
    "build_chunk_6"  # 8s chunks
    "build_chunk_7"  # 9s chunks
    "build_chunk_8"  # 10s chunks
)

# Load required modules
module purge
module load python/3.9.0

# Change to project directory
cd /scratch/dt2307/machine-learning

# Activate virtual environment
source /scratch/dt2307/venv/bin/activate

# Run builder script with config from array
python build.py --config ${configs[$SLURM_ARRAY_TASK_ID]}

# Deactivate environment
deactivate