#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-3    # 4 new durations
#SBATCH -o logs/build_long_chunk_%A_%a.out
#SBATCH -e logs/build_long_chunk_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "build_chunk_9"   # 11s chunks
    "build_chunk_10"  # 12s chunks
    "build_chunk_11"  # 13s chunks
    "build_chunk_12"  # 14s chunks
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