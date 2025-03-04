#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-5
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o logs/build_overlap_%A_%a.out
#SBATCH -e logs/build_overlap_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Array of config names
declare -a configs=(
    "build_overlap_0"  # overlap=0.25
    "build_overlap_1"  # overlap=0.375
    "build_overlap_2"  # overlap=0.5
    "build_overlap_3"  # overlap=0.625
    "build_overlap_4"  # overlap=0.75
    "build_overlap_5"  # overlap=0.875
)

# Load required modules
module purge
module load python/3.9.0

# Change to project directory
cd /scratch/dt2307/machine-learning

# Activate virtual environment
source /scratch/dt2307/venv/bin/activate

# Run builder script
python build.py --config ${configs[$SLURM_ARRAY_TASK_ID]}

# Deactivate environment
deactivate