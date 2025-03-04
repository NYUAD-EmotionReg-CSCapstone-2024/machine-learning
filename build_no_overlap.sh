#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o logs/build_no_overlap_%j.out
#SBATCH -e logs/build_no_overlap_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

# Load required modules
module purge
module load python/3.9.0

# Change to project directory
cd /scratch/dt2307/machine-learning

# Activate virtual environment
source /scratch/dt2307/venv/bin/activate

# Run builder script
python build.py --config build_no_overlap

# Deactivate environment
deactivate