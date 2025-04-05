#!/bin/bash
#SBATCH --job-name=build_datasets       # Job name
#SBATCH --array=0-1                    # Array range (11 tasks: 0 to 10)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=64G                        # Memory per task
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=/scratch/ap7146/machine-learning/logs/build_%A_%a.out
#SBATCH --error=/scratch/ap7146/machine-learning/logs/build_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=ap7146@nyu.edu


# Initialize the modules system
source /etc/profile.d/modules.sh

# Load the Python module
module purge
module load python/3.11.3

# Activate the virtual environment
source /scratch/ap7146/machine-learning/venv/bin/activate

# Navigate to the directory containing build.py and config files
cd /scratch/ap7146/machine-learning/

# Array of configuration file names
config_files=("seedv14s50o" "seedv12s50o")

# Get the config file corresponding to the current task ID
config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

# Execute the build.py script with the appropriate config file
python build.py --config "$config_file"
