#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/reproduce_40_%j.out
#SBATCH -e logs/reproduce_40_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

# Debug GPU
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== Reproducing LN-SO ~40% result with no-overlap, LR=0.001 ==="

python train.py --config exp_lnso_conformer_no_overlap_lr001 --load

deactivate
