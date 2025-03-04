#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -t 4-00:00:00
#SBATCH -o logs/train_lnso_ffexp4_%j.out
#SBATCH -e logs/train_lnso_ffexp4_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=dt2307@nyu.edu

module purge
module load cuda/11.2.67
module load python/3.9.0

cd /scratch/dt2307/machine-learning
source /scratch/dt2307/venv/bin/activate

CONFIG_NAME="exp_lnso_emb32_heads4_ff4"
python train.py --config "$CONFIG_NAME" --load

deactivate
