#!/bin/bash
module purge
module load cuda/11.2.67
module load python/3.9.0

# Create and activate virtual environment
python -m venv /scratch/dt2307/venv
source /scratch/dt2307/venv/bin/activate

# Upgrade pip and install compatible packages
pip install --upgrade pip
pip install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    h5py==3.12.1 \
    mne==1.8.0 \
    torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 \
    scikit-learn==1.5.2 \
    braindecode==0.8.1 \
    tensorboard==2.18.0 \
    torchinfo==1.8.0 \
    einops==0.8.0 \
    tqdm==4.66.6 \
    pyyaml==6.0.2

echo "Environment setup complete."
deactivate
