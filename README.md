# Datasets and Models

This repository handles the development of the machine learning model for the capstone research.

## Supported Datasets: `SEED-V`

### To build the dataset:

Create a build config file inside `config/builds` named `set_{uniq_num}.yaml` with the following:

```yaml
root_dir: "/data/SEED-V"              # Raw EEG Dataset Location
dataset: "seedv"                      # Type of dataset
outfile: "seedv4s0o"                  # Dataset outfile name
chunk_duration: 4                     # EEG Window duration length
overlap: 0.5                          # Chunk window overlap ratio

# Optional preprocessing
notch_freq: 50
bandpass_freqs:
  - 1
  - 50
resample_freq: 200
normalize: True
```

You can then run: 
```
python build.py --config {config_file_name} --overwrite {True/False}
```

For example: 
```
python build.py --config set_00 --overwrite
```

### To train the model

Create a train config file inside `config/experiments` named `exp_{uniq_num}.yaml` with:

```yaml
# Experiment configuration
exp_dir: "./experiments"
device: "cuda:0"

# Dataset configuration
dataset:
  name: "seedv"                       # Name of the dataset
  params:
    root_dir: /data/SEED-V/           # Dataset location
    h5file: seedv4s0o.h5              # Processed dataset h5 file
    participants: [1, 2]              # SEEDV - 16 Participants
    sessions: [1, 2, 3]               # SEEDV - 3 Sessions
    emotions: [0, 1, 2, 3, 4]         # SEEDV - 5 Emotion labels
    load_all: False                   # True if you have enough memory

# Model configuration
model:
  name: "atcnet"                      # Name of the model
  params:                             # Look in factory for model params
    n_chans: 62                        
    n_classes: 5
    input_window_seconds: 4
    sfreq: 200

# Splitting configuration 
splitter:
  name: random                        # Splitter method
  dataset: seedv4s0o.h5               # Dataset used for splitting
  params:                             # Look in factory for splitter params
    train_ratio: 0.8
    shuffle: True 
    
# Optimizer configuration
optimizer:
  name: "adam"                        # Type of optimizer
  params:
    lr: 0.0005                        # Learning Rate

# Scheduler configuration
scheduler:
  name: "cosine_warmup"               # Scheduler name
  params:
    T_0: 10                           # No. of epochs before restart
    T_mult: 1                         # Multiplicative factor T_0 * T_mult
    eta_min: 0.0001                   # Minimum Learning Rate

# Training configuration
epochs: 100                           # Epochs (time)
batch_size: 256                       # Batch Size per epoch
eval_every: 5                         # Validation Results every n epoch
patience: 10                          # No. of epochs before early stop
```

You can then run: 
```
python train.py --config {config_file_name} --load {True/False} --resume {True/False}
```

For example: 
```
python train.py --config exp_00 --load --resume
```

# Training Logs and Checkpoints

## Directory Structure

- `experiments/exp_{uniq_num}/`: Contains logs, checkpoints, and metrics.
  - `checkpoints/`: Contains saved model checkpoints (e.g., `model_epoch_{epoch_num}.pth`) and `latest_checkpoint.pth`.
  - `train.log`: Logs with training details (e.g., loss, accuracy, training progress).
  - `metrics.png`: Plot showing training loss, validation loss, and accuracy.

## Checkpoints

- Checkpoints are saved periodically (every `eval_every` epochs).
  - Latest checkpoint saved as `checkpoints/latest_checkpoint.pth`.

### Example checkpoint contents:

```python
{
    "epoch": 10,
    "model_state_dict": <model_weights>,
    "optimizer_state_dict": <optimizer_state>,
    "metrics": {
        "train_loss": [...],
        "val_loss": [...],
        "accuracy": [...],
    }
}
```

## Example of an Experiment Directory

```bash
experiments/
├── exp_1/
│   ├── checkpoints/
│   │   ├── model_epoch_10.pth
│   │   ├── model_epoch_20.pth
│   │   ├── latest_checkpoint.pth
│   ├── train.log
│   ├── metrics.png
└── exp_2/
    ├── checkpoints/
    │   ├── model_epoch_5.pth
    │   ├── latest_checkpoint.pth
    ├── train.log
    ├── metrics.png
```

## Using HPC

Please make sure the latest repository is in your scratch/[NETID] folder. Once that is done follow these steps for the setup: 

1. Make sure all the config/experiments you want to run are there
2. Create an empty folder titled `logs` within the repository to see log outputs if needed
3. Edit the script: `hpc/hpc_train.sh`

In the `hpc/hpc_train.sh` script, look for the following values: 
- `#SBATCH --array=0-6`: Please adjust how many jobs you would like to run simultaneously. In this example we are running 7 jobs (job 0 - job 6)
- `config_files=("exp_01" "exp_01")`: Please the name of your config files here
- Obviously change the directories to your NETID

4. If the above is all ready, then in the terminal you can run: `sbatch hpc/hpc_train.sh` to execute the job.  



