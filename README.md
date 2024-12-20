# Datasets and Models

This repository handles the development of the machine learning model for the capstone research.

## Supported Datasets: `SEED-V`

### To build the dataset

Create a build config file inside `config/builds` named `set_{uniq_num}.yaml` with the following:

```yaml
root_dir: "/data/SEED-V"
dataset: "seedv"
outfile: "seedv4s0o"
chunk_duration: 4

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
  name: "seedv"
  params:
    root_dir: /data/SEED-V/
    h5file: seedv4s0o.h5
    participants: [1, 2]
    sessions: [1, 2, 3]
    emotions: [0, 1, 2, 3, 4]
    load_all: False # make it True if you have enough memory

# Model configuration
model:
  name: "atcnet"
  params:
    n_chans: 62
    n_classes: 5
    input_window_seconds: 4
    sfreq: 200

# Splitting configuration 
splitter:
  name: random
  dataset: seedv4s0o.h5
  params:
    train_ratio: 0.8
    overlap_ratio: 0.5
    shuffle: True 
    
# Optimizer configuration
optimizer:
  name: "adam"
  params:
    lr: 0.0005

# Training configuration
epochs: 100
batch_size: 256
eval_every: 5
patience: 10
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
