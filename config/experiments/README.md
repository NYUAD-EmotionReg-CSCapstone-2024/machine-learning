# Experiment Setup Example

Create a file with name `exp_{uniq_num}.yaml`. An example config is here.

```yaml
# Experiment configuration
exp_dir: "./experiments"
exp_num: 1

# Dataset configuration
dataset:
  name: "seedv"
  params:
    root_dir: /data/SEED-V/
    h5file: seedv4s0o.h5
    participants: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    sessions: [1, 2, 3]
    emotions: [0, 1, 2, 3, 4]

# Splitter configuration
splitter:
  name: "random"
  params:
    train_size: 0.8

# Model configuration
model:
  name: "atcnet"
  params:
    n_chans: 62
    n_classes: 5
    input_window_seconds: 4
    sfreq: 200

# Loss function
loss_fn: "cross_entropy"

# Optimizer configuration
optimizer:
  name: "adam"
  params:
    lr: 0.0005

# Training configuration
epochs: 1
batch_size: 256
eval_every: 1
patience: 10
```
