# NYUAD Emotion Regulation Capstone - Deep Learning Component

This repository handles the development of the machine learning model pipeline for our capstone research on emotion regulation

### Supported Datasets: `SEED-V`

# Setup: 

For our setup, you will need to create a new directory known as `./config`. In this folder, we will place all the necessary configurations for all the steps required for the setup of our pipeline. 

## 1. Builds

Builds are used to create a dataset build which have been preprocessed and are optimized for training the models. 

- To create a build config file, you must create a new directory inside the `./config` folder called `config/builds`. 

- Then you can name your build file however you like: `{build_name}.yaml` with the following parameters:

```yaml
# Build Configuration Settings: 
root_dir: "/data/SEED-V"              # Raw EEG Dataset Location
dataset: "seedv"                      # Type of dataset
outfile: "seedv4s0o"                  # Dataset outfile name
chunk_duration: 4                     # EEG Window duration length
overlap: 0                            # Chunk window overlap ratio

# Optional preprocessing
notch_freq: 50
bandpass_freqs:
  - 1
  - 50
resample_freq: 256
normalize: True
```

> Note: Please refer to the dataset factory to see what values you can use for exploration under `./factories`

You can then execute the build by running: 
```
python build.py --config {config_file_name} --overwrite {True/False}
```

For example: 
```
python build.py --config build_00 --overwrite
```

## 2. Training

Training is the main component to which we can specify which model we want to train based on the build we have created.

- To start training you must create a new directory inside the `./config` folder called `config/experiments`. 

- Then you can name your train file however you like: `{exp_name}.yaml` with the following parameters:

```yaml
# Experiment Configuration Settings: 
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
  enocder:                              # Encoder head from Universal LEMs (optional)
  name: "eegpt"                         # Name of LEM with (pre-trained weights file path)
  params: 
    ckpt_path: ./EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
    freeze: True                        # Freezes weights or not
    classifier_input_shape: [62, 2400]  # Input shape for the classifier   
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

> Note: Please refer to the model, encoder, optimizer, scheduler and splitter factories to see what values you can use for exploration under `./factories`

You can then execute the training by running the command: 
```
python train.py --config {config_file_name} --load {True/False} --resume {True/False}
```

For example: 
```
python train.py --config exp_00 --load --resume
```

## 3. Interface

The interface is what connects the model with the real-time EEG data collection headset from X.on. This section is all about collecting the data in real time and feeding it into the model for a prediction inference. 

- To start the interface you must create a new directory inside the `./config` folder called `config/interface`. 

- Then you can name your train file however you like: `{int_name}.yaml` with the following parameters:

```yaml
# Interface Configuration Settings: 
model_filepath: "./experiments/eegpt/eegpt_atcnet_trained_model.pth"
window_size: 4                
sample_frequency: 256         
polling_frequency: 0.01       
``` 

You can then execute the training by running the command: 
```
python interface.py --config {config_file_name}
```

For example: 
```
python interface.py --config int_00
```

# Training Logs and Checkpoints
## Directory Structure

When executing a training, a new directory called `./experiments` will display at the repository level which will contain information such as logs, checkpoints, metrics, trained weights, etc. 

- `experiments/{exp_name}/`:
  - `checkpoints/`: Contains saved model checkpoints for resuming (e.g., `model_epoch_{epoch_num}.pth`) and `latest_checkpoint.pth`.
  - `train.log`: Logs with training details (e.g., loss, accuracy, training progress).
  - `metrics.png`: Plot showing training loss, validation loss, and accuracy.
  - `{model_name}_trained_model.pth`: Trained weights of the final model 

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
│   ├── atcnet_trained_model.pth
└── exp_2/
    ├── checkpoints/
    │   ├── model_epoch_5.pth
    │   ├── latest_checkpoint.pth
    ├── train.log
    ├── metrics.png
    ├── ertnet_trained_model.pth
```

## Training with NYUAD HPC

### Setup: 

1. Please make sure the latest repository is in your scratch/[NETID] folder. 
2. Make sure all the `./config` files you want to run are there
3. Create an empty folder titled `logs` within the repository to see log outputs

### Editing the HPC Scripts: 

We have a folder containing pre-made hpc scripts in `./hpc/`. In this folder you will find two main scripts which are `hpc_build.sh` and `hpc_train.sh`

The build script is used for building the datasets and the train script is for training the models. Both of which have a similar structure.

To edit the scripts to your desire, make sure to change the following: 

1. Make sure the paths include your `NETID`

2. `#SBATCH --array=0-6`: Please adjust how many jobs you would like to run simultaneously. In this example we are running 7 jobs (job 0 - job 6)

3. `config_files=("exp_00" "exp_01" "exp_02")`: Please the name of your config files here

### Executing the script:
If the above is all ready, then in HPC terminal you can run: `sbatch hpc/hpc_train.sh` to execute the job.  



