# Experiment Setup Example

Create a file with name `exp_{uniq_num}.yaml`. An example config is here.

```yaml
root_dir: "/data/SEED-V"
exp_dir: "./experiments"
exp_num: 12
participants:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
  - 10
  - 11
  - 12
  - 13
  - 14
  - 15
  - 16
sessions:
  - 1
  - 2
  - 3
emotions:
  - "happy"
  - "sad"
  - "fear"
  - "neutral"
  - "angry"
dataset: "seedv4s0o"
model: "atcnet"
epochs: 100
trainset_size: 14
testset_size: 2
batch_size: 256
lr: 0.0005
optimizer: "adam"
loss_fn: "cross_entropy"
save_every: 10
eval_every: 5
```
