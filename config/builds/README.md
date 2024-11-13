# Dataset Builder Config Example

Create a file with name `set_{uniq_num}.yaml`. An example config is here.

```yaml
root_dir: "/data/SEED-V"
dataset: "seedv"
outfile: "seedv4s0o"
chunk_duration: 4
overlap: 0
notch_freq: 50
bandpass_freqs:
  - 1
  - 50
resample_freq: 200
normalize: True
```
