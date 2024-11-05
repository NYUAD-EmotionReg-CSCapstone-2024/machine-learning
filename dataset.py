import os

from datasets.seedv.dataset import SeedVDataset

ROOT = "/data/SEED-V"
H5FILE = "seedv.h5"

seedv = SeedVDataset(root=ROOT, h5file=H5FILE)

breakpoint()