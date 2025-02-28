from .base_factory import BaseFactory
from datasets.seedv import SeedVDataset

class DatasetFactory(BaseFactory):
    REGISTRY = {
        "seedv": {
            "dataset": SeedVDataset,
            "mandatory_params": ["root_dir", "h5file", "device"],
            "optional_params": ["transform", "participants", "sessions", "emotions", "channels", "load"]
        }
    }
    ITEM_KEY = "dataset"