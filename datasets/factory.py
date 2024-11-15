from utils.factory import BaseFactory

from .seedv import SeedVDataset
from .splitters import RandomSplitter, LNSOSplitter

class DatasetFactory(BaseFactory):
    REGISTRY = {
        "seedv": {
            "dataset": SeedVDataset,
            "mandatory_params": ["root_dir", "h5file"],
            "optional_params": ["transform", "participants", "sessions", "emotions", "channels", "load"]
        }
    }
    ITEM_KEY = "dataset"

class SplitterFactory(BaseFactory):
    REGISTRY = {
        "random": {
            "splitter": RandomSplitter,
            "mandatory_params": ["dataset"],
            "optional_params": ["train_ratio", "shuffle"]
        },
        "lnso": {
            "splitter": LNSOSplitter,
            "mandatory_params": ["dataset"],
            "optional_params": ["num_participants", "shuffle"]
        }
    }
    ITEM_KEY = "splitter"