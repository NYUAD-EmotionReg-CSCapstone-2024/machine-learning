from utils.factory import BaseFactory

from .seedv import SeedVDataset
from .splitters import RandomSplitter, LNSOSplitter, KFoldSplitter

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
            "optional_params": ["train_ratio", "shuffle", "overlap_ratio"]  
        },
        "lnso": {
            "splitter": LNSOSplitter,
            "mandatory_params": ["dataset", "num_participants"],
            "optional_params": ["overlap_ratio"]  
        },
        "kfold": {
            "splitter": KFoldSplitter,
            "mandatory_params": ["dataset", "k"],
            "optional_params": ["shuffle", "overlap_ratio"]  
        }
    }
    ITEM_KEY = "splitter"