from datasets.splitters import RandomSplitter, LNSOSplitter, KFoldSplitter
from .base_factory import BaseFactory

class SplitterFactory(BaseFactory):
    REGISTRY = {
        "random": {
            "splitter": RandomSplitter,
            "mandatory_params": ["dataset"],
            "optional_params": ["train_ratio", "shuffle"]  
        },
        "lnso": {
            "splitter": LNSOSplitter,
            "mandatory_params": ["dataset", "num_participants"],
        },
        "kfold": {
            "splitter": KFoldSplitter,
            "mandatory_params": ["dataset", "k"],
            "optional_params": ["shuffle"]  
        }
    }
    ITEM_KEY = "splitter"