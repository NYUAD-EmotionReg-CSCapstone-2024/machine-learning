from .seedv import SeedVDataset
from .splitters import RandomSplitter, KFoldSplitter, LNSOSplitter

class DatasetFactory:
    DATASETS = {
        "seedv": {
            "dataset": SeedVDataset,
            "mandatory_params": ["root_dir", "h5file"],
            "optional_params": ["transform", "participants", "sessions", "emotions", "channels", "load_all"]
        }
    }

    @classmethod
    def get_dataset(cls, dataset_name, **kwargs):
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Invalid dataset: {dataset_name}")
        
        config = cls.DATASETS[dataset_name]
        
        # Check for mandatory parameters
        for param in config["mandatory_params"]:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")
        
        # Filter out valid parameters
        valid_params = {k: kwargs[k] for k in config["mandatory_params"] + config["optional_params"] if k in kwargs}
        
        return config["dataset"](**valid_params)


class SplitterFactory:
    SPLITTERS = {
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
        # "kfoldsplitter": {
        #     "splitter": KFoldSplitter,
        #     "mandatory_params": ["dataset"],
        #     "optional_params": ["k", "shuffle"]
        # },
    }

    @classmethod
    def get_splitter(cls, splitter_name, **kwargs):
        if splitter_name not in cls.SPLITTERS:
            raise ValueError(f"Invalid splitter: {splitter_name}")
        
        config = cls.SPLITTERS[splitter_name]
        
        # Check for mandatory parameters
        for param in config["mandatory_params"]:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")
        
        # Filter out valid parameters
        valid_params = {k: kwargs[k] for k in config["mandatory_params"] + config["optional_params"] if k in kwargs}
        
        return config["splitter"](**valid_params)