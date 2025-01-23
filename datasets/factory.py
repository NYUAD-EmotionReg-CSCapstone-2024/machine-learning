from utils.factory import BaseFactory
from .seedv import SeedVDataset
from .splitters import RandomSplitter, LNSOSplitter

class DatasetFactory(BaseFactory):
    """
    Factory class for creating datasets. Includes debug statements to trace the dataset creation process.
    """
    REGISTRY = {
        "seedv": {
            "dataset": SeedVDataset,
            "mandatory_params": ["root_dir", "h5file"],
            "optional_params": ["transform", "participants", "sessions", "emotions", "channels", "load"]
        }
    }
    ITEM_KEY = "dataset"

    @classmethod
    def create(cls, name, **kwargs):
        """
        Create a dataset instance with the given name and parameters.
        Logs debug messages before and after dataset creation.
        """
        # Debug: Log the dataset creation attempt
        print(f"[DEBUG] DatasetFactory: Attempting to create dataset '{name}' with parameters: {kwargs}")
        
        # Create the dataset using the parent class's method
        dataset = super().create(name, **kwargs)
        
        # Debug: Confirm the dataset was created
        print(f"[DEBUG] DatasetFactory: Dataset '{name}' created successfully.")
        
        return dataset


class SplitterFactory(BaseFactory):
    """
    Factory class for creating dataset splitters. Includes debug statements to trace the splitter creation process.
    """
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

    @classmethod
    def create(cls, name, **kwargs):
        """
        Create a splitter instance with the given name and parameters.
        Logs debug messages before and after splitter creation.
        """
        # Debug: Log the splitter creation attempt
        print(f"[DEBUG] SplitterFactory: Attempting to create splitter '{name}' with parameters: {kwargs}")
        
        # Create the splitter using the parent class's method
        splitter = super().create(name, **kwargs)
        
        # Debug: Confirm the splitter was created
        print(f"[DEBUG] SplitterFactory: Splitter '{name}' created successfully.")
        
        return splitter
