import torch

from utils.factory import BaseFactory

from .atcnet import ATCNet
from .ERTNet import ERTNet

class ModelFactory(BaseFactory):
    REGISTRY = {
        "atcnet": {
            "model": ATCNet,
            "mandatory_params": ["n_chans", "n_classes", "input_window_seconds", "sfreq"],
        },
        "ertnet": {
            "model": ERTNet,
            "mandatory_params": [],
        }
    }
    ITEM_KEY = "model"


class OptimizerFactory(BaseFactory):
    REGISTRY = {
        "adam": {
            "optimizer": torch.optim.Adam,
            "mandatory_params": ["lr"],
        },
        "adamw": {
            "optimizer": torch.optim.AdamW,
            "mandatory_params": ["lr"],
        }
    }
    ITEM_KEY = "optimizer"