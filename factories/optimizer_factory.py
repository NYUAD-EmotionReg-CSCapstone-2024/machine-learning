import torch
from .base_factory import BaseFactory

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