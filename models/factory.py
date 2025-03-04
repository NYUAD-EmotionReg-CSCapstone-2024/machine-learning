import torch
from torch.optim import lr_scheduler

from utils.factory import BaseFactory

from .atcnet import ATCNet
from .ERTNet import ERTNet
from .conformer import Conformer
from .conformer_downsized import DownsizedConformer  # <--- new import

class ModelFactory(BaseFactory):
    REGISTRY = {
        "atcnet": {
            "model": ATCNet,
            "mandatory_params": ["n_chans", "n_classes", "input_window_seconds", "sfreq"],
        },
        "ertnet": {
            "model": ERTNet,
            "mandatory_params": [],
        },
        "conformer": {
            "model": Conformer,
            "mandatory_params": ["emb_size", "depth", "n_classes"],
            "optional_params": [
                "num_heads", "ff_expansion_factor", "conv_kernel_size",
                "embed_dropout", "block_dropout"
            ]
        },
        "downsized_conformer": {
            "model": DownsizedConformer,
            "mandatory_params": ["emb_size", "depth", "n_classes"],
            "optional_params": [
                "num_heads", "ff_expansion_factor", "conv_kernel_size",
                "embed_dropout", "block_dropout"
            ]
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


class SchedulerFactory(BaseFactory):
    REGISTRY = {
        "cosine_warmup": {
            "scheduler": lr_scheduler.CosineAnnealingWarmRestarts,
            "mandatory_params": ["T_0", "eta_min"],
            "optional_params": ["T_mult"]
        }
    }
    ITEM_KEY = "scheduler"


    