import torch
from torch.optim import lr_scheduler 

from utils.factory import BaseFactory

from .ATCNet import ATCNet
from .ERTNet import ERTNet
from .ConvTrans import ConvTransformer
from .EEGNet import EEGNet
from .EEGPTCausal import EEGPTCausal
from .ShallowNet import ShallowConvNet
from .CNN_BiLSTM import CNN_BiLSTM
from .GRUNet import GRUNet
from .DeepConvNet import DeepConvNet

class ModelFactory(BaseFactory):
    REGISTRY = {
        "atcnet": {
            "model": ATCNet,
            "mandatory_params": ["n_chans", "n_classes", "input_window_seconds", "sfreq"],
        },
        "ertnet": {
            "model": ERTNet,
            "mandatory_params": ["n_channels", "kernLength", "F1", "F2", "D", "heads", "dropoutRate"],
        },
        "conv_transformer": {
            "model": ConvTransformer,
            "mandatory_params": ["n_channels", "n_heads", "n_layers"],
        },
        "eegnet": {
            "model": EEGNet,
            "mandatory_params": ["n_channels", "kernLength", "F1", "F2", "D"],
        },
        "shallownet": {
            "model": ShallowConvNet,
            "mandatory_params": ["n_channels", "dropoutRate"],
        },
        "cnn_bilstm": {
            "model": CNN_BiLSTM,
            "mandatory_params": ["n_channels", "kernLength", "F1", "num_lstm", "F2", "D", "dropoutRate"],
        },
        "grunet": {
            "model": GRUNet,
            "mandatory_params": ["dropoutRate", "L1", "L2"],
        },
        "deepconvnet": {
            "model": DeepConvNet,
            "mandatory_params": ["n_channels", "dropoutRate"],
        },
        "eegpt": {
            "model": EEGPTCausal,
            "mandatory_params": ["ckpt_path", "window", "freq", "patch_size", "patch_stride"],
            "optional_params": ["embed_num", "embed_dim", "num_classes"]
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