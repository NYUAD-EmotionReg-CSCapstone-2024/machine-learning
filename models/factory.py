import torch
from torch.optim import lr_scheduler 

from utils.factory import BaseFactory

from .ATCNet import ATCNet
from .ERTNet import ERTNet
from .ConvTrans import ConvTransformer
from .EEGNet import EEGNet
from .MLP import MLP
from .ShallowNet import ShallowConvNet
from .CNN_BiLSTM import CNN_BiLSTM
from .GRUNet import GRUNet
from .DeepConvNet import DeepConvNet
from .EEGPT import EEGPTEncoder

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
        "mlp": {
            "model": MLP,
            "mandatory_params": ["n_classes", "n_channels"],
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


class EncoderFactory(BaseFactory):
    """Factory for creating encoders that wrap models"""
    
    REGISTRY = {
        "eegpt": {
            "encoder": EEGPTEncoder,
            "mandatory_params": ["ckpt_path"],
            "optional_params": ["window", "freq", "patch_size", "patch_stride", "embed_num", "embed_dim"]
        }
    }
    ITEM_KEY = "encoder"

    @classmethod
    def wrap(cls, name, model, **kwargs):
        if name.lower() not in cls.REGISTRY:
            raise ValueError(f"Encoder {name} not found in EncoderFactory.")
        
        encoder = cls.create(name, **kwargs)

        class WrappedModel(torch.nn.Module):
            def __init__(self, encoder, model):
                super().__init__()
                self.encoder = encoder
                if kwargs.get("freeze", False):
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                self.connector = self._get_connector()
                self.model = model

            def _get_connector(self):
                out_shape = self.encoder.out_shape
                out_dim = out_shape[0] * out_shape[1]
                in_shape = 8 * 60
                if isinstance(in_shape, int):
                    return torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Linear(out_dim, in_shape),
                        torch.nn.ReLU()
                    )
                elif len(in_shape) == 2:
                    return torch.nn.Conv2d(
                        in_channels=1,
                        out_channels=in_shape[0],
                        kernel_size=(1, 1),
                        stride=1
                    )
                else:
                    raise ValueError(f"Invalid input shape {in_shape}")

            def forward(self, x):
                x = self.encoder(x)
                x = self.connector(x)
                x = self.model(x)
                return x

        wrapped_model = WrappedModel(encoder, model)
        return wrapped_model