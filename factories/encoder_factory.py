import torch

from models import EEGPTEncoder
from .base_factory import BaseFactory

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
                in_shape = kwargs.get("classifier_input_shape")
                if len(in_shape) == 1:
                    return torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Linear(out_dim, in_shape[0]),
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