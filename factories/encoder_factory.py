import torch
import torch.nn as nn

from models import EEGPTEncoder
from .base_factory import BaseFactory

# Define WrappedModel as a top-level class for pickling
class WrappedModel(torch.nn.Module):
    def __init__(self, encoder, model, **kwargs):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder if requested
        if kwargs.get("freeze", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.connector = self._get_connector(kwargs)
        self.model = model
        
    def _get_connector(self, kwargs):
        """
        Decide how to connect the encoder's output to the model based on
        'classifier_input_shape'.
        """
        out_shape = self.encoder.out_shape  # e.g. (31*4, 512) => (124, 512)
        out_dim = out_shape[0] * out_shape[1]
        in_shape = kwargs.get("classifier_input_shape")
        encoder_name = kwargs.get("name", "")
        
        # If user gave a 1D shape => Flatten + Linear
        if len(in_shape) == 1:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_dim, in_shape[0]),
                nn.ReLU()
            )
        
        elif len(in_shape) == 2 and encoder_name == "eegpt":
            # x.shape = B=256, n=31, e=4, d=512
            # Reshape to [B, 124, 512]
            # View to [B, 62, 1024] (Input Shape)
            # Unsqueece to [B, 62, 1024, 1] for ATCNet
            # return lambda x: x.reshape(
            #     x.shape[0], 
            #     x.shape[1] * x.shape[2], 
            #     x.shape[3]
            # ).view(                         
            #     x.shape[0], 
            #     in_shape[0], 
            #     in_shape[1]
            # ).unsqueeze(-1)

            # Reshape to [256, 512, 124]
            return lambda x: x.reshape(
                x.shape[0],  # Batch
                x.shape[3],  # Features dimension (512)
                x.shape[1] * x.shape[2]  # Time dimension (31*4 = 124)
            )

        # Commented out until labram works
        # elif len(in_shape) == 2 and encoder_name == "labram": 
        #     return nn.Identity()

        else:
            raise ValueError(f"Invalid input shape {in_shape}")
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.connector(x)
        x = self.model(x)
        return x

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
        # Pass the name along to the WrappedModel
        kwargs["name"] = name
        return WrappedModel(encoder, model, **kwargs)