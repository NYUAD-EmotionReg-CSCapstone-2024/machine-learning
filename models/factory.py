from .atcnet import ATCNet
from .ERTNet import ERTNet

class ModelFactory:
    MODELS = {
        "atcnet": {
            "model": ATCNet,
            "mandatory_params": ["n_chans", "n_classes", "input_window_seconds", "sfreq"]
        },
        "ertnet": {
            "model": ERTNet,
            "mandatory_params": [],
        }
    }

    @classmethod
    def get_model(cls, model_name, **kwargs):
        if model_name not in cls.MODELS:
            raise ValueError(f"Invalid model: {model_name}")
        
        config = cls.MODELS[model_name]
        
        # Check for mandatory parameters
        for param in config["mandatory_params"]:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")
        
        # Filter out valid parameters
        valid_params = {k: kwargs[k] for k in config["mandatory_params"] if k in kwargs}
        
        return config["model"](**valid_params)