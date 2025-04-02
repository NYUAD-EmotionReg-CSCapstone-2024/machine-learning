from .base_factory import BaseFactory

from models import ATCNet, ERTNet, ConvTransformer, EEGNet, MLP, ShallowConvNet, CNN_BiLSTM, GRUNet, DeepConvNet, ModifiedATCNet

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
        },
        "modified_atcnet": {
            "model": ModifiedATCNet,
            "mandatory_params": ["n_chans", "timestamps", "n_outputs"]
        }
    }
    ITEM_KEY = "model"