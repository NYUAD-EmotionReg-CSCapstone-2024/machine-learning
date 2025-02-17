import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, base_model, num_classes, freeze_base=True):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get the dimension of the last layer's output
        with torch.no_grad():
            dummy_input = torch.randn(1, *base_model.input_shape)
            output = base_model(dummy_input)
            feature_dim = output.shape[1]
        
        # Add linear probe
        self.probe = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.base_model(x)
        return self.probe(features)
    
    def freeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
