import torch
from torchviz import make_dot
from models.base import BaseModel

# Create model instance
model = BaseModel(n_samples=250, n_classes=5, n_channels=62, n_heads=8, n_layers=4)

# Create a dummy input tensor
dummy_input = torch.randn(32, 250, 62)
y = model(dummy_input)

# Generate the graph
dot = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

dot.format = 'png'
dot.render('./model_graph')