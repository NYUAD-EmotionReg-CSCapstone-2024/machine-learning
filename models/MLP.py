import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_classes=5, n_channels=60):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Define two linear layers
        self.linear = nn.Sequential(
            nn.Linear(n_channels*8, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        return self.linear(x)