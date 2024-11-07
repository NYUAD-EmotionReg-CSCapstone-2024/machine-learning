import torch
import torch.nn as nn

class BaseTransformer(nn.Module):
    def __init__(self, n_samples, n_channels, n_classes):
        super(BaseTransformer, self).__init__()
        self.embedding = nn.Linear(n_channels, 128)

        self.pe = nn.Parameter(torch.randn(n_samples, 128), requires_grad=True)
        self.transfomer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True),
            num_layers=6,
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pe
        x = self.transfomer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x