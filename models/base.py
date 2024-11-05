import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, n_channels, embed_dim, num_heads, num_classes):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(n_channels, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=2
        )
        self.fc = nn.Linear(embed_dim, num_classes)  # Change input to embed_dim for the final layer

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, n_channels) -> (batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # Shape: (batch_size, embed_dim) -> (sequence_length, batch_size, embed_dim)
        x = self.encoder(x)  # Forward through transformer
        x = x.mean(dim=0)  # Average pooling across sequence length
        x = self.fc(x)  # Final classification layer
        return x