import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.seedv.channel_mappings import _channel_mappings

class ConvHead(nn.Module):
    def __init__(self, in_channels):
        super(ConvHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # --> (batch_size, 200, 4, 4)
        return x.view(x.size(0), -1) # Flatten to (batch_size, 200 * 4 * 4)

class TransformerHead(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(TransformerHead, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True),
            num_layers=n_layers,
        )

    def forward(self, x):
        x = self.transformer(x)  # Shape: (batch_size, seq_len, d_model)
        return x[:, -1, :]  # Get the last output

class ConvTransformer(nn.Module):
    def __init__(self, n_samples=200, n_classes=5, n_channels=62, n_heads=2, n_layers=6):
        super(ConvTransformer, self).__init__()
        d_model = 64
        self.fc_proj = nn.Linear(n_channels, d_model)
        self.conv_head = ConvHead(n_samples) # --> (batch_size, d_model)
        self.conv_proj = nn.Linear(16 * 4 * 4, d_model)
        self.transformer_head = TransformerHead(
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers,
            # n_samples=n_samples
        ) # --> (batch_size, d_model)
        self.fc = nn.Linear(d_model * 2, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def spatial_transform(self, x):
        batch_size, seq_len, _ = x.shape
        grid_len = 9
        spatial_data = torch.zeros((batch_size, seq_len, grid_len, grid_len), dtype=x.dtype, device=x.device)
        # use _channel_mappings to transform x to spatial_data
        for idx, (i, j) in enumerate(_channel_mappings):
            spatial_data[:, :, i, j] = x[:, :, idx]
        return spatial_data

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Convolutional path
        conv_input = self.spatial_transform(x)
        conv_out = self.conv_head(conv_input)
        conv_out = self.conv_proj(conv_out)

        # Transformer path
        transformer_input = self.fc_proj(x.view(batch_size, seq_len, -1))  # Flatten last two dims
        transformer_out = self.transformer_head(transformer_input)

        # Combine outputs
        out = torch.cat((conv_out, transformer_out), dim=1)  # Shape: (batch_size, 128)
        out = self.fc(out)  # Shape: (batch_size, n_classes)
        return self.softmax(out)  # Apply softmax for probabilities