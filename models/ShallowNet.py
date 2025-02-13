import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=62, dropoutRate=0.5, device=None):
        super(ShallowConvNet, self).__init__()

        # Detect and set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === First Convolution Block ===
        self.conv1 = nn.Conv1d(n_channels, 40, kernel_size=25, stride=1, padding=12, bias=False)  # Matches SEED-V
        self.spatial_conv = nn.Conv1d(40, 40, kernel_size=1, groups=40, bias=False)  # Depthwise Spatial Filtering
        self.bn1 = nn.BatchNorm1d(40)

        # Squaring Activation Function
        self.square_activation = lambda x: x ** 2  # Custom square activation

        # Pooling & Log Activation
        self.pool = nn.AvgPool1d(kernel_size=10, stride=3)
        self.log_activation = lambda x: torch.log(torch.clamp(x, min=1e-6))  # Log transformation with stability

        # Dropout
        self.dropout = nn.Dropout(dropoutRate)

        # Fully Connected Layer (Dynamically Set in Forward)
        self.fc = None  # Placeholder, will initialize dynamically in forward()

        # Move Model to Correct Device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # First Convolution & Spatial Filtering
        x = self.conv1(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)

        # Squaring Activation
        x = self.square_activation(x)

        # Pooling & Log Activation
        x = self.pool(x)
        x = self.log_activation(x)

        # Dropout
        x = self.dropout(x)

        # Flattening
        x = torch.flatten(x, start_dim=1)

        # Ensure Fully Connected Layer is Initialized Correctly
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], 5).to(self.device)  # 5 classes for SEED-V

        # Final Classification
        x = self.fc(x)
        return x
