import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, samples=2400, dropoutRate=0.5, device=None):
        super(DeepConvNet, self).__init__()

        # Detect and set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # First Convolution Block
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=10, stride=1, padding=5, bias=False)
        self.spatial_conv = nn.Conv1d(32, 32, kernel_size=1, groups=32, bias=False)  # Depthwise Spatial Filtering
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dropout1 = nn.Dropout(dropoutRate)

        # Second Convolution Block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dropout2 = nn.Dropout(dropoutRate)

        # Third Convolution Block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dropout3 = nn.Dropout(dropoutRate)

        # Fourth Convolution Block
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dropout4 = nn.Dropout(dropoutRate)

        # Fully Connected Layer (Dynamically Set in Forward)
        self.fc = None  # Placeholder, will initialize dynamically in forward()

        # Move Model to Correct Device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # Block 1
        x = self.conv1(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flattening
        x = torch.flatten(x, start_dim=1)

        # Ensure Fully Connected Layer is Initialized Correctly
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], 5).to(self.device)  # 5 classes for SEED-V

        # Final Classification
        x = self.fc(x)
        return x
