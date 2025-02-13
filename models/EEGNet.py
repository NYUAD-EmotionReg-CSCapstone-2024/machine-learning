import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, samples=2400,
                 dropoutRate=0.5, kernLength=14, F1=16,
                 D=4, F2=32, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        # Define dropout type
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout(dropoutRate)
        else:
            raise ValueError("dropoutType must be 'SpatialDropout2D' or 'Dropout'.")

        # First Temporal Convolution (1D conv for EEG signals)
        self.conv1 = nn.Conv1d(n_channels, F1, kernLength, padding=kernLength // 2, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(F1)

        # Depthwise Convolution (Separable Convolution)
        self.depthwise_conv = nn.Conv1d(F1, F1 * D, kernel_size=1, groups=F1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(F1 * D)

        # Pooling & Dropout
        self.pool1 = nn.AvgPool1d(kernel_size=4)
        self.dropout1 = self.dropout

        # Second Convolution (Feature Extraction)
        self.conv2 = nn.Conv1d(F1 * D, F2, kernel_size=16, padding=16 // 2, bias=False)
        self.batch_norm3 = nn.BatchNorm1d(F2)

        # Second Pooling & Dropout
        self.pool2 = nn.AvgPool1d(kernel_size=8)
        self.dropout2 = self.dropout

        # Fully Connected Layer (Adjust based on flattened size)
        flattened_size = (samples // 32) * F2  # Adjusted to SEED-V
        self.fc = nn.Linear(flattened_size, nb_classes)

    def forward(self, x):
        # Temporal Convolution
        x = self.conv1(x)  # (batch, F1, 2400)
        x = self.batch_norm1(x)
        x = F.relu(x)

        # Depthwise Convolution
        x = self.depthwise_conv(x)  # (batch, F1 * D, 2400)
        x = self.batch_norm2(x)
        x = F.relu(x)

        # Pooling & Dropout
        x = self.pool1(x)  # (batch, F1 * D, 600)
        x = self.dropout1(x)

        # Feature Extraction
        x = self.conv2(x)  # (batch, F2, 600)
        x = self.batch_norm3(x)
        x = F.relu(x)

        # Second Pooling & Dropout
        x = self.pool2(x)  # (batch, F2, 75)
        x = self.dropout2(x)

        # Flatten for classification
        x = torch.flatten(x, start_dim=1)  # (batch, F2 * 75)
        x = self.fc(x)  # (batch, nb_classes)
        
        return x
