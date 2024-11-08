import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """
    A simple 1D CNN model for EEG classification.
    Takes raw EEG signals and processes them with 1D convolutions.
    """
    def __init__(self, n_channels, n_classes, n_samples=200):
        super(CNN1D, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Calculate the size of flattened features
        self._to_linear = self._get_conv_output_size(n_channels, n_samples)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.dropout5 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, n_classes)

    def _get_conv_output_size(self, channels, length):
        """
        Helper function to calculate the size of the flattened features
        after convolution and pooling operations
        """
        # Create a dummy input
        dummy_input = torch.zeros(1, channels, length)
        
        # Forward pass through convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Calculate flattened size
        return x.numel()

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Input tensor of shape (batch_size, n_channels, n_samples)
        Returns:
            Class probabilities
        """
        # If input is in (batch_size, n_samples, n_channels), transpose it
        if x.shape[1] != self.conv1.in_channels:
            x = x.transpose(1, 2)
        
        # Convolutional blocks
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout4(F.relu(self.fc1(x)))
        x = self.dropout5(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)