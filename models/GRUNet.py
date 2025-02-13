import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, samples=2400, dropoutRate=0.3, L1=64, L2=32, device=None):
        super(GRUNet, self).__init__()

        # Detect and set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GRU Layers
        self.gru1 = nn.GRU(input_size=samples, hidden_size=L1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropoutRate)

        self.gru2 = nn.GRU(input_size=L1 * 2, hidden_size=L2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropoutRate)

        # Fully Connected Layer
        self.fc = nn.Linear(L2 * 2, nb_classes)

        # Move Model to Correct Device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # First GRU Layer
        x, _ = self.gru1(x)  # (batch, channels, 2*L1)
        x = self.dropout1(x)

        # Second GRU Layer
        x, _ = self.gru2(x)  # (batch, channels, 2*L2)
        x = self.dropout2(x)

        # Global Average Pooling
        x = torch.mean(x, dim=1)  # (batch, 2*L2)

        # Fully Connected Layer
        x = self.fc(x)  # (batch, nb_classes)

        return x
