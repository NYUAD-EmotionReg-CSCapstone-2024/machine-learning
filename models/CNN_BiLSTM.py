import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, samples=2400,
                 dropoutRate=0.5, kernLength=14, F1=16, num_lstm=128,
                 D=4, F2=32, device=None):
        super(CNN_BiLSTM, self).__init__()

        # Detect and set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === First Convolution Block (Temporal Feature Extraction) ===
        self.conv1 = nn.Conv1d(n_channels, F1, kernLength, padding=kernLength // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(F1)

        # === Depthwise Convolution (Spatial Feature Extraction) ===
        self.depthwise_conv = nn.Conv1d(F1, F1 * D, 1, groups=F1, bias=False)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool1d(kernel_size=5)
        self.dropout1 = nn.Dropout(dropoutRate)

        # === Separable Convolution ===
        self.sep_conv = nn.Conv1d(F1 * D, F2, kernel_size=16, padding=16 // 2, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)
        self.activation2 = nn.ELU()

        # === BiLSTM Layer ===
        self.bilstm = nn.GRU(input_size=F2, hidden_size=num_lstm, batch_first=True, bidirectional=True)

        # Fully Connected Layer
        self.fc = nn.Linear(num_lstm * 2, nb_classes)

        # Move Model to Correct Device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # === Convolution Blocks ===
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.activation2(x)

        # === Transpose for BiLSTM (Batch, Features, Time) → (Batch, Time, Features) ===
        x = x.permute(0, 2, 1)

        # === BiLSTM ===
        x, _ = self.bilstm(x)  # (batch, time, 2*num_lstm)
        x = torch.mean(x, dim=1)  # Global Average Pooling

        # === Fully Connected Layer ===
        x = self.fc(x)  # (batch, nb_classes)
        return x
