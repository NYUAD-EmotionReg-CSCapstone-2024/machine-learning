import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, n_samples=200, n_channels=62, n_classes=5):
        super(BaseModel, self).__init__()
        self.linear1 = nn.Linear(n_samples * n_channels, 8192)
        self.linear2 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.2)
        self.linear3 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.2)
        self.linear4 = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x