import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2400):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8, dim_feedforward=256, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # Change to (seq_len, batch, feature)
        
        # Multi-Head Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))  # Residual Connection 1
        
        # Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))  # Residual Connection 2
        
        x = x.permute(1, 2, 0)  # Back to (batch, feature, seq_len)
        return x


class ERTNet(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, n_samples=2400, dropoutRate=0.5, kernLength=14, F1=16, heads=8, D=4, F2=64):
        super(ERTNet, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, F1, kernLength, padding=kernLength // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(F1)
        self.depthwise_conv = nn.Conv1d(F1, F1 * D, 1, groups=F1, bias=False)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout1 = nn.Dropout(dropoutRate)
        
        self.sep_conv = nn.Conv1d(F1 * D, F2, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)

        self.pos_encoder = PositionalEncoding(F2, n_samples//4)
        self.transformer = TransformerBlock(F2, heads, dim_feedforward=256, dropout=dropoutRate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(dropoutRate)
        self.dense = nn.Linear(F2, nb_classes)

    def forward(self, x):
        # Input x shape: (batch_size, seq_length, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout1(x)
        
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.activation(x)
                
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # (256, 64)

        x = self.dropout2(x)
        x = self.dense(x)
        return x