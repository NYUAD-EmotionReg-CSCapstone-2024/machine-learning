import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

class ERTNet(nn.Module):
    def __init__(self, nb_classes=5, n_channels=62, n_samples=800, dropoutRate=0.5, kernLength=64, F1=8, heads=8, D=2, F2=16):
        super(ERTNet, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, F1, kernLength, padding=kernLength // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(F1)
        self.depthwise_conv = nn.Conv1d(F1, F1 * D, 1, groups=F1, bias=False)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout1 = nn.Dropout(dropoutRate)
        
        self.sep_conv = nn.Conv1d(F1 * D, F2, 16, padding=16 // 2, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)

        self.pos_encoder = PositionalEncoding(F2)
        self.transformer = TransformerBlock(F2, heads, dropoutRate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(dropoutRate)
        self.dense = nn.Linear(F2, nb_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
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
        
        x = x.transpose(1, 2)  # Change to (batch_size, seq_length, features)
        
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Change to (seq_length, batch_size, features)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Change back to (batch_size, seq_length, features)
        
        x = x.transpose(1, 2)  # Change to (batch_size, features, seq_length)
        x = self.global_avg_pool(x).squeeze(2)
        x = self.dropout2(x)
        x = self.dense(x)
        return x