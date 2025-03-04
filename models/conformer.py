import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce

###############################################################################
# 1) PatchEmbedding for EEG
###############################################################################
class PatchEmbedding(nn.Module):
    """
    Adapts input of shape (B, 1, 62, time) -> (B, seq_len, emb_size).
    Using a shallow CNN plus a final projection.
    """
    def __init__(self, emb_size=32, embed_dropout=0.25):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), stride=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(62, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            nn.Dropout(embed_dropout),
        )

        # Project to emb_size, then flatten
        self.projection = nn.Sequential(
            nn.Conv2d(32, emb_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.Flatten(start_dim=2),  # (B, emb_size, T')
            nn.Dropout(embed_dropout),
        )

    def forward(self, x):
        """
        x shape: (B, 1, 62, time)
        Returns (B, T', emb_size)
        """
        x = self.shallownet(x)      # (B, 32, 1, T')
        x = self.projection(x)      # (B, emb_size, T')
        x = x.transpose(1, 2)       # (B, T', emb_size)
        return x

###############################################################################
# 2) Conformer Sub-Modules: FeedForward, ConvModule, MHSA
###############################################################################
class FeedForward(nn.Module):
    """
    A simpler feed-forward block with optional scaling, good for EEG.
    """
    def __init__(self, dim, expansion_factor=2, dropout=0.25, ff_scale=0.5):
        super().__init__()
        self.ff_scale = ff_scale
        inner_dim = int(dim * expansion_factor)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.ReLU(),      # or nn.GELU()
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff_scale * self.net(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Standard MHSA with batch_first=True for EEG tasks.
    """
    def __init__(self, dim, num_heads=4, dropout=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, dim)
        residual = x
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return residual + self.dropout(out)

class ConvolutionModule(nn.Module):
    """
    Conformer-style convolution block:
      1) LayerNorm
      2) Pointwise Conv -> 2*dim -> GLU
      3) Depthwise Conv
      4) BatchNorm
      5) Activation (SiLU)
      6) Pointwise Conv -> dim
      7) Residual + Dropout
    """
    def __init__(self, dim, kernel_size=15, dropout=0.25):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2*dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                        groups=dim, padding="same")
        self.batchnorm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, dim)
        residual = x
        x = self.norm(x)
        x = rearrange(x, 'b t d -> b d t')  # (B, dim, T)

        x = self.pointwise_conv1(x)        # (B, 2*dim, T)
        left, right = x.chunk(2, dim=1)
        x = left * torch.sigmoid(right)    # GLU => (B, dim, T)

        x = self.depthwise_conv(x)         # (B, dim, T)
        x = self.batchnorm(x)
        x = F.silu(x)

        x = self.pointwise_conv2(x)        # (B, dim, T)
        x = rearrange(x, 'b d t -> b t d') # (B, T, dim)
        return residual + self.dropout(x)

###############################################################################
# 3) ConformerBlock
###############################################################################
class ConformerBlock(nn.Module):
    """
    FF -> MHSA -> Conv -> FF -> final LayerNorm
    """
    def __init__(self,
                 dim,
                 num_heads=4,
                 ff_expansion_factor=2,
                 conv_kernel_size=15,
                 dropout=0.25,
                 ff_scale=0.5):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion_factor, dropout, ff_scale)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.ff2 = FeedForward(dim, ff_expansion_factor, dropout, ff_scale)
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.final_ln(x)
        return x

###############################################################################
# 4) Conformer Model
###############################################################################
class Conformer(nn.Module):
    """
    A simpler Conformer for EEG.
    By default:
      - 2 blocks
      - emb_size=32
      - embed_dropout=0.25
      - block_dropout=0.25
    """
    def __init__(self,
                 emb_size=32,
                 depth=2,
                 num_heads=4,
                 ff_expansion_factor=2,
                 conv_kernel_size=15,
                 embed_dropout=0.25,
                 block_dropout=0.25,
                 n_classes=5):
        super().__init__()
        # PatchEmbedding uses embed_dropout
        self.patch_embedding = PatchEmbedding(emb_size, embed_dropout)

        # Conformer blocks use block_dropout
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim=emb_size,
                num_heads=num_heads,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=block_dropout,
                ff_scale=0.5
            )
            for _ in range(depth)
        ])

        # Optionally add more dropout here if needed
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(block_dropout),          # extra dropout in the head
            Reduce('b t d -> b d', 'mean'),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        """
        x: (B, 62, time).
        We'll unsqueeze to (B, 1, 62, time) for 2D conv usage.
        """
        x = x.unsqueeze(1)            # (B, 1, 62, time)
        x = self.patch_embedding(x)   # (B, T', emb_size)
        for block in self.blocks:
            x = block(x)              # (B, T', emb_size)
        x = self.head(x)              # (B, n_classes)
        return x
