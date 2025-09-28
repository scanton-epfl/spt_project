from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

MAX_TOKENS = 128

class VideoDataset(Dataset):
    """
    Dataset object for video data
    """
    def __init__(self, videos: torch.Tensor, labels: torch.Tensor):
        self.videos = videos
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.videos[idx], self.labels[idx]

class VideoMotionDataset(VideoDataset):
    """
    Dataset object for video and relative motion between frames data
    """
    def __init__(self, videos: torch.Tensor, motion: torch.Tensor, labels: torch.Tensor):
        super().__init__(videos, labels)
        self.motion = motion
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.videos[idx], self.motion[idx], self.labels[idx]
    
class MultiheadAttention(nn.Module):
    """
    Object for representing multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(self.embed_dim, self.head_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.head_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through self-attention mechanism
        """
        batch_size = x.shape[0]

        # Linear projections: shape (batch_size, num_heads, num_tokens, head_dim)
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.head_dim)

        attn_weights = F.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Concat heads together
        context = context.transpose(1,2).continguous().view(batch_size, -1, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        return output   

class FeedForward(nn.Module):
    """
    Object for representing a feed forward network
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through MLP
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """
    Object for representing a transformer block
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        
        # Attention mechanism
        self.self_attn = MultiheadAttention(embed_dim, num_heads)

        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through a transformer block

        Args:
            x: torch.Tensor (batch_size, num_tokens, embed_dim)
                Batch of sequences for input to the transformer
        """
        # Attention and skip connection 
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward and skip connection
        ff_output = self.mlp(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
    
class Transformer(nn.Module):
    """
    Object for representing a transformer encoder
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_encoding = use_pos_encoding

        # Learning positional encoding (optional)
        if self.use_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, MAX_TOKENS, embed_dim))
        
        # Transformer blocks
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final norm layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Pass through complete transformer encoder
        """
        if self.use_pos_encoding:
            x = x + self.pos_embedding[:, x.shape[1], :]

        for layer in self.encoder_layers:
            x = layer(x)

        return self.norm(x)
