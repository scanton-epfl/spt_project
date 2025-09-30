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
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
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

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Concat heads together
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        return output   

class FeedForward(nn.Module):
    """
    Object for representing a feed forward network in a transformer block
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
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
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

class RegressionHead(nn.Module):
    """
    Object representing the MLP head for prediction of the diffusion parameters
    """
    def __init__(self, input_dim: int, hidden_dim: int=128, dropout: float=0.0, output_dim: int=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through prediction head on transformer output
        """
        return self.mlp(x)
    
class ResidualBlock(nn.Module):
    """
    Object representing a residual block in a ResNet
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        # Set stride if reducing dimensionality
        stride = 2 if downsample else 1
        
        # First layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Handle mismatched dimension in residual connection
        self.skip = nn.Sequential()
        if out_channels != in_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )
            
    def forward(self, x):
        """
        Pass through residual block
        """
        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)

        return self.relu(out + self.skip(x))

class DeepResNetEmbedding(nn.Module):
    """
    Object representing a ResNet for embedding image data
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to reduce spatial dims
        self.fc = nn.Linear(128, embed_dim)  # Project to embed_dim

    def forward(self, x):
        """
        Forward pass through ResNet embedder
        """
        batch_size, num_images, h, w = x.shape
        x = x.reshape(batch_size * num_images, 1, h, w)  # Flatten num_images into batch

        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.global_pool(x)  # (B * num_images, 128, 1, 1)
        x = x.view(batch_size, num_images, -1)  # Reshape back: (B, num_images, 128)
        
        return self.fc(x)  # Final projection to embed_dim

class DiffusionTensorRegModelBase(nn.Module):
    """
    Object representing the model for predicting the diffusion tensor
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False):
        super().__init__()

        # Instantiate embedding model
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Class token used for regression
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the base model
        """
        # Get encoded videos
        encoded = self.image_encoder(x) # (batch_size, num_frames, embed_dim)

        # Normalize for input to the model
        encoded = self.norm(encoded)

        # Get input for transformer by combining cls token with image sequences
        batch_size, _, _ = encoded.shape
        cls_token = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, encoded), dim=1) # (batch_size, num_frames + 1, embed_dim)

        # Pass through transformer
        output = self.transformer(tokens)

        # Pass cls representations to regression head
        cls_tokens = output[:, 0, :]
        predictions = self.mlp(cls_tokens)

        return predictions
    
def construct_matrix_log(log_p1: torch.Tensor, log_p2: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Construct matrix log from log(eigenvalues) and angles

    Args:
        log_p1: torch.Tensor (batch_size,)
            Log(eigenvalues) in the first prinicipal component
        log_p2: torch.Tensor (batch_size,)
            Log(eigenvalues) in the second principal component
        theta: torch.Tensor (batch_size,)
            Angle with respect to the x-axis of the eigenvector associated with the first prinicipal component

    Returns:
        mlog: torch.Tensor (batch_size, 2, 2)
            Matrix logarithm for each instance in a batch
    """

    # Define rotation matrices
    c = torch.cos(theta) # (batch_size,)
    s = torch.sin(theta) # (batch_size,)

    R_pred = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=-2) # (batch_size, 2, 2)


    # Define diagonal matrices
    log_eig = torch.stack([log_p1, log_p2], dim=-1)
    
    # Compute matrix logarithms of predictions and labels
    mlog = R_pred @ torch.diag_embed(log_eig) @ R_pred.transpose(-1,-2)

    return mlog

def log_euclidean_loss(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean loss function

    Args:
        labels: torch.Tensor (batch_size, 3)
        predictions: torch.Tensor (batch_size, 3)
    
    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the log-euclidean distance metric

    Diffusion tensor is a symmetric positive definite matrix so we can easily decompose into eigenvalues and eigenvectors using:
        
    D = R @ diag(eigenvalues) @ R^T where R is the rotation matrix

    Matrix logarithm is built by decomposing a matrix and taking the natural logarithm of the eigenvalues then reconstructing 
    """
    # Decompose to each parameter
    p1_pred, p2_pred, theta_pred = predictions[:,0], predictions[:,1], predictions[:,2]
    p1_label, p2_label, theta_label = labels[:,0], labels[:,1], labels[:,2]

    # Compute matrix log for predictions and labels
    mlog_pred = construct_matrix_log(p1_pred, p2_pred, theta_pred)
    mlog_label = construct_matrix_log(p1_label, p2_label, theta_label)

    # Compute loss across batch w/ Frobenious norm
    loss = ((mlog_pred - mlog_label)**2).sum(dim=(1,2)).mean()

    return loss