from os import sysconf
from types import LambdaType
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math

MAX_TOKENS = 128

# -----------------------------------------------------------------------------------------
# Dataset objects for PyTorch implementations

class VideoDataset(Dataset):
    """
    Dataset object for video data
    """
    def __init__(self, videos: torch.Tensor, labels: torch.Tensor):
        self.videos = videos
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.videos[idx], torch.Tensor(), self.labels[idx]

class VideoMotionDataset(VideoDataset):
    """
    Dataset object for video and relative displacement between frames data
    """
    def __init__(self, videos: torch.Tensor, displacements: torch.Tensor, labels: torch.Tensor):
        super().__init__(videos, labels)
        # Add zero displacement to front of each sequence in batch to match dimension with image sequences
        batch_size, _, _ = displacements.shape
        zeros = torch.zeros(batch_size, 1, 2)
        self.displacements = torch.cat((zeros, displacements), dim=1) # (batch_size, nFrames, 2)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.videos[idx], self.displacements[idx], self.labels[idx]

class DisplacementDataset(Dataset):
    """
    Dataset object for displacement data
    """
    def __init__(self, displacements: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        batch_size, _, _ = displacements.shape
        zeros = torch.zeros(batch_size, 1, 2)
        self.displacements = torch.cat((zeros,displacements), dim=1)
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.displacements[idx], torch.Tensor(), self.labels[idx]

# -----------------------------------------------------------------------------------------
# Attention mechanisms
 
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

        # Replace below with torch implementation
        # context = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=False)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Concat heads together
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        return output   

class CrossAttention(nn.Module):
    """
    Object for representing cross attention
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for both modalities
        self.q_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj_a = nn.Linear(self.embed_dim, self.embed_dim)

        self.q_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj_b = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = dropout

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross attention module
        """
        batch_size = tokens_a.shape[0]

        # Linear projections: shape (batch_size, num_heads, num_tokens, head_dim)
        q_a = self.q_proj_a(tokens_a).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k_a = self.k_proj_a(tokens_a).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v_a = self.v_proj_a(tokens_a).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        q_b = self.q_proj_b(tokens_b).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k_b = self.k_proj_b(tokens_b).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v_b = self.v_proj_b(tokens_b).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # Get heads from each attention mechanism
        context_a = F.scaled_dot_product_attention(q_b, k_a, v_a, dropout_p=self.dropout) # (batch_size, num_heads, seq_len, head_dim)
        context_b = F.scaled_dot_product_attention(q_a, k_b, v_b, dropout_p=self.dropout)
        
        # Concatenate heads
        context_a = context_a.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        context_b = context_b.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final projection
        output_a = self.out_proj_a(context_a)
        output_b = self.out_proj_b(context_b)

        return output_a, output_b
    
# -----------------------------------------------------------------------------------------
# Model building blocks

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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float=0.0):
        super().__init__()
        
        # Attention mechanism
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)

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
    
class CrossTransformerBlock(nn.Module):
    """
    Object representing a transformer block in a cross attention based model
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float=0.0):
        super().__init__()
        
        # Attention mechanism
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)

        # Normalization layers
        self.norm1_a = nn.LayerNorm(embed_dim)
        self.norm2_a = nn.LayerNorm(embed_dim)
        self.norm1_b = nn.LayerNorm(embed_dim)
        self.norm2_b = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.mlp_a = FeedForward(embed_dim, hidden_dim, dropout)
        self.mlp_b = FeedForward(embed_dim, hidden_dim, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through a transformer block with cross attention
        """
        # Attention
        output_a, output_b = self.cross_attn(tokens_a, tokens_b)
        
        # Skip connection and norm
        tokens_a = self.apply_subblock(tokens_a, lambda x: output_a, self.norm1_a)
        tokens_b = self.apply_subblock(tokens_b, lambda x: output_b, self.norm1_b)

        # Feed-forward and skip connection
        tokens_a = self.apply_subblock(tokens_a, self.mlp_a, self.norm2_a)
        tokens_b = self.apply_subblock(tokens_b, self.mlp_b, self.norm2_b)

        return tokens_a, tokens_b
    
    def apply_subblock(self, x: torch.Tensor, sublayer: LambdaType, norm: LambdaType):
        """
        Helper function to apply a sublayer within a transformer block
        """
        x = x + self.dropout(sublayer(x))
        
        return norm(x)
    
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

class CrossTransformer(nn.Module):
    """
    Object representing a cross attention based transformer
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer blocks
        self.encoder_layers = nn.ModuleList([
            CrossTransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final norm layer
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross attention transformer
        """
        for layer in self.encoder_layers:   
            tokens_a, tokens_b = layer(tokens_a, tokens_b)
        
        return self.norm_a(tokens_a), self.norm_b(tokens_b)
    
class RegressionHead(nn.Module):
    """
    Object representing the MLP head for prediction of the diffusion parameters
    """
    def __init__(self, input_dim: int, hidden_dim: int=128, dropout: float=0.0, output_dim: int=4):
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
    
class MLP(nn.Module):
    """
    General MLP
    """
    def __init__(self, input_dim: int=2, hidden_dim: int=128, num_layers: int=2, output_dim: int=32, dropout: float=0.0):
        super().__init__()
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

# -----------------------------------------------------------------------------------------
# Model implementations

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

class DiffusionTensorRegModel(DiffusionTensorRegModelBase):
    """
    Object representing the model for predicting the diffusion tensor w/ displacement data as a second mode
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, 
                 dropout: float, use_pos_encoding: bool=False, use_sum: bool=False):
        super().__init__(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)
        
        # Method for combining
        self.use_sum = use_sum
        
        # Encoder for displacement data
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Class token used for regression
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        """
        Forward pass through summation/concatentation multimodal model
        """
        # Get encoded videos
        encoded_images = self.image_encoder(x) # (batch_size, num_frames, embed_dim)

        # Normalize for input to the model
        encoded_images = self.norm(encoded_images)

        # Get encoded displacement vectors
        encoded_disp = self.disp_encoder(disp) # (batch_size, num_frames, output_dim)
        
        # Combine image and displacement data
        if self.use_sum:
            encoded = encoded_images + encoded_disp
        else:
            encoded = torch.cat((encoded_images, encoded_disp), dim=1)

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

class DisplacementBasedModel(nn.Module):
    """
    Object representing a model that only uses encoded displacement features
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False):
        super().__init__()

        # Encoder for displacement data
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)
        
        # Class token used for regression
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the base model
        """

        # Encode displacement data
        encoded = self.disp_encoder(x)

        # Get input for transformer by combining cls token with displacement sequences
        batch_size, _, _ = encoded.shape
        cls_token = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, encoded), dim=1) # (batch_size, num_frames + 1, embed_dim)

        # Pass through transformer
        output = self.transformer(tokens)

        # Pass cls representations to regression head
        cls_tokens = output[:, 0, :]
        predictions = self.mlp(cls_tokens)

        return predictions

class HierarchicalModel(nn.Module):
    """
    Object representing a hierarchical model (one to multi stream)
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False):
        super().__init__()
        # Instantiate image embedding model
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Instantiate displacement encoder
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Class token used for regression
        self.cls1 = nn.Parameter(torch.randn(1,1,embed_dim))
        self.cls2 = nn.Parameter(torch.randn(1,1,embed_dim))
        self.cls3 = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoders
        self.t1 = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)
        self.t2 = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)
        self.t3 = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim, hidden_dim, dropout=dropout)

        # Parameter for learning weights to combine cls tokens
        self.alpha = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical model
        """
        # Encode images
        encoded_images = self.image_encoder(x)
        encoded_images = self.norm(encoded_images)

        # Encode displacements
        encoded_disp = self.disp_encoder(disp)

        # Expand cls token to batch
        batch_size, seq_len, _ = encoded_images.shape
        cls1_token = self.cls1.expand(batch_size, -1, -1)

        # Concatenate for input to first transformer
        tokens = torch.cat((cls1_token, encoded_images, encoded_disp), dim=1)

        # Pass through first transformer
        t1_output = self.t1(tokens)

        # Get inputs for second and third transformers
        cls2_token = self.cls2.expand(batch_size, -1, -1)
        cls3_token = self.cls3.expand(batch_size, -1, -1)
        image_context = torch.cat((cls2_token, t1_output[:, 1:seq_len+1, :]), dim=1)
        disp_context = torch.cat((cls3_token, t1_output[:, seq_len+1:, :]), dim=1)

        # Pass through two transformers
        t2_output = self.t2(image_context)
        t3_output = self.t3(disp_context)

        # Aggregrate cls tokens for regression
        weights = torch.softmax(self.alpha, dim=0)
        output = (
            weights[0] * t1_output[:,0,:]
            + weights[1] * t2_output[:,0,:]
            + weights[2] * t3_output[:,0,:]
        )

        # Make predictions
        predictions = self.mlp(output)

        return predictions

class CrossAttentionModelBase(nn.Module):
    """
    Object representing a model based on cross attention between two modalities
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int=4):
        super().__init__()
        # Instantiate embedding model
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Class token used for regression
        self.cls1 = nn.Parameter(torch.randn(1,1,embed_dim))
        self.cls2 = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = CrossTransformer(embed_dim, num_heads, hidden_dim, num_layers, dropout)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim, hidden_dim, dropout=dropout, output_dim=output_dim)

        # Parameter for learning weight to combine cls tokens
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def get_cross_transformer_out(self, x: torch.Tensor, disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function for abstraction
        """
        # Encode displacements and images
        encoded_images = self.image_encoder(x)
        encoded_images = self.norm(encoded_images)
        encoded_disp = self.disp_encoder(disp)
        
        # Add cls token to front of each sequence
        batch_size, _, _ = encoded_images.shape
        cls1_token = self.cls1.expand(batch_size, -1, -1)
        cls2_token = self.cls2.expand(batch_size, -1, -1)

        image_context = torch.cat((cls1_token, encoded_images), dim=1)
        disp_context = torch.cat((cls2_token, encoded_disp), dim=1)

        # Pass context to transformer
        image_out, disp_out = self.transformer(image_context, disp_context)

        return image_out, disp_out
    
    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        """
        Forward pass through cross attention model
        """
        # Get output sequences from cross transformer
        image_out, disp_out = self.get_cross_transformer_out(x, disp)

        # Aggregrate cls tokens from each
        output = torch.sigmoid(self.alpha)*image_out[:,0,:] + (1-torch.sigmoid(self.alpha))*disp_out[:,0,:]

        # Make predictions
        predictions = self.mlp(output)

        return predictions
    
class CrossAttentionModel(CrossAttentionModelBase):
    """
    Object representing a model that uses cross attention and self attention across two layers
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int=4):
        super().__init__(embed_dim, num_heads, hidden_dim, num_layers, dropout, output_dim)

        # Instantiate transformer to handle output of CrossTransformer
        self.t2 = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout)
    
    def forward(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross transformer and a transformer on top of it
        """
        # Cross transformer output
        image_out, disp_out = self.get_cross_transformer_out(x, disp)

        # Aggregrate cls token
        cls_token = torch.sigmoid(self.alpha)*image_out[:,0,:] + (1-torch.sigmoid(self.alpha))*disp_out[:,0,:]

        # Concatenate sequence for input
        tokens = torch.cat((cls_token.unsqueeze(1), image_out[:,1:,:], disp_out[:,1:,:]), dim=1)

        # Pass through transformer
        output = self.t2(tokens)

        # Make predictions
        predictions = self.mlp(output[:,0,:])

        return predictions
    
# -----------------------------------------------------------------------------------------
# Loss function helpers

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


    # Define diagonals
    log_eig = torch.stack([log_p1, log_p2], dim=-1)
    
    # Compute matrix logarithms of predictions and labels
    mlog = R_pred @ torch.diag_embed(log_eig) @ R_pred.transpose(-1,-2)

    return mlog

def log_euclidean_loss(labels: torch.Tensor, predictions: torch.Tensor, angle_reg: float=1) -> torch.Tensor:
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
    p1_pred, p2_pred, sin_2theta_pred, cos_2theta_pred = predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3]
    p1_label, p2_label, theta_label = labels[:,0], labels[:,1], labels[:,2]

    # Compute matrix log for predictions and labels
    theta_pred = 0.5 * torch.atan2(sin_2theta_pred, cos_2theta_pred)

    mlog_pred = construct_matrix_log(p1_pred, p2_pred, theta_pred)
    mlog_label = construct_matrix_log(torch.log(p1_label), torch.log(p2_label), theta_label)

    # Compute loss across batch w/ squared Frobenious norm
    loss = torch.mean(torch.linalg.norm(mlog_pred - mlog_label, dim=(1,2))**2)

    # Angle regularization
    # norm_error = (sin_2theta_pred**2 + cos_2theta_pred**2 - 1)**2
    # loss_angle_reg = torch.mean(norm_error)
    sin_2theta_true = torch.sin(2 * theta_label)
    cos_2theta_true = torch.cos(2 * theta_label)
    loss_angle_reg = ((sin_2theta_pred - sin_2theta_true)**2 + (cos_2theta_pred - cos_2theta_true)**2).mean()
    
    # reconstruct predicted eigenvalues (not log)
    p1_pred2, p2_pred2 = torch.exp(predictions[:,0]), torch.exp(predictions[:,1])

    # mean diffusivity penalty (physical space)
    L_scale = torch.mean((p1_pred2 - p1_label)**2 + (p2_pred2 - p2_label)**2)

    # combined loss
    loss = loss #+ 0.5*loss_angle_reg + 0.1*L_scale
    
    return loss

def mse_loss(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """
    MSE loss function
    Args:
        labels: torch.Tensor (batch_size, 3)
        predictions: torch.Tensor (batch_size, 4)

    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the MSE metric
    """
    # Decompose to each parameter
    p1_pred, p2_pred, sin_2theta_pred, cos_2theta_pred = predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3]

    # # Normalize predicted sin/cos to unit circle
    norm = torch.sqrt(sin_2theta_pred**2 + cos_2theta_pred**2 + 1e-8)
    sin_2theta_pred = sin_2theta_pred / norm
    cos_2theta_pred = cos_2theta_pred / norm
    
    # Compute true values
    theta = labels[:,-1]
    sin_2theta_true = torch.sin(2 * theta)
    cos_2theta_true = torch.cos(2 * theta)

    # MSE on linear and angular parts
    mse_linear = ((p1_pred - labels[:,0])**2 + (p2_pred - labels[:,1])**2).mean()
    mse_angle = ((sin_2theta_pred - sin_2theta_true)**2 + (cos_2theta_pred - cos_2theta_true)**2).mean()

    return mse_linear + mse_angle

def mse_loss_coeff(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """
    MSE loss function for purely diffusion coefficient predictions
    Args:
        labels: torch.Tensor (batch_size, 3)
        predictions: torch.Tensor (batch_size, 2)

    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the MSE metric
    """
    return torch.mean((predictions - labels[:,:-1])**2)