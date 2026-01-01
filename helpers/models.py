from types import LambdaType
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math

MAX_TOKENS = 512

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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, use_rotary=False):
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

        # Rotary embeddings
        self.use_rotary = use_rotary
        if use_rotary:
            self.rope = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through self-attention mechanism
        """
        batch_size = x.shape[0]

        # Linear projections: shape (batch_size, num_heads, num_tokens, head_dim)
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # Apply RoPE embeddings if requested
        if self.use_rotary:
            cos, sin = self.rope(q)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Replace below with torch implementation
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=False)

        # Concat heads together
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        return output   

class CrossAttention(nn.Module):
    """
    Object for representing cross attention
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, use_rotary=False):
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

        # Rotary embeddings
        self.use_rotary = use_rotary
        if use_rotary:
            self.rope = Rotary(self.head_dim)

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

        # Apply RoPE embeddings if requested
        if self.use_rotary:
            cos, sin = self.rope(q_a) # can use this for both since sequence lengths are equal
            q_a, k_a = apply_rotary_pos_emb(q_a, k_a, cos, sin)
            q_b, k_b = apply_rotary_pos_emb(q_b, k_b, cos, sin)

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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float=0.0, use_rotary: bool=False):
        super().__init__()
        
        # Attention mechanism
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, use_rotary)

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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float=0.0, use_rotary: bool=False):
        super().__init__()
        
        # Attention mechanism
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout, use_rotary)

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

        # # Pre-norm cross-attention
        # a_attn, b_attn = self.cross_attn(
        #     self.norm1_a(tokens_a),
        #     self.norm1_b(tokens_b)
        # )

        # tokens_a = tokens_a + self.dropout(a_attn)
        # tokens_b = tokens_b + self.dropout(b_attn)

        # # Pre-norm feedforward
        # tokens_a = tokens_a + self.dropout(self.mlp_a(self.norm2_a(tokens_a)))
        # tokens_b = tokens_b + self.dropout(self.mlp_b(self.norm2_b(tokens_b)))


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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False, use_rotary: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        if use_pos_encoding and use_rotary:
            print('Only apply Sinusoidal or RoPE embeddings, not both!')
            use_pos_encoding = False
        self.use_pos_embed = use_pos_encoding

        # Positional embeddings
        if self.use_pos_embed:
            #self.pos_embedding = nn.Parameter(torch.rand(1, MAX_TOKENS, embed_dim))
            self.add_pos_embed = SinusoidalPositionEmbedding(embed_dim)
        
        # Transformer blocks
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout, use_rotary)
            for _ in range(num_layers)
        ])

        # Final norm layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Pass through complete transformer encoder
        """
        if self.use_pos_embed:
            x = self.add_pos_embed(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return self.norm(x)

class CrossTransformer(nn.Module):
    """
    Object representing a cross attention based transformer
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, use_pos_encoding: bool=False, use_segment_embed: bool=False, use_rotary: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        if use_pos_encoding and use_rotary:
            print('Only apply Sinusoidal or RoPE embeddings, not both! Defaulting to RoPE')
            use_pos_encoding = False
        self.use_pos_embed = use_pos_encoding
        self.use_seg_embed = use_segment_embed

        # Positional embeddings
        if self.use_pos_embed:
            self.add_pos_embed = SinusoidalPositionEmbedding(embed_dim)
            
        # Segment embeddings: 0 = tokens_a, 1 = tokens_b
        if self.use_seg_embed:
            self.segment_embedding = nn.Embedding(2,embed_dim)

        # Transformer blocks
        self.encoder_layers = nn.ModuleList([
            CrossTransformerBlock(embed_dim, num_heads, hidden_dim, dropout, use_rotary)
            for _ in range(num_layers)
        ])

        # Final norm layer
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
    
    def get_embeddings(self, x: torch.Tensor, segment_id: int=0):
        """
        Apply positional and segment embeddings
        """
        batch_size, seq_len, _ = x.shape

        if self.use_pos_embed:
            x = self.add_pos_embed(x)

        if self.use_seg_embed:
            seg_ids = torch.full((batch_size, seq_len), segment_id, dtype=torch.long, device=x.device)
            x = x + self.segment_embedding(seg_ids)

        return x

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross attention transformer
        """
        tokens_a = self.get_embeddings(tokens_a, segment_id=0)
        tokens_b = self.get_embeddings(tokens_b, segment_id=1)

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
                 dropout: float, use_pos_encoding: bool=False, use_sum: bool=False, use_rotary: bool=False):
        super().__init__(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding)
        
        # Method for combining
        self.use_sum = use_sum
        
        # Encoder for displacement data
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Class token used for regression
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding, use_rotary)

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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int=4, use_pos_embed: bool=False, use_segment_embed: bool=False, use_rotary: bool=False):
        super().__init__()
        # Instantiate embedding model
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Class token used for regression
        self.cls1 = nn.Parameter(torch.randn(1,1,embed_dim))
        self.cls2 = nn.Parameter(torch.randn(1,1,embed_dim))

        # Instantiate transformer encoder
        self.transformer = CrossTransformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_embed, use_segment_embed, use_rotary)

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
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int=4, use_pos_embed: bool=False, use_segment_embed: bool=False, use_rotary: bool=False):
        super().__init__(embed_dim, num_heads, hidden_dim, num_layers, dropout, output_dim, use_pos_embed, use_segment_embed, use_rotary)

        # Instantiate transformer to handle output of CrossTransformer
        self.t2 = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_rotary=use_rotary)
    
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

class MultiStateModel(nn.Module):
    """
    Object representing a cross transformer based model for multi-state diffusion prediction
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int=4, use_pos_embed: bool=False, use_segment_embed: bool=False, use_rotary: bool=False):
        super().__init__()
        # Instantiate embedding model
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.disp_encoder = MLP(input_dim=2, hidden_dim=hidden_dim, num_layers=2, output_dim=embed_dim)

        # Instantiate cross transformer encoder
        self.cross_transformer = CrossTransformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_embed, use_segment_embed, use_rotary)
        
        # Instantiate transformer to handle output of cross transformer
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout, use_rotary=use_rotary)

        # Instantiate regression head
        self.mlp = RegressionHead(embed_dim*2, hidden_dim, dropout=dropout, output_dim=output_dim)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model
        """
        # Encode images and displacements
        encoded_images = self.image_encoder(tokens_a)
        encoded_images = self.norm(encoded_images)
        encoded_disp = self.disp_encoder(tokens_b)

        # Pass context to transformer
        image_out, disp_out = self.cross_transformer(encoded_images, encoded_disp)

        # Combine outputs to get input to final transformer
        tokens = torch.cat((image_out, disp_out), dim=1)

        # Pass through final transformer
        output = self.transformer(tokens)

        # Get prediction per token
        _, seq_len, _ = image_out.shape
        image_out = output[:, :seq_len, :]
        disp_out = output[:, seq_len:, :]
        mlp_input = torch.cat((image_out, disp_out), dim=-1)

        predictions = self.mlp(mlp_input)

        return predictions

# -----------------------------------------------------------------------------------------
# Baseline Models

class LSTM(nn.Module):
    """
    Object representing a LSTM model for comparison to transformer implementation
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int, output_dim: int=4, bidirectional: bool=False, dropout: float=0, pointwise: bool=False):
        super().__init__()
        self.pointwise = pointwise
        self.image_encoder = DeepResNetEmbedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = RegressionHead(hidden_dim, hidden_dim, dropout, output_dim)

    def forward(self, x):
        """
        Forward pass through LSTM-based model
        """
        # Extract CNN features for each frame
        frame_features = self.image_encoder(x)
        frame_features = self.norm(frame_features)

        # Process temporal sequence
        lstm_out, _ = self.lstm(frame_features)
        if not self.pointwise:
            if self.lstm.bidirectional:
                forward_last = lstm_out[:, -1, :self.lstm.hidden_size]   # forward final timestep
                backward_last = lstm_out[:, 0, self.lstm.hidden_size:]   # backward first timestep
                mlp_input = torch.cat([forward_last, backward_last], dim=-1)  # shape (B, 2*hidden_size)
            else:
                mlp_input = lstm_out[:, -1, :]
        else:
            mlp_input = lstm_out

        # Regression output
        out = self.fc(mlp_input)

        return out

class Pix2D(nn.Module):
    """
    Object representing the model used in the Pix2D paper https://github.com/ha-park/Pix2D-NN-diffusivity-mapping
    """
    def __init__(self, n_channel=16, output_dim=4, pointwise=False):
        super().__init__()

        # Whether the model should make predictions per-frame or not
        self.pointwise = pointwise
        
        self.model = nn.Sequential(
            # First conv block
            nn.Conv2d(1, n_channel*2, kernel_size=3, padding=1),  # 'same' padding
            nn.BatchNorm2d(n_channel*2),
            nn.SiLU(),
            
            # Second conv block with stride=2 (downsampling)
            nn.Conv2d(n_channel*2, n_channel*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(n_channel*2),
            nn.SiLU(),
            
            # Third conv block
            nn.Conv2d(n_channel*2, n_channel*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channel*4),
            nn.SiLU(),
            
            # Fourth conv block with stride=2
            nn.Conv2d(n_channel*4, n_channel*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(n_channel*4),
            nn.SiLU(),
            
            # Fifth conv block
            nn.Conv2d(n_channel*4, n_channel*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channel*8),
            nn.SiLU(),
            
            # Sixth conv block
            nn.Conv2d(n_channel*8, n_channel*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channel*8),
            nn.SiLU(),
            
            # Dropout
            nn.Dropout(0.2),
            
            # Global Average Pooling to flatten
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # Fully connected regression layer
        self.fc = nn.Linear(n_channel*8, output_dim)
    
    def forward(self, x):
        batch_size, num_images, h, w = x.shape
        x = x.reshape(batch_size * num_images, 1, h, w)  # Flatten num_images into batch

        x = self.model(x)

        x = x.view(batch_size, num_images, -1)
        if not self.pointwise:
            x = x.mean(dim=1)

        return self.fc(x)

# -----------------------------------------------------------------------------------------
# Loss function helpers

def construct_matrix(p1: torch.Tensor, p2: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Construct matrix from eigenvalues and angles

    Args:
        p1: torch.Tensor (batch_size,) or (batch_size, nFrames)
            Eigenvalues in the first prinicipal component
        p2: torch.Tensor (batch_size,) or (batch_size, nFrames)
            Eigenvalues in the second principal component
        theta: torch.Tensor (batch_size,) or (batch_size, nFrames)
            Angle with respect to the x-axis of the eigenvector associated with the first prinicipal component

    Returns:
        m: torch.Tensor (batch_size, 2, 2) or (batch_size, nFrames, 2, 2)
            Matrix for each instance in a batch
    """

    # Define rotation matrices
    c = torch.cos(theta)
    s = torch.sin(theta)

    R_pred = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=-2)


    # Define diagonals
    eig = torch.stack([p1, p2], dim=-1)
    
    # Compute matrix logarithms of predictions and labels
    m = R_pred @ torch.diag_embed(eig) @ R_pred.transpose(-1,-2)

    return m

def log_euclidean_distance(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean distance

    Args:
        predictions: torch.Tensor (batch_size, 4) or (batch_size, nFrames, 4)
        labels: torch.Tensor (batch_size, 3) or (batch_size, nFrames, 3)
    
    Returns:
        d: torch.Tensor (batch_size,) or (batch_size, nFrames)
            Distance computed via the log-euclidean distance metric

    Diffusion tensor is a symmetric positive definite matrix so we can easily decompose into eigenvalues and eigenvectors using:
        
    D = R @ diag(eigenvalues) @ R^T where R is the rotation matrix

    Matrix logarithm is built by decomposing a matrix and taking the natural logarithm of the eigenvalues then reconstructing 
    """
    # Decompose to each parameter
    p1_pred, p2_pred, sin_2theta_pred, cos_2theta_pred = predictions[...,0], predictions[...,1], predictions[...,2], predictions[...,3]
    p1_label, p2_label, theta_label = labels[...,0], labels[...,1], labels[...,2]

    # Compute matrix log for predictions and labels
    theta_pred = 0.5 * torch.atan2(sin_2theta_pred, cos_2theta_pred)

    mlog_pred = construct_matrix(torch.log(p1_pred), torch.log(p2_pred), theta_pred) # handle whether or not to apply log on predictions based on model output
    mlog_label = construct_matrix(torch.log(p1_label), torch.log(p2_label), theta_label)

    # Compute distance w/ squared Frobenious norm
    d = torch.linalg.norm(mlog_pred - mlog_label, dim=(-2,-1))
    
    return d

def log_euclidean_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Log-Euclidean loss function

    Args:
        predictions: torch.Tensor (batch_size, 4) or (batch_size, nFrames, 4)
        labels: torch.Tensor (batch_size, 3) or (batch_size, nFrames, 3)
    
    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the log-euclidean distance metric

    Diffusion tensor is a symmetric positive definite matrix so we can easily decompose into eigenvalues and eigenvectors using:
        
    D = R @ diag(eigenvalues) @ R^T where R is the rotation matrix

    Matrix logarithm is built by decomposing a matrix and taking the natural logarithm of the eigenvalues then reconstructing 
    """
    # Compute distance between predictions
    d = log_euclidean_distance(predictions, labels)

    # Compute loss across batch
    loss = torch.mean(d)
    
    return loss

def mse_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    MSE loss function
    Args:
        predictions: torch.Tensor (batch_size, 4) or (batch_size, nFrames, 4)
        labels: torch.Tensor (batch_size, 3) or (batch_size, nFrames, 3)

    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the MSE metric
    """
    # Decompose to each parameter
    p1_pred = predictions[..., 0]
    p2_pred = predictions[..., 1]
    sin_2theta_pred = predictions[..., 2]
    cos_2theta_pred = predictions[..., 3]

    # Normalize predicted sin/cos
    norm = torch.sqrt(sin_2theta_pred**2 + cos_2theta_pred**2 + 1e-8)
    sin_2theta_pred = sin_2theta_pred / norm
    cos_2theta_pred = cos_2theta_pred / norm
    
    # Compute true values
    theta = labels[..., -1]
    sin_2theta_true = torch.sin(2 * theta)
    cos_2theta_true = torch.cos(2 * theta)

    # MSE on diffusion and angular parts
    mse_diffusion = (p1_pred - labels[..., 0])**2 + (p2_pred - labels[..., 1])**2
    mse_angle = (sin_2theta_pred - sin_2theta_true)**2 + (cos_2theta_pred - cos_2theta_true)**2

    return (mse_diffusion + mse_angle).mean()

def mse_loss_coeff(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    MSE loss function for purely diffusion coefficient predictions
    Args:
        predictions: torch.Tensor (batch_size, 4) or (batch_size, nFrames, 4)
        labels: torch.Tensor (batch_size, 3) or (batch_size, nFrames, 3)
    Returns:
        loss: torch.Tensor (1,)
            Loss computed via the MSE metric
    """
    return torch.mean((predictions - labels[..., :-1])**2)

def mse_similarity(predictions: torch.Tensor, labels: torch.Tensor):
    """
    Compute similarity of two diffusion tensors based on Euclidean distance and RBF kernel
    """
    # Compute diffusion tensors
    theta_pred = 0.5 * torch.atan2(predictions[..., -2], predictions[..., -1])
    m1 = construct_matrix(predictions[..., 0], predictions[..., 1], theta_pred)
    m2 = construct_matrix(labels[..., 0], labels[..., 1], labels[..., -1])

    # Compute Frobenius norm
    d = torch.linalg.norm(m1 - m2, dim=(-2,-1))

    # Convert distance to a similarity
    gamma = 1
    similarity = torch.mean(torch.exp(-gamma * d ** 2))

    return similarity 

def get_changepoints(x: np.ndarray):
    """
    In a two-state scenario, find where the changepoint is located. Goal is to minimize the variance of each side according to the split index
    
    Args:
        x: (N, nFrames, 2)
            Array containing the pointwise diffusion coefficient
    Returns:
        changepoints: (N,2)
            Changepoints per data entry and diffusion coeffcient 
    """
    N, nFrames, _ = x.shape
    
    cost = np.zeros((N, nFrames-1))
    for j in range(1,nFrames):
        # Compute cost / variance of each side of split point
        mean1 = np.mean(x[:, :j], axis=1, keepdims=True)
        mean2 = np.mean(x[:, j:], axis=1, keepdims=True)
        cost[:, j-1] = np.sum((x[:, :j] - mean1)**2, axis=(1,2)) + np.sum((x[:, j:] - mean2)**2, axis=(1,2)) # sum cost over both predictions D_1 and D_2
    
    return np.argmin(cost, axis=1)

def compute_changepoint_error(pred: np.ndarray, labels: np.ndarray):
    """
    Compute the error between predictions and labels changepoint values
    
    Args:
        pred: (N, nFrames, 4)
            Multistate model predictions
        labels: (N, nFrames, 3)
            Multistate model labels
    
    Returns:
        loss:
            Average error between prediction and label
    """
    changepoints_pred = get_changepoints(pred[..., :2])
    changepoints_labels = get_changepoints(labels[...,:-1])
    
    return np.abs(changepoints_pred - changepoints_labels).mean()

# -----------------------------------------------------------------------------------------
# Positional embeddings helpers

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=MAX_TOKENS):
        super().__init__()
        # Create a (max_len, embed_dim) matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape [1, max_len, embed_dim] for broadcasting

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class Rotary(torch.nn.Module):
    """
    Object for representing a cached rotary matrix for computing RoPE embeddings
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=2):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq) # shape: [seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :] # shape: [1, 1, seq_len, dim]
            self.sin_cached = emb.sin()[None, None, :, :]

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    """
    Rotates half the hidden dims of the input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply RoPE embedding
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed