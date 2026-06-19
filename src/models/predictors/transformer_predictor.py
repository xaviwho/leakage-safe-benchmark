"""
Transformer-based predictor for cell state trajectories.

Uses Transformer encoder architecture for capturing long-range
dependencies in cell differentiation dynamics.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional
import logging

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds position information to input embeddings using sine/cosine functions.
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPredictor(BasePredictor):
    """
    Transformer-based time-series predictor.

    Architecture:
    - Input projection layer
    - Positional encoding
    - Transformer encoder layers
    - Output projection layer
    """

    def __init__(
        self,
        input_size: int = 3,  # P, D, N
        output_size: int = 3,
        config: Optional[Dict] = None
    ):
        """
        Initialize Transformer predictor.

        Args:
            input_size: Dimension of input features (default: 3 for P, D, N)
            output_size: Dimension of output (default: 3)
            config: Configuration dictionary with Transformer parameters
        """
        super().__init__(input_size, output_size, config)

        # Get hyperparameters from config
        transformer_config = self.config.get('ml_models', {}).get('transformer', {})
        self.d_model = transformer_config.get('d_model', 128)
        self.nhead = transformer_config.get('nhead', 8)
        self.num_layers = transformer_config.get('num_layers', 4)
        self.dim_feedforward = transformer_config.get('dim_feedforward', 512)
        self.dropout = transformer_config.get('dropout', 0.1)

        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward // 2, output_size)
        )

        # Move to device
        self.to(self.device)

        logger.info(f"Transformer Predictor initialized with {self.count_parameters():,} parameters")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  nhead: {self.nhead}")
        logger.info(f"  num_layers: {self.num_layers}")
        logger.info(f"  dim_feedforward: {self.dim_feedforward}")
        logger.info(f"  Device: {self.device}")

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            src_mask: Optional mask for attention

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        # Transformer expects (seq_len, batch, d_model) for pos encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x, mask=src_mask)

        # Output projection
        output = self.output_projection(x)

        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for autoregressive prediction.

        Args:
            sz: Size of mask
            device: Device to create mask on

        Returns:
            Causal attention mask
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class TransformerWithMemory(TransformerPredictor):
    """
    Transformer with explicit memory mechanism for very long sequences.
    """

    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 3,
        config: Optional[Dict] = None,
        memory_size: int = 64
    ):
        """
        Initialize Transformer with memory.

        Args:
            input_size: Dimension of input features
            output_size: Dimension of output
            config: Configuration dictionary
            memory_size: Size of memory bank
        """
        super().__init__(input_size, output_size, config)

        self.memory_size = memory_size

        # Memory bank (learnable)
        self.memory_bank = nn.Parameter(torch.randn(1, memory_size, self.d_model))

        # Cross-attention to memory
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )

        logger.info(f"Memory bank added with size {memory_size}")

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with memory.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            src_mask: Optional mask for attention

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        batch_size = x.size(0)

        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        # Expand memory for batch
        memory = self.memory_bank.expand(batch_size, -1, -1)

        # Transformer encoding
        x = self.transformer_encoder(x, mask=src_mask)

        # Attend to memory
        x_with_memory, _ = self.memory_attention(x, memory, memory)

        # Combine with residual
        x = x + x_with_memory

        # Output projection
        output = self.output_projection(x)

        return output
