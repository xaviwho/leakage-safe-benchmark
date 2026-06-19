"""
LSTM-based predictor for cell state trajectories.

Uses stacked LSTM layers to predict future cell states based on
historical trajectories.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """
    LSTM-based time-series predictor.

    Architecture:
    - Stacked LSTM layers with dropout
    - Fully connected output layer
    - Optional batch normalization
    """

    def __init__(
        self,
        input_size: int = 3,  # P, D, N
        output_size: int = 3,
        config: Optional[Dict] = None
    ):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Dimension of input features (default: 3 for P, D, N)
            output_size: Dimension of output (default: 3)
            config: Configuration dictionary with LSTM parameters
        """
        super().__init__(input_size, output_size, config)

        # Get hyperparameters from config
        lstm_config = self.config.get('ml_models', {}).get('lstm', {})
        self.hidden_size = lstm_config.get('hidden_size', 128)
        self.num_layers = lstm_config.get('num_layers', 2)
        self.dropout = lstm_config.get('dropout', 0.2)
        self.bidirectional = lstm_config.get('bidirectional', False)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        # Output projection
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, output_size)
        )

        # Move to device
        self.to(self.device)

        logger.info(f"LSTM Predictor initialized with {self.count_parameters():,} parameters")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Bidirectional: {self.bidirectional}")
        logger.info(f"  Device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Apply output projection to each timestep
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        output = self.fc(lstm_out)

        return output

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden and cell states.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(self.device)
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(self.device)
        return h0, c0


class AttentionLSTMPredictor(LSTMPredictor):
    """
    LSTM with attention mechanism for better long-range dependencies.
    """

    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 3,
        config: Optional[Dict] = None
    ):
        """Initialize Attention LSTM predictor."""
        super().__init__(input_size, output_size, config)

        # Attention mechanism
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )

        # Update output projection to include attention
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, output_size)
        )

        logger.info("Attention mechanism added to LSTM")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Concatenate LSTM and attention outputs
        combined = torch.cat([lstm_out, attn_out], dim=-1)

        # Output projection
        output = self.fc(combined)

        return output
