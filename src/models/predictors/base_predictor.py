"""
Base class for ML predictors.

This defines the interface that all predictors must implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BasePredictor(ABC, nn.Module):
    """
    Abstract base class for ML predictors.

    All predictors should inherit from this class and implement
    the required methods.
    """

    def __init__(self, input_size: int, output_size: int, config: Optional[Dict] = None):
        """
        Initialize predictor.

        Args:
            input_size: Dimension of input features
            output_size: Dimension of output predictions
            config: Configuration dictionary
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        pass

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions given historical data.

        Args:
            history: Historical states of shape (seq_len, state_dim)
            horizon: Number of steps to predict ahead
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            predictions: Predicted states of shape (horizon, state_dim)
            uncertainty: Uncertainty estimates (if requested)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(history).unsqueeze(0).to(self.device)

            # Autoregressive prediction
            predictions = []
            current_sequence = x

            for _ in range(horizon):
                # Predict next step
                pred = self.forward(current_sequence)
                next_step = pred[:, -1:, :]  # Take last prediction
                predictions.append(next_step)

                # Update sequence
                current_sequence = torch.cat([current_sequence[:, 1:, :], next_step], dim=1)

            # Concatenate predictions
            predictions = torch.cat(predictions, dim=1)
            predictions = predictions.squeeze(0).cpu().numpy()

            uncertainty = None
            if return_uncertainty:
                # Simple uncertainty: use dropout at test time (MC Dropout)
                uncertainty = self._estimate_uncertainty(history, horizon)

            return predictions, uncertainty

    def _estimate_uncertainty(
        self,
        history: np.ndarray,
        horizon: int,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Estimate prediction uncertainty using MC Dropout.

        Args:
            history: Historical states
            horizon: Prediction horizon
            n_samples: Number of MC samples

        Returns:
            Standard deviation of predictions (uncertainty)
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            pred, _ = self.predict(history, horizon, return_uncertainty=False)
            predictions.append(pred)

        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)

        self.eval()
        return uncertainty

    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'input_size': self.input_size,
            'output_size': self.output_size
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
