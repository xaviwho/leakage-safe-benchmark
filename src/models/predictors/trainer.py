"""
Training utilities for ML predictors.

Provides training loop, data generation, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class CellStateDataset(Dataset):
    """
    Dataset for cell state trajectories.

    Each sample is a sequence of cell states (P, D, N) over time.
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        sequence_length: int = 20,
        prediction_horizon: int = 10
    ):
        """
        Initialize dataset.

        Args:
            trajectories: List of trajectory arrays, each of shape (T, 3)
            sequence_length: Length of input sequences
            prediction_horizon: Length of prediction target
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.samples = []

        # Create sliding window samples from trajectories
        for traj in trajectories:
            traj_len = len(traj)
            total_len = sequence_length + prediction_horizon

            for i in range(traj_len - total_len + 1):
                input_seq = traj[i:i + sequence_length]
                target_seq = traj[i + sequence_length:i + total_len]
                self.samples.append((input_seq, target_seq))

        logger.info(f"Dataset created with {len(self.samples)} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        input_seq, target_seq = self.samples[idx]
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target_seq)
        )


class PredictorTrainer:
    """
    Trainer for ML predictors.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None,
        experiment_dir: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            experiment_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.config = config or {}
        self.device = model.device

        # Setup experiment directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.experiment_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
        else:
            self.experiment_dir = None

        # Training settings
        model_config = self.config.get('ml_models', {}).get('lstm', {})
        self.learning_rate = model_config.get('learning_rate', 0.001)
        self.batch_size = model_config.get('batch_size', 32)
        self.epochs = model_config.get('epochs', 100)

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        logger.info("Trainer initialized")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Epochs: {self.epochs}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            save_best: Whether to save best model checkpoint

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

        best_val_loss = float('inf')

        logger.info("Starting training...")
        for epoch in range(self.epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validation
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.experiment_dir:
                        self._save_checkpoint('best_model.pt', epoch, val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}"
                )

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Checkpoint every 10 epochs
            if self.experiment_dir and (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, train_loss)

        logger.info("Training complete!")

        # Save final model and history
        if self.experiment_dir:
            self._save_checkpoint('final_model.pt', self.epochs, train_loss)
            self._save_history()

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_input)

            # Take only the prediction horizon steps
            predictions = predictions[:, -batch_target.size(1):, :]

            # Compute loss
            loss = self.criterion(predictions, batch_target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)

                # Forward pass
                predictions = self.model(batch_input)
                predictions = predictions[:, -batch_target.size(1):, :]

                # Compute loss
                loss = self.criterion(predictions, batch_target)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def evaluate(
        self,
        test_dataset: Dataset,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_dataset: Test dataset
            return_predictions: Whether to return predictions

        Returns:
            Dictionary of evaluation metrics
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)

                # Forward pass
                predictions = self.model(batch_input)
                predictions = predictions[:, -batch_target.size(1):, :]

                # Compute metrics
                mse = self.criterion(predictions, batch_target)
                mae = torch.mean(torch.abs(predictions - batch_target))

                total_loss += mse.item()
                total_mae += mae.item()
                n_batches += 1

                if return_predictions:
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(batch_target.cpu().numpy())

        metrics = {
            'mse': total_loss / n_batches,
            'rmse': np.sqrt(total_loss / n_batches),
            'mae': total_mae / n_batches
        }

        if return_predictions:
            metrics['predictions'] = np.concatenate(all_predictions, axis=0)
            metrics['targets'] = np.concatenate(all_targets, axis=0)

        logger.info(f"Evaluation - MSE: {metrics['mse']:.6f}, "
                   f"RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")

        return metrics

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.experiment_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
