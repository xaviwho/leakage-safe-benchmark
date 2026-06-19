"""
Training script for ML predictors.

This script:
1. Generates synthetic training data from ODE simulator
2. Trains LSTM or Transformer predictor
3. Evaluates on validation set
4. Saves trained model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import argparse
import logging
from datetime import datetime

from src.models.simulators import iPSCDifferentiationSimulator
from src.models.predictors import LSTMPredictor, TransformerPredictor
from src.models.predictors.trainer import PredictorTrainer, CellStateDataset
from src.data.data_generator import SyntheticDataGenerator
from src.utils import load_config, setup_logger

# Setup logging
logger = setup_logger("train_predictor", level="INFO")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML predictor for cell state prediction")

    parser.add_argument(
        '--model',
        type=str,
        default='lstm',
        choices=['lstm', 'transformer'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--n_train',
        type=int,
        default=1000,
        help='Number of training trajectories to generate'
    )
    parser.add_argument(
        '--n_val',
        type=int,
        default=100,
        help='Number of validation trajectories to generate'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=20,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--prediction_horizon',
        type=int,
        default=10,
        help='Number of steps to predict ahead'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--load_data',
        type=str,
        default=None,
        help='Path to pre-generated dataset (if None, generates new)'
    )
    parser.add_argument(
        '--save_data',
        action='store_true',
        help='Save generated dataset for future use'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this experiment (default: auto-generated)'
    )

    return parser.parse_args()


def generate_data(args, config):
    """Generate or load training data."""
    logger.info("=" * 60)
    logger.info("DATA GENERATION")
    logger.info("=" * 60)

    # Initialize simulator
    simulator = iPSCDifferentiationSimulator(config)

    # Initialize data generator
    data_generator = SyntheticDataGenerator(simulator, config)

    if args.load_data:
        # Load pre-generated data
        logger.info(f"Loading data from {args.load_data}")
        from src.data.data_generator import load_dataset
        train_trajectories = load_dataset(args.load_data)

        # Convert to numpy array if it's a list
        if isinstance(train_trajectories, list):
            train_trajectories = np.array(train_trajectories)

        # Check if data is raw numpy arrays (real data) or tuples (synthetic)
        if isinstance(train_trajectories[0], np.ndarray):
            # Real data format: just states
            logger.info(f"Loaded real trajectory data")

            # Split real data into train/val/test (70/15/15 split)
            n_total = len(train_trajectories)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            # test = remaining

            # Shuffle for random split with fixed seed for reproducibility
            np.random.seed(42)
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train+n_val]
            test_indices = indices[n_train+n_val:]

            train_states = [train_trajectories[i] for i in train_indices]
            val_states = [train_trajectories[i] for i in val_indices]
            test_states = [train_trajectories[i] for i in test_indices]

            logger.info(f"Split real data: {len(train_states)} train, {len(val_states)} val, {len(test_states)} test (held-out)")
        else:
            # Synthetic data format: (time, states)
            logger.info(f"Loaded synthetic trajectory data")
            train_states = [states for _, states in train_trajectories]

            # Generate synthetic validation set
            val_trajectories = data_generator.generate_validation_set(n_trajectories=args.n_val)
            val_states = [states for _, states in val_trajectories]
    else:
        # Generate new data
        logger.info(f"Generating {args.n_train} training trajectories...")
        train_trajectories = data_generator.generate_dataset(
            n_trajectories=args.n_train,
            timesteps=100,
            add_noise=True,
            noise_level=0.02,
            save_path='data/simulated/train_data.pkl' if args.save_data else None
        )

        logger.info(f"Generating {args.n_val} validation trajectories...")
        val_trajectories = data_generator.generate_validation_set(
            n_trajectories=args.n_val,
            save_path='data/simulated/val_data.pkl' if args.save_data else None
        )

        # Extract just the states (not time points) for dataset
        train_states = [states for _, states in train_trajectories]
        val_states = [states for _, states in val_trajectories]

    # Create datasets
    train_dataset = CellStateDataset(
        train_states,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )

    val_dataset = CellStateDataset(
        val_states,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_model(args, config, feature_size=None):
    """Create model based on arguments."""
    logger.info("=" * 60)
    logger.info("MODEL INITIALIZATION")
    logger.info("=" * 60)

    # Auto-detect feature size from data if not provided
    if feature_size is None:
        feature_size = 3  # Default: P, D, N
        logger.info(f"Using default feature size: {feature_size}")
    else:
        logger.info(f"Detected feature size from data: {feature_size}")

    input_size = feature_size
    output_size = feature_size

    if args.model == 'lstm':
        logger.info("Creating LSTM predictor...")
        model = LSTMPredictor(
            input_size=input_size,
            output_size=output_size,
            config=config
        )
    elif args.model == 'transformer':
        logger.info("Creating Transformer predictor...")
        model = TransformerPredictor(
            input_size=input_size,
            output_size=output_size,
            config=config
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Device: {model.device}")

    return model


def train_model(args, model, train_dataset, val_dataset, config):
    """Train the model."""
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    # Create experiment directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.model}_{timestamp}"

    experiment_dir = project_root / "experiments" / "results" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment directory: {experiment_dir}")

    # Update config with command line args
    if 'ml_models' not in config:
        config['ml_models'] = {}
    if args.model not in config['ml_models']:
        config['ml_models'][args.model] = {}

    config['ml_models'][args.model]['epochs'] = args.epochs
    config['ml_models'][args.model]['batch_size'] = args.batch_size

    # Create trainer
    trainer = PredictorTrainer(
        model=model,
        config=config,
        experiment_dir=experiment_dir
    )

    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_best=True
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Models saved to: {experiment_dir / 'checkpoints'}")

    return trainer, history


def evaluate_model(trainer, val_dataset):
    """Evaluate the trained model."""
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    metrics = trainer.evaluate(val_dataset, return_predictions=True)

    logger.info(f"Test MSE: {metrics['mse']:.6f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.6f}")
    logger.info(f"Test MAE: {metrics['mae']:.6f}")

    # Calculate per-variable metrics
    predictions = metrics['predictions']
    targets = metrics['targets']

    # Dynamically determine variable names based on feature count
    n_features = predictions.shape[-1]
    var_names = ['Pluripotency', 'Differentiation', 'Population'][:n_features]

    for i, var_name in enumerate(var_names):
        pred_var = predictions[:, :, i]
        target_var = targets[:, :, i]

        mae = np.mean(np.abs(pred_var - target_var))
        rmse = np.sqrt(np.mean((pred_var - target_var) ** 2))

        logger.info(f"{var_name} - MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config()

    logger.info("=" * 60)
    logger.info("iPSC DIGITAL TWIN - ML PREDICTOR TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Training trajectories: {args.n_train}")
    logger.info(f"Validation trajectories: {args.n_val}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Prediction horizon: {args.prediction_horizon}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Generate data
    train_dataset, val_dataset = generate_data(args, config)

    # Detect feature size from data
    sample_input, sample_target = train_dataset[0]
    feature_size = sample_input.shape[-1]  # Last dimension is features
    logger.info(f"Auto-detected feature size: {feature_size} from data shape {sample_input.shape}")

    # Create model
    model = create_model(args, config, feature_size=feature_size)

    # Train model
    trainer, history = train_model(args, model, train_dataset, val_dataset, config)

    # Evaluate model
    metrics = evaluate_model(trainer, val_dataset)

    # Save test metrics to JSON
    test_results = {
        'test_mse': float(metrics['mse']),
        'test_rmse': float(metrics['rmse']),
        'test_mae': float(metrics['mae'])
    }

    # Add per-feature metrics if available
    predictions = metrics['predictions']
    n_features = predictions.shape[-1]
    var_names = ['Pluripotency', 'Differentiation', 'Population'][:n_features]
    targets = metrics['targets']

    for i, var_name in enumerate(var_names):
        pred_var = predictions[:, :, i]
        target_var = targets[:, :, i]
        mae = np.mean(np.abs(pred_var - target_var))
        test_results[f'test_mae_{var_name}'] = float(mae)

    test_results_path = args.experiment_dir / 'test_results.json'
    import json
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to: {test_results_path}")

    logger.info("=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
