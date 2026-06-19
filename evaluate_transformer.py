"""
Evaluate trained Transformer on test set and save results.

This ensures we have REAL test MAE, not hardcoded values.
"""
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from src.models.predictors import TransformerPredictor
from src.models.predictors.trainer import PredictorTrainer
from src.utils import load_config

print("=" * 80)
print("TRANSFORMER TEST SET EVALUATION")
print("=" * 80)

# Load data
data_path = Path('data/processed/dopaminergic_trajectories_pseudotime.pkl')
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

trajectories = np.array(trajectories)
print(f"\nLoaded {len(trajectories)} trajectories")
print(f"Shape: {trajectories.shape}")

# Same split as training (70/15/15)
np.random.seed(42)
indices = np.random.permutation(len(trajectories))
n_train = int(0.7 * len(trajectories))
n_val = int(0.15 * len(trajectories))

train_trajs = trajectories[:n_train]
val_trajs = trajectories[n_train:n_train+n_val]
test_trajs = trajectories[n_train+n_val:]

print(f"Split: {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test")

# Create dataset
from src.models.predictors.trainer import CellStateDataset

seq_length = 2  # For 3-timepoint data
pred_horizon = 1

test_dataset = CellStateDataset(
    test_trajs,
    sequence_length=seq_length,
    prediction_horizon=pred_horizon
)

print(f"\nTest dataset: {len(test_dataset)} samples")

# Load config
config = load_config()

# Load trained model
checkpoint_path = Path('experiments/results/dopaminergic_transformer_2D/checkpoints/best_model.pt')
if not checkpoint_path.exists():
    print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
    print("Please train the Transformer first.")
    exit(1)

print(f"\nLoading checkpoint from: {checkpoint_path}")

# Detect feature size from data
sample_input, sample_target = test_dataset[0]
feature_size = sample_input.shape[-1]
print(f"Feature size: {feature_size}")

# Create model with same architecture as training
model = TransformerPredictor(
    input_size=feature_size,
    output_size=feature_size,
    config=config
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']}")

# Create trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

trainer = PredictorTrainer(
    model=model,
    config=config,
    experiment_dir='experiments/results/dopaminergic_transformer_2D'
)

# Evaluate on test set
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

metrics = trainer.evaluate(test_dataset, return_predictions=True)

print(f"\nTest MSE:  {metrics['mse']:.6f}")
print(f"Test RMSE: {metrics['rmse']:.6f}")
print(f"Test MAE:  {metrics['mae']:.6f}")

# Per-feature metrics
predictions = metrics['predictions']
targets = metrics['targets']

n_features = predictions.shape[-1]
var_names = ['Pluripotency', 'Differentiation', 'Population'][:n_features]

print(f"\nPer-feature metrics:")
for i, var_name in enumerate(var_names):
    pred_var = predictions[:, :, i]
    target_var = targets[:, :, i]

    mae = np.mean(np.abs(pred_var - target_var))
    rmse = np.sqrt(np.mean((pred_var - target_var) ** 2))

    print(f"  {var_name:20s} - MAE: {mae:.6f}, RMSE: {rmse:.6f}")

# Save results
test_results = {
    'test_mse': float(metrics['mse']),
    'test_rmse': float(metrics['rmse']),
    'test_mae': float(metrics['mae'])
}

for i, var_name in enumerate(var_names):
    pred_var = predictions[:, :, i]
    target_var = targets[:, :, i]
    mae = np.mean(np.abs(pred_var - target_var))
    test_results[f'test_mae_{var_name}'] = float(mae)

output_path = Path('experiments/results/dopaminergic_transformer_2D/test_results.json')
with open(output_path, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n[SAVED] Test results: {output_path}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"\nThis is the REAL test MAE: {metrics['mae']:.4f}")
print("Use this value in all comparisons (not hardcoded values).")
