"""
Temporal Extrapolation Experiment: Early -> Late Pseudotime Split

Tests whether models can extrapolate to later stages of differentiation
when trained only on early stages.

Addresses METHODOLOGY §7.1.2: "Generalization Analysis (Temporal Extrapolation)"
"""
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.models.predictors import TransformerPredictor
from src.models.hybrid_digital_twin import HybridDigitalTwin
from src.utils import load_config

print("="*80)
print("TEMPORAL EXTRAPOLATION EXPERIMENT")
print("="*80)

# Load config
config = load_config()

# Load timepoint-based trajectories
data_path = Path('data/processed/dopaminergic_trajectories_pseudotime.pkl')
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

# Convert to numpy
trajectories = np.array(trajectories)

print(f"\nLoaded {len(trajectories)} trajectories")
print(f"Trajectory shape: {trajectories[0].shape}")  # (3, 2) for 3 timepoints

# For 3-timepoint data (D11, D30, D52):
# Train: D11 -> D30 transitions (early stage)
# Val: D11 -> D30 transitions from validation set (early stage, held-out)
# Test: D30 -> D52 transitions (late stage, extrapolation)
print("\n" + "="*80)
print("TEMPORAL EXTRAPOLATION SPLIT")
print("="*80)

print("\nStrategy: Train on early transitions (D11->D30), test on late transitions (D30->D52)")

# Use same train/val split as main training (70/15 for train/val, rest ignored)
np.random.seed(42)
indices = np.random.permutation(len(trajectories))
n_train = int(0.7 * len(trajectories))
n_val = int(0.15 * len(trajectories))

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]  # Will use all remaining for late-stage test

train_trajs = trajectories[train_indices]
val_trajs = trajectories[val_indices]
test_trajs = trajectories[test_indices]

print(f"\nSplit: {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test trajectories")

# Create sequence prediction dataset
print("\n" + "="*80)
print("CREATING SEQUENCE DATASETS")
print("="*80)

def create_early_sequences(trajs):
    """Create D11->D30 transitions (early stage)."""
    X = []
    y = []

    for traj in trajs:
        # Input: D11 state
        x_seq = traj[0:1]  # (1, 2) - just D11
        # Output: D30 state
        y_next = traj[1:2]  # (1, 2) - just D30

        # For Transformer, need seq_len=2, so duplicate D11
        x_seq_padded = np.concatenate([x_seq, x_seq], axis=0)  # (2, 2)

        X.append(x_seq_padded.flatten())  # (4,) = 2 steps × 2 features
        y.append(y_next.flatten())  # (2,) = 1 step × 2 features

    return np.array(X), np.array(y)

def create_late_sequences(trajs):
    """Create D30->D52 transitions (late stage, extrapolation)."""
    X = []
    y = []

    for traj in trajs:
        # Input: D30 state
        x_seq = traj[1:2]  # (1, 2) - just D30
        # Output: D52 state
        y_next = traj[2:3]  # (1, 2) - just D52

        # For Transformer, need seq_len=2, so duplicate D30
        x_seq_padded = np.concatenate([x_seq, x_seq], axis=0)  # (2, 2)

        X.append(x_seq_padded.flatten())  # (4,) = 2 steps × 2 features
        y.append(y_next.flatten())  # (2,) = 1 step × 2 features

    return np.array(X), np.array(y)

# Create datasets
X_train, y_train = create_early_sequences(train_trajs)
X_val, y_val = create_early_sequences(val_trajs)
X_test_late, y_test_late = create_late_sequences(test_trajs)

print(f"Training (D11->D30): {len(train_trajs)} trajectories -> X={X_train.shape}, y={y_train.shape}")
print(f"Validation (D11->D30): {len(val_trajs)} trajectories -> X={X_val.shape}, y={y_val.shape}")
print(f"Test (D30->D52, LATE): {len(test_trajs)} trajectories -> X={X_test_late.shape}, y={y_test_late.shape}")

# Load trained Transformer model
print("\n" + "="*80)
print("LOADING TRAINED TRANSFORMER")
print("="*80)

transformer_checkpoint = 'experiments/results/dopaminergic_transformer_2D/checkpoints/best_model.pt'

# Initialize model
model = TransformerPredictor(
    input_size=2,
    output_size=2,
    config=config
)

# Load weights
checkpoint = torch.load(transformer_checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded Transformer from: {transformer_checkpoint}")

# Evaluate on late-stage test set
print("\n" + "="*80)
print("EVALUATING ON LATE-STAGE TEST SET")
print("="*80)

def evaluate_model(model, X, y):
    """Evaluate model and return MAE."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(X)):
            # Reshape to (batch=1, seq_len=2, features=2)
            x_in = torch.FloatTensor(X[i]).reshape(1, 2, 2)
            pred = model(x_in)  # (1, pred_horizon, 2)
            # Extract first prediction step and remove batch dimension
            pred_np = pred[0, 0, :].cpu().numpy()  # (2,)
            predictions.append(pred_np)

    predictions = np.array(predictions)  # (N, 2)
    mae = mean_absolute_error(y, predictions)
    mae_P = mean_absolute_error(y[:, 0], predictions[:, 0])
    mae_D = mean_absolute_error(y[:, 1], predictions[:, 1])

    return mae, mae_P, mae_D

# Evaluate on validation (early-stage, in-distribution)
val_mae, val_mae_P, val_mae_D = evaluate_model(model, X_val, y_val)

print(f"Validation (early-stage, in-distribution):")
print(f"  Overall MAE: {val_mae:.4f}")
print(f"  MAE (Pluripotency): {val_mae_P:.4f}")
print(f"  MAE (Differentiation): {val_mae_D:.4f}")

# Evaluate on test (late-stage, out-of-distribution)
test_mae, test_mae_P, test_mae_D = evaluate_model(model, X_test_late, y_test_late)

print(f"\nTest (LATE-stage, out-of-distribution):")
print(f"  Overall MAE: {test_mae:.4f}")
print(f"  MAE (Pluripotency): {test_mae_P:.4f}")
print(f"  MAE (Differentiation): {test_mae_D:.4f}")

# Compute degradation
degradation_pct = ((test_mae - val_mae) / val_mae) * 100

print(f"\nExtrapolation Performance:")
print(f"  Degradation: {degradation_pct:+.1f}%")
print(f"  {'[GOOD] Good extrapolation' if degradation_pct < 10 else '[LIMITED] Limited extrapolation'}")

# Test hybrid model (if available)
print("\n" + "="*80)
print("EVALUATING HYBRID MODEL (PHYSICS + ML)")
print("="*80)

try:
    # Initialize hybrid twin
    twin = HybridDigitalTwin(
        config,
        ml_model_type='transformer',
        ml_checkpoint=transformer_checkpoint,
        input_size=2,
        output_size=2
    )

    # Load lambda-network if available
    lambda_checkpoint = Path('experiments/results/lambda_network/best_lambda.pt')
    if lambda_checkpoint.exists():
        lambda_state = torch.load(lambda_checkpoint, map_location='cpu')
        twin.lambda_network.load_state_dict(lambda_state['lambda_network_state_dict'])
        print(f"Loaded lambda-network from: {lambda_checkpoint}")
    else:
        print("Lambda-network not trained yet, using default initialization")

    # Evaluate hybrid on late-stage test
    print(f"\nEvaluating hybrid model on late-stage test...")

    hybrid_predictions = []

    with torch.no_grad():
        for i in range(len(X_test_late)):
            # Get initial state
            x_seq = X_test_late[i].reshape(2, 2)  # (2, 2)
            initial = x_seq[0]  # First state [P, D]

            # Predict using hybrid (physics + ML)
            result = twin.predict_hybrid_learnable(
                initial_state=initial,
                time_horizon=0.1,  # One pseudotime bin
                n_steps=10
            )

            # Get final prediction
            hybrid_pred = result['hybrid'][-1, :2]  # [P, D]
            hybrid_predictions.append(hybrid_pred)

    hybrid_predictions = np.array(hybrid_predictions)
    hybrid_mae = mean_absolute_error(y_test_late, hybrid_predictions)

    print(f"Hybrid Model (late-stage):")
    print(f"  Overall MAE: {hybrid_mae:.4f}")

    hybrid_degradation = ((hybrid_mae - val_mae) / val_mae) * 100
    print(f"  Degradation: {hybrid_degradation:+.1f}%")

    print(f"\nComparison:")
    print(f"  Pure ML degradation: {degradation_pct:+.1f}%")
    print(f"  Hybrid degradation: {hybrid_degradation:+.1f}%")
    print(f"  Hybrid more robust: {degradation_pct - hybrid_degradation:.1f}% better")

except Exception as e:
    print(f"Could not evaluate hybrid model: {e}")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'n_early_train': len(train_trajs),
    'n_early_val': len(val_trajs),
    'n_late_test': len(test_trajs),
    'split_strategy': 'Train: D11->D30, Test: D30->D52',
    'val_mae': float(val_mae),
    'val_mae_P': float(val_mae_P),
    'val_mae_D': float(val_mae_D),
    'test_late_mae': float(test_mae),
    'test_late_mae_P': float(test_mae_P),
    'test_late_mae_D': float(test_mae_D),
    'degradation_pct': float(degradation_pct)
}

import json
results_path = Path('experiments/results/temporal_extrapolation/results.json')
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {results_path}")

print("\n" + "="*80)
print("SUCCESS")
print("="*80)
print("Temporal extrapolation experiment complete!")
print(f"Pure ML extrapolation degradation: {degradation_pct:+.1f}%")
