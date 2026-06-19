"""
Train the λ-network for learnable hybrid weighting.

Two-stage training:
1. Stage 1: Transformer and ODE already trained independently
2. Stage 2: Freeze both, train λ-network to minimize hybrid prediction error

Addresses METHODOLOGY §4.1: "Learnable state-dependent hybrid weighting"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
from pathlib import Path
from src.models.hybrid_digital_twin import HybridDigitalTwin
from src.utils import load_config

print("="*80)
print("LAMBDA-NETWORK TRAINING (TWO-STAGE)")
print("="*80)

# Load config
config = load_config()

# Load timepoint-based trajectories
data_path = Path('data/processed/dopaminergic_trajectories_pseudotime.pkl')
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

# Convert to numpy if needed
trajectories = np.array(trajectories)

print(f"\nLoaded {len(trajectories)} trajectories")
print(f"Trajectory shape: {trajectories[0].shape}")  # (3, 2) for 3 timepoints

# Use same split as training (70/15/15)
np.random.seed(42)
indices = np.random.permutation(len(trajectories))
n_train = int(0.7 * len(trajectories))
n_val = int(0.15 * len(trajectories))

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

train_trajs = [trajectories[i] for i in train_indices]
val_trajs = [trajectories[i] for i in val_indices]
test_trajs = [trajectories[i] for i in test_indices]

print(f"Split: {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test")

# Initialize hybrid digital twin with trained Transformer
print("\n" + "="*80)
print("LOADING TRAINED MODELS")
print("="*80)

transformer_checkpoint = 'experiments/results/dopaminergic_transformer_2D/checkpoints/best_model.pt'
twin = HybridDigitalTwin(
    config,
    ml_model_type='transformer',
    ml_checkpoint=transformer_checkpoint,
    input_size=2,
    output_size=2
)

print(f"Loaded Transformer from: {transformer_checkpoint}")
print(f"Lambda-network initialized: {twin.lambda_network}")

# Prepare training data for lambda-network
print("\n" + "="*80)
print("PREPARING LAMBDA-NETWORK TRAINING DATA")
print("="*80)

def create_lambda_training_data(trajs):
    """
    Create (initial_state, target_state, horizon) tuples for lambda training.

    For 3-timepoint data (D11, D30, D52):
    - D11 -> D30: horizon = 19 days
    - D30 -> D52: horizon = 22 days
    - D11 -> D52: horizon = 41 days
    """
    lambda_data = []

    # Timepoint intervals (in days)
    horizons = {
        (0, 1): 19,  # D11 -> D30
        (1, 2): 22,  # D30 -> D52
        (0, 2): 41   # D11 -> D52
    }

    for traj in trajs:
        # traj shape: (3, 2) - 3 timepoints × [P, D]
        for (start_idx, end_idx), horizon_days in horizons.items():
            initial = traj[start_idx]  # [P, D]
            target = traj[end_idx]  # [P, D]

            # Normalize horizon to [0, 1] range for consistency
            tau_horizon = horizon_days / 52.0  # 52 days is max

            lambda_data.append((initial, target, tau_horizon))

    return lambda_data

train_lambda_data = create_lambda_training_data(train_trajs)
val_lambda_data = create_lambda_training_data(val_trajs)

print(f"Created {len(train_lambda_data)} training samples for lambda-network")
print(f"Created {len(val_lambda_data)} validation samples for lambda-network")

# Train lambda-network
print("\n" + "="*80)
print("TRAINING LAMBDA-NETWORK")
print("="*80)

optimizer = optim.Adam(twin.lambda_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training hyperparameters
n_epochs = 50
alpha = 0.01  # L1 regularization weight
batch_size = 32

# Training loop
train_losses = []
val_losses = []

twin.lambda_network.train()
best_val_loss = float('inf')

for epoch in range(n_epochs):
    # Shuffle training data
    np.random.shuffle(train_lambda_data)

    epoch_loss = 0
    epoch_reg = 0
    n_batches = 0

    for batch_start in range(0, len(train_lambda_data), batch_size):
        batch_end = min(batch_start + batch_size, len(train_lambda_data))
        batch = train_lambda_data[batch_start:batch_end]

        batch_loss = 0
        batch_reg_loss = 0

        for initial, target, horizon in batch:
            # Get physics and ML predictions
            tau, physics_pred = twin.predict_physics_only(initial, horizon, n_steps=20)
            final_physics = physics_pred[-1, :2]  # [P, D] at final pseudotime

            # ML prediction
            if len(physics_pred) >= 3:
                recent = physics_pred[-3:-1]  # Last 2 states
                ml_pred = twin.predict_ml_only(recent)
                if hasattr(ml_pred, 'shape') and len(ml_pred.shape) > 1:
                    ml_pred = ml_pred[0]  # Extract from batch dimension
                ml_pred = ml_pred[:2]  # [P, D] only
            else:
                continue  # Skip if not enough history

            # Residual
            residual = ml_pred - final_physics

            # Predicted lambda
            state_tensor = torch.FloatTensor(final_physics)
            lambda_pred = twin.lambda_network(state_tensor)

            # Hybrid prediction
            hybrid = final_physics + lambda_pred.item() * residual

            # Loss: distance to true target + L1 regularization
            target_tensor = torch.FloatTensor(target[:2])
            hybrid_tensor = torch.FloatTensor(hybrid)
            mse_loss = criterion(hybrid_tensor, target_tensor)
            l1_reg = alpha * torch.abs(lambda_pred).mean()

            loss = mse_loss + l1_reg

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += mse_loss.item()
            batch_reg_loss += l1_reg.item()

        epoch_loss += batch_loss
        epoch_reg += batch_reg_loss
        n_batches += 1

    avg_train_loss = epoch_loss / len(train_lambda_data)
    train_losses.append(avg_train_loss)

    # Validation
    twin.lambda_network.eval()
    val_loss = 0

    with torch.no_grad():
        for initial, target, horizon in val_lambda_data[:100]:  # Sample for speed
            tau, physics_pred = twin.predict_physics_only(initial, horizon, n_steps=20)
            final_physics = physics_pred[-1, :2]

            if len(physics_pred) >= 3:
                recent = physics_pred[-3:-1]
                ml_pred = twin.predict_ml_only(recent)
                if hasattr(ml_pred, 'shape') and len(ml_pred.shape) > 1:
                    ml_pred = ml_pred[0]
                ml_pred = ml_pred[:2]
            else:
                continue

            residual = ml_pred - final_physics
            state_tensor = torch.FloatTensor(final_physics)
            lambda_pred = twin.lambda_network(state_tensor)
            hybrid = final_physics + lambda_pred.item() * residual

            target_tensor = torch.FloatTensor(target[:2])
            hybrid_tensor = torch.FloatTensor(hybrid)
            loss = criterion(hybrid_tensor, target_tensor)
            val_loss += loss.item()

    avg_val_loss = val_loss / min(100, len(val_lambda_data))
    val_losses.append(avg_val_loss)

    twin.lambda_network.train()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} (MSE: {epoch_loss/len(train_lambda_data):.6f}, Reg: {epoch_reg/len(train_lambda_data):.6f})")
        print(f"  Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = Path('experiments/results/lambda_network/best_lambda.pt')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'lambda_network_state_dict': twin.lambda_network.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss
        }, save_path)

twin.lambda_network.eval()

print("\n" + "="*80)
print("LAMBDA-NETWORK TRAINING COMPLETE")
print("="*80)
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Saved to: experiments/results/lambda_network/best_lambda.pt")

# Analyze learned lambda values
print("\n" + "="*80)
print("ANALYZING LEARNED LAMBDA VALUES")
print("="*80)

lambda_values = []
states_PD = []

with torch.no_grad():
    for traj in test_trajs[:20]:  # Sample test trajectories
        for state in traj:
            state_tensor = torch.FloatTensor(state[:2])  # [P, D]
            lambda_val = twin.lambda_network(state_tensor).item()
            lambda_values.append(lambda_val)
            states_PD.append(state[:2])

lambda_values = np.array(lambda_values)
states_PD = np.array(states_PD)

print(f"\nLambda statistics on test trajectories:")
print(f"  Mean: {lambda_values.mean():.3f}")
print(f"  Std: {lambda_values.std():.3f}")
print(f"  Min: {lambda_values.min():.3f}")
print(f"  Max: {lambda_values.max():.3f}")
print(f"  Median: {np.median(lambda_values):.3f}")

# Correlations with state
print(f"\nCorrelation with state variables:")
print(f"  Lambda vs P: {np.corrcoef(states_PD[:, 0], lambda_values)[0, 1]:.3f}")
print(f"  Lambda vs D: {np.corrcoef(states_PD[:, 1], lambda_values)[0, 1]:.3f}")

# Save results to JSON
results = {
    'best_val_loss': float(best_val_loss),
    'lambda_mean': float(lambda_values.mean()),
    'lambda_std': float(lambda_values.std()),
    'lambda_min': float(lambda_values.min()),
    'lambda_max': float(lambda_values.max()),
    'lambda_median': float(np.median(lambda_values)),
    'corr_lambda_P': float(np.corrcoef(states_PD[:, 0], lambda_values)[0, 1]),
    'corr_lambda_D': float(np.corrcoef(states_PD[:, 1], lambda_values)[0, 1])
}

results_path = Path('experiments/results/lambda_network/results.json')
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] Results: {results_path}")

print("\n" + "="*80)
print("SUCCESS")
print("="*80)
print("Lambda-network trained with two-stage approach!")
print("Physics and ML models frozen, lambda learned adaptively.")
