"""
Evaluate ODE (physics-only) baseline on the SAME test set as ML baselines.

CRITICAL: This ensures fair comparison by using:
- Same test trajectories
- Same one-step prediction task (2 timepoints -> next)
- Same normalized [0,1] scale
- Same MAE metric
"""
import numpy as np
import pickle
import json
from pathlib import Path
from src.models.simulators import iPSCDifferentiationSimulator
from src.utils import load_config

print("=" * 80)
print("ODE BASELINE EVALUATION (Fair Comparison)")
print("=" * 80)

# Load trajectories
data_path = Path('data/processed/dopaminergic_trajectories_pseudotime.pkl')
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

trajectories = np.array(trajectories)  # (N, 3, 2) for D11, D30, D52
print(f"\nLoaded {len(trajectories)} trajectories")
print(f"Shape: {trajectories.shape}")

# Same train/val/test split as baselines
n_total = len(trajectories)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

train_trajs = trajectories[:n_train]
val_trajs = trajectories[n_train:n_train+n_val]
test_trajs = trajectories[n_train+n_val:]

print(f"\nSplit: {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test")

# Load calibrated ODE parameters
params_path = Path('config/calibrated_ode_params.json')
if params_path.exists():
    with open(params_path, 'r') as f:
        calibrated_params = json.load(f)
    print(f"\nLoaded calibrated ODE parameters:")
    print(f"  k_pluri_deg:   {calibrated_params['k_pluri_deg']:.6f}")
    print(f"  k_diff_synth:  {calibrated_params['k_diff_synth']:.6f}")
    print(f"  diff_rate:     {calibrated_params['diff_rate']:.6f}")
    print(f"  k_basal:       {calibrated_params['k_basal']:.6f}")
else:
    print("\nWARNING: No calibrated parameters found, using defaults")
    calibrated_params = None

# Initialize simulator
config = load_config()
sim = iPSCDifferentiationSimulator(config)

# Apply calibrated parameters if available
if calibrated_params:
    sim.params['k_pluri_deg'] = calibrated_params['k_pluri_deg']
    sim.params['k_diff_synth'] = calibrated_params['k_diff_synth']
    sim.params['diff_rate'] = calibrated_params['diff_rate']
    sim.params['k_basal'] = calibrated_params['k_basal']

print("\n" + "=" * 80)
print("EVALUATION ON TEST SET (Same as ML Baselines)")
print("=" * 80)

def evaluate_ode_one_step(trajs, split_name):
    """
    Evaluate ODE on one-step predictions matching ML baseline evaluation.

    For 3-timepoint data:
    - Prediction 1: D11 -> D30 (19 days)
    - Prediction 2: D30 -> D52 (22 days)
    """
    all_errors_P = []
    all_errors_D = []

    for traj in trajs:
        # Transition 1: D11 -> D30 (19 days)
        state_D11 = traj[0]  # [P, D]
        target_D30 = traj[1]

        # ODE simulation: D11 -> D30
        # Need to add N (cell count) - use approximate value
        initial_state_1 = np.array([state_D11[0], state_D11[1], 50000.0])
        times_1, states_1 = sim.run_simulation(
            duration=19.0,  # 19 days
            timesteps=20,
            initial_state=initial_state_1
        )
        pred_D30 = states_1[-1, :2]  # [P, D]

        # Clip predictions to [0,1] (same constraint as normalized data)
        pred_D30 = np.clip(pred_D30, 0.0, 1.0)

        # Compute errors (normalized scale [0,1])
        error_P_1 = abs(pred_D30[0] - target_D30[0])
        error_D_1 = abs(pred_D30[1] - target_D30[1])
        all_errors_P.append(error_P_1)
        all_errors_D.append(error_D_1)

        # Transition 2: D30 -> D52 (22 days)
        state_D30 = traj[1]
        target_D52 = traj[2]

        # ODE simulation: D30 -> D52
        initial_state_2 = np.array([state_D30[0], state_D30[1], 50000.0])
        times_2, states_2 = sim.run_simulation(
            duration=22.0,  # 22 days
            timesteps=23,
            initial_state=initial_state_2
        )
        pred_D52 = states_2[-1, :2]

        # Clip predictions to [0,1]
        pred_D52 = np.clip(pred_D52, 0.0, 1.0)

        error_P_2 = abs(pred_D52[0] - target_D52[0])
        error_D_2 = abs(pred_D52[1] - target_D52[1])
        all_errors_P.append(error_P_2)
        all_errors_D.append(error_D_2)

    # Compute MAE (same formula as ML baselines)
    mae_P = np.mean(all_errors_P)
    mae_D = np.mean(all_errors_D)
    mae_overall = (mae_P + mae_D) / 2.0  # Average of P and D

    print(f"\n{split_name} Set Results:")
    print(f"  MAE (overall): {mae_overall:.4f}")
    print(f"  MAE (P):       {mae_P:.4f}")
    print(f"  MAE (D):       {mae_D:.4f}")
    print(f"  N predictions: {len(all_errors_P)}")

    return {
        'mae': mae_overall,
        'mae_P': mae_P,
        'mae_D': mae_D,
        'n_predictions': len(all_errors_P)
    }

# Evaluate on all splits
print("\nEvaluating ODE on same splits as ML baselines...")
train_results = evaluate_ode_one_step(train_trajs, "Train")
val_results = evaluate_ode_one_step(val_trajs, "Validation")
test_results = evaluate_ode_one_step(test_trajs, "Test")

# Save results
output = {
    'test_mae': test_results['mae'],
    'test_mae_P': test_results['mae_P'],
    'test_mae_D': test_results['mae_D'],
    'val_mae': val_results['mae'],
    'val_mae_P': val_results['mae_P'],
    'val_mae_D': val_results['mae_D'],
    'train_mae': train_results['mae'],
    'n_test_predictions': test_results['n_predictions'],
    'evaluation_note': 'ODE evaluated on same normalized [0,1] scale as ML baselines'
}

output_path = Path('experiments/results/ode_baseline_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 80)
print("COMPARISON WITH ML BASELINES")
print("=" * 80)

# Load baseline results for comparison
baseline_path = Path('experiments/results/baselines/comparison.json')
if baseline_path.exists():
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    lr_mae = baseline_data['LinearRegression']['test_mae']
    rf_mae = baseline_data['RandomForest']['test_mae']
    transformer_mae = baseline_data['comparison']['transformer_test_mae']
    ode_mae = test_results['mae']

    print(f"\nTest MAE Comparison (all on normalized [0,1] scale):")
    print(f"  Linear Regression:  {lr_mae:.4f}")
    print(f"  Random Forest:      {rf_mae:.4f}")
    print(f"  Transformer:        {transformer_mae:.4f}")
    print(f"  ODE (Physics-only): {ode_mae:.4f}")

    # Compute relative performance
    best_baseline = min(lr_mae, rf_mae)
    ode_vs_best = ((ode_mae - best_baseline) / best_baseline) * 100

    print(f"\nODE vs. Best Baseline (Linear Regression):")
    print(f"  {ode_vs_best:+.1f}% ({'worse' if ode_vs_best > 0 else 'better'})")

    if abs(ode_mae - best_baseline) > 0.4:
        print("\nWARNING: Large difference detected!")
        print("This may indicate ODE is not well-suited for this prediction task,")
        print("or that ODE parameters need further tuning.")
    else:
        print("\nODE performance is reasonable compared to baselines.")

print(f"\n[SAVED] ODE baseline results: {output_path}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nAll models are now evaluated on the SAME scale!")
print("Ready to update ablation table with fair comparison.")
