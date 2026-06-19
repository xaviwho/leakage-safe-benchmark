"""
Benchmark inference latency and memory footprint for all 4 models.

Measures:
- Single-sample inference time (ms) averaged over 100 runs
- Batch inference time (ms) for 30 samples (full test set)
- Model memory footprint (MB)
- Parameter count

Results saved to experiments/results/inference_benchmarks.json
"""
import numpy as np
import pickle
import json
import time
import sys
import os
import joblib
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.models.predictors import TransformerPredictor
from src.models.simulators import iPSCDifferentiationSimulator
from src.utils import load_config

print("=" * 80)
print("INFERENCE LATENCY & MEMORY BENCHMARKING")
print("=" * 80)

# =========================================================================
# Load test data (same split as all evaluations)
# =========================================================================
data_path = Path('data/processed/dopaminergic_trajectories_pseudotime.pkl')
with open(data_path, 'rb') as f:
    trajectories = pickle.load(f)

trajectories = np.array(trajectories)
np.random.seed(42)
indices = np.random.permutation(len(trajectories))
n_train = int(0.7 * len(trajectories))
n_val = int(0.15 * len(trajectories))
test_trajs = trajectories[indices[n_train + n_val:]]

# Prepare inputs
# sklearn format: flattened (n_samples, 4)
X_test_flat = np.array([t[:2].flatten() for t in test_trajs])  # (N, 4)
single_flat = X_test_flat[:1]  # (1, 4)

# Transformer format: (n_samples, seq_len=2, features=2)
X_test_seq = np.array([t[:2] for t in test_trajs])  # (N, 2, 2)

n_test = len(test_trajs)
N_WARMUP = 10
N_RUNS = 100

print(f"\nTest samples: {n_test}")
print(f"Warmup runs: {N_WARMUP}, Timed runs: {N_RUNS}")

config = load_config()
results = {}


def get_object_size_mb(obj):
    """Estimate memory footprint using pickle serialization size."""
    import io
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell() / (1024 * 1024)


# =========================================================================
# 1. Linear Regression
# =========================================================================
print("\n" + "-" * 60)
print("1. LINEAR REGRESSION (Ridge)")
print("-" * 60)

lr_path = Path('experiments/results/baselines/linear_regression.pkl')
lr = joblib.load(lr_path)

# Parameter count: coefficients + intercept
lr_params = lr.coef_.size + lr.intercept_.size
lr_mem = get_object_size_mb(lr)

# Warmup
for _ in range(N_WARMUP):
    lr.predict(single_flat)

# Single-sample latency
times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    lr.predict(single_flat)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    times.append(elapsed)
lr_single_ms = np.median(times)

# Batch latency (full test set)
times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    lr.predict(X_test_flat)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
lr_batch_ms = np.median(times)

results['LinearRegression'] = {
    'single_latency_ms': float(lr_single_ms),
    'batch_latency_ms': float(lr_batch_ms),
    'memory_mb': float(lr_mem),
    'parameters': int(lr_params)
}

print(f"  Single inference: {lr_single_ms:.3f} ms")
print(f"  Batch ({n_test}):      {lr_batch_ms:.3f} ms")
print(f"  Memory:           {lr_mem:.4f} MB")
print(f"  Parameters:       {lr_params}")


# =========================================================================
# 2. Random Forest
# =========================================================================
print("\n" + "-" * 60)
print("2. RANDOM FOREST")
print("-" * 60)

rf_path = Path('experiments/results/baselines/random_forest.pkl')
rf = joblib.load(rf_path)

# Parameter count: approximate (n_estimators * avg_nodes)
rf_params = sum(tree.tree_.node_count for est in rf.estimators_
                for tree in (est if hasattr(est, '__iter__') else [est]))
rf_mem = get_object_size_mb(rf)

# Warmup
for _ in range(N_WARMUP):
    rf.predict(single_flat)

# Single-sample latency
times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    rf.predict(single_flat)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
rf_single_ms = np.median(times)

# Batch latency
times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    rf.predict(X_test_flat)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
rf_batch_ms = np.median(times)

results['RandomForest'] = {
    'single_latency_ms': float(rf_single_ms),
    'batch_latency_ms': float(rf_batch_ms),
    'memory_mb': float(rf_mem),
    'parameters': int(rf_params)
}

print(f"  Single inference: {rf_single_ms:.3f} ms")
print(f"  Batch ({n_test}):      {rf_batch_ms:.3f} ms")
print(f"  Memory:           {rf_mem:.4f} MB")
print(f"  Parameters:       {rf_params} (tree nodes)")


# =========================================================================
# 3. Transformer
# =========================================================================
print("\n" + "-" * 60)
print("3. TRANSFORMER")
print("-" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = TransformerPredictor(input_size=2, output_size=2, config=config)
checkpoint_path = Path('experiments/results/dopaminergic_transformer_2D/checkpoints/best_model.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Parameter count
tf_params = sum(p.numel() for p in model.parameters())
tf_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Memory: model state dict size
import io
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
tf_mem = buf.tell() / (1024 * 1024)

# Prepare tensors
single_tensor = torch.FloatTensor(X_test_seq[:1]).to(device)
batch_tensor = torch.FloatTensor(X_test_seq).to(device)

# Warmup
with torch.no_grad():
    for _ in range(N_WARMUP):
        model(single_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

# Single-sample latency
times = []
with torch.no_grad():
    for _ in range(N_RUNS):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(single_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
tf_single_ms = np.median(times)

# Batch latency
times = []
with torch.no_grad():
    for _ in range(N_RUNS):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(batch_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
tf_batch_ms = np.median(times)

results['Transformer'] = {
    'single_latency_ms': float(tf_single_ms),
    'batch_latency_ms': float(tf_batch_ms),
    'memory_mb': float(tf_mem),
    'parameters': int(tf_params),
    'trainable_parameters': int(tf_trainable),
    'device': str(device)
}

print(f"  Single inference: {tf_single_ms:.3f} ms")
print(f"  Batch ({n_test}):      {tf_batch_ms:.3f} ms")
print(f"  Memory:           {tf_mem:.4f} MB")
print(f"  Parameters:       {tf_params:,}")


# =========================================================================
# 4. ODE (Physics-Only)
# =========================================================================
print("\n" + "-" * 60)
print("4. ODE (Physics-Only)")
print("-" * 60)

sim = iPSCDifferentiationSimulator(config)

# Load calibrated params
params_path = Path('config/calibrated_ode_params.json')
if params_path.exists():
    with open(params_path, 'r') as f:
        calibrated_params = json.load(f)
    sim.params['k_pluri_deg'] = calibrated_params['k_pluri_deg']
    sim.params['k_diff_synth'] = calibrated_params['k_diff_synth']
    sim.params['diff_rate'] = calibrated_params['diff_rate']
    sim.params['k_basal'] = calibrated_params['k_basal']

ode_params = len(calibrated_params) if params_path.exists() else 4
ode_mem = get_object_size_mb(sim)

# Warmup
for _ in range(N_WARMUP):
    init = np.array([test_trajs[0][0][0], test_trajs[0][0][1], 50000.0])
    sim.run_simulation(duration=19.0, timesteps=20, initial_state=init)

# Single-sample latency (one D11->D30 prediction)
times = []
for _ in range(N_RUNS):
    state = test_trajs[0][0]
    init = np.array([state[0], state[1], 50000.0])
    start = time.perf_counter()
    _, states = sim.run_simulation(duration=19.0, timesteps=20, initial_state=init)
    pred = np.clip(states[-1, :2], 0.0, 1.0)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
ode_single_ms = np.median(times)

# Batch latency (all test trajectories, both transitions)
times = []
for _ in range(min(N_RUNS, 20)):  # fewer runs since ODE is slower
    start = time.perf_counter()
    for traj in test_trajs:
        # D11 -> D30
        init1 = np.array([traj[0][0], traj[0][1], 50000.0])
        _, s1 = sim.run_simulation(duration=19.0, timesteps=20, initial_state=init1)
        np.clip(s1[-1, :2], 0.0, 1.0)
        # D30 -> D52
        init2 = np.array([traj[1][0], traj[1][1], 50000.0])
        _, s2 = sim.run_simulation(duration=22.0, timesteps=23, initial_state=init2)
        np.clip(s2[-1, :2], 0.0, 1.0)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
ode_batch_ms = np.median(times)

results['ODE'] = {
    'single_latency_ms': float(ode_single_ms),
    'batch_latency_ms': float(ode_batch_ms),
    'memory_mb': float(ode_mem),
    'parameters': int(ode_params)
}

print(f"  Single inference: {ode_single_ms:.3f} ms")
print(f"  Batch ({n_test}×2):    {ode_batch_ms:.3f} ms")
print(f"  Memory:           {ode_mem:.4f} MB")
print(f"  Parameters:       {ode_params}")


# =========================================================================
# Save results
# =========================================================================
output_path = Path('experiments/results/inference_benchmarks.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

results['metadata'] = {
    'n_test_samples': int(n_test),
    'n_warmup': N_WARMUP,
    'n_timed_runs': N_RUNS,
    'aggregation': 'median',
    'platform': sys.platform,
    'device': str(device)
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVED] Benchmarks: {output_path}")

# =========================================================================
# Summary Table
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY: INFERENCE BENCHMARKS")
print("=" * 80)
print(f"\n{'Model':<25} {'Single (ms)':>12} {'Batch (ms)':>12} {'Memory (MB)':>12} {'Params':>10}")
print("-" * 75)
for name in ['LinearRegression', 'RandomForest', 'Transformer', 'ODE']:
    r = results[name]
    print(f"{name:<25} {r['single_latency_ms']:>12.3f} {r['batch_latency_ms']:>12.3f} {r['memory_mb']:>12.4f} {r['parameters']:>10,}")
print("-" * 75)
