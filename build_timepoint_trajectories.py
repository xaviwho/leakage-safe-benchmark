"""
Build trajectories from actual experimental timepoints (D11, D30, D52).

This replaces the broken diffusion pseudotime approach with ground-truth temporal ordering.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import pickle
from scipy.stats import spearmanr

print("=" * 80)
print("BUILDING TIMEPOINT-BASED TRAJECTORIES")
print("=" * 80)

# Load data
print("\n[1/5] Loading data...")
data_path = Path("data/raw/dopaminergic_all_timepoints.h5")
adata = sc.read_h5ad(data_path)
print(f"   Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Filter to untreated cells only
print("\n[2/5] Filtering to untreated cells...")
adata = adata[adata.obs['treatment'] == 'NONE'].copy()
print(f"   Filtered: {adata.n_obs:,} cells (treatment='NONE')")

# Preprocess
print("\n[3/5] Preprocessing...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Compute marker scores
print("\n[4/5] Computing marker scores...")
pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']
differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']

pluri_found = [g for g in pluripotency_genes if g in adata.var_names]
diff_found = [g for g in differentiation_genes if g in adata.var_names]

print(f"   Pluripotency genes: {len(pluri_found)}/{len(pluripotency_genes)}")
print(f"   Differentiation genes: {len(diff_found)}/{len(differentiation_genes)}")

# Raw mean expression
pluri_expr = adata[:, pluri_found].X
diff_expr = adata[:, diff_found].X

if hasattr(pluri_expr, 'toarray'):
    pluri_expr = pluri_expr.toarray()
if hasattr(diff_expr, 'toarray'):
    diff_expr = diff_expr.toarray()

P_raw = pluri_expr.mean(axis=1)
D_raw = diff_expr.mean(axis=1)

# Normalize to [0, 1]
P_min, P_max = P_raw.min(), P_raw.max()
D_min, D_max = D_raw.min(), D_raw.max()

P_norm = (P_raw - P_min) / (P_max - P_min)
D_norm = (D_raw - D_min) / (D_max - D_min)

adata.obs['P'] = P_norm
adata.obs['D'] = D_norm

print(f"\n   Normalization:")
print(f"   P: [{P_norm.min():.3f}, {P_norm.max():.3f}]")
print(f"   D: [{D_norm.min():.3f}, {D_norm.max():.3f}]")

# Validate biological trends across timepoints
print("\n[5/5] Validating biological trends...")
timepoints = ['D11', 'D30', 'D52']
time_numeric = {'D11': 11, 'D30': 30, 'D52': 52}

print(f"\n   Population-level statistics:")
print(f"   {'Timepoint':<12} {'n_cells':<10} {'P_median':<12} {'D_median':<12}")
print(f"   {'-'*50}")

pop_stats = []
for tp in timepoints:
    mask = adata.obs['time_point'] == tp
    n = mask.sum()
    P_med = np.median(P_norm[mask])
    D_med = np.median(D_norm[mask])
    pop_stats.append({'tp': tp, 'n': n, 'P': P_med, 'D': D_med})
    print(f"   {tp:<12} {n:<10,} {P_med:>11.3f} {D_med:>11.3f}")

# Check biological expectation
P_vals = [s['P'] for s in pop_stats]
D_vals = [s['D'] for s in pop_stats]

P_decreasing = P_vals[0] > P_vals[-1]
D_increasing = D_vals[0] < D_vals[-1]

print(f"\n   Biological validation:")
print(f"   P decreasing? {P_decreasing} (D11={P_vals[0]:.3f} > D52={P_vals[-1]:.3f})")
print(f"   D increasing? {D_increasing} (D11={D_vals[0]:.3f} < D52={D_vals[-1]:.3f})")

if not P_decreasing:
    print("\n   WARNING: P not decreasing - may need to invert marker interpretation")
if not D_increasing:
    print("\n   WARNING: D not increasing - may need different markers")

# Build trajectories
print("\n" + "=" * 80)
print("BUILDING TRAJECTORIES")
print("=" * 80)

n_trajectories = 200

trajectories = []
for traj_idx in range(n_trajectories):
    trajectory = []

    for tp in timepoints:
        # Get cells at this timepoint
        mask = adata.obs['time_point'] == tp
        cell_indices = np.where(mask)[0]

        # Sample one cell
        sampled_idx = np.random.choice(cell_indices)

        P_val = P_norm[sampled_idx]
        D_val = D_norm[sampled_idx]

        trajectory.append([P_val, D_val])

    trajectories.append(trajectory)

trajectories = np.array(trajectories)
print(f"\nGenerated {len(trajectories)} trajectories")
print(f"Shape: {trajectories.shape} (n_traj, n_timepoints, n_features)")

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

all_P = trajectories[:, :, 0]
all_D = trajectories[:, :, 1]

# Population trends
median_P = np.median(all_P, axis=0)
median_D = np.median(all_D, axis=0)

print(f"\nPopulation median trends across trajectories:")
print(f"P: {median_P[0]:.3f} -> {median_P[1]:.3f} -> {median_P[2]:.3f}")
print(f"D: {median_D[0]:.3f} -> {median_D[1]:.3f} -> {median_D[2]:.3f}")
print(f"\nTotal change:")
print(f"P: {median_P[2] - median_P[0]:+.3f} (expect negative)")
print(f"D: {median_D[2] - median_D[0]:+.3f} (expect positive)")

# Per-trajectory correlations
rho_P_list = []
rho_D_list = []

timepoint_nums = np.array([11, 30, 52])
for i in range(len(trajectories)):
    rho_P, _ = spearmanr(timepoint_nums, all_P[i])
    rho_D, _ = spearmanr(timepoint_nums, all_D[i])
    rho_P_list.append(rho_P)
    rho_D_list.append(rho_D)

pct_P_decreasing = 100 * np.sum(np.array(rho_P_list) < -0.5) / len(rho_P_list)
pct_D_increasing = 100 * np.sum(np.array(rho_D_list) > 0.5) / len(rho_D_list)

print(f"\nPer-trajectory monotonicity (with only 3 points, threshold=0.5):")
print(f"  {pct_P_decreasing:.1f}% trajectories have rho(P,time) < -0.5")
print(f"  {pct_D_increasing:.1f}% trajectories have rho(D,time) > +0.5")

mean_rho_P = np.mean(rho_P_list)
mean_rho_D = np.mean(rho_D_list)
print(f"\nMean Spearman correlations:")
print(f"  rho(P,time) = {mean_rho_P:.3f} (expect negative)")
print(f"  rho(D,time) = {mean_rho_D:.3f} (expect positive)")

# Save
print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save trajectories (REPLACE old file)
traj_path = output_dir / "dopaminergic_trajectories_pseudotime.pkl"
with open(traj_path, 'wb') as f:
    pickle.dump(trajectories.tolist(), f)
print(f"\n[OK] Trajectories saved: {traj_path}")
print(f"      (REPLACED old pseudotime-based trajectories)")

# Save normalization bounds
bounds = {
    'P_min': float(P_min),
    'P_max': float(P_max),
    'D_min': float(D_min),
    'D_max': float(D_max)
}
bounds_path = output_dir / "normalization_bounds.pkl"
with open(bounds_path, 'wb') as f:
    pickle.dump(bounds, f)
print(f"[OK] Normalization bounds: {bounds_path}")

# Save population stats
pop_df = pd.DataFrame([
    {
        'timepoint': s['tp'],
        'n_cells': int(s['n']),
        'P_median': float(s['P']),
        'D_median': float(s['D'])
    }
    for s in pop_stats
])
pop_path = output_dir / "dopaminergic_population_dynamics.csv"
pop_df.to_csv(pop_path, index=False)
print(f"[OK] Population stats: {pop_path}")

# Save validation metrics
validation = {
    'n_trajectories': len(trajectories),
    'n_timepoints': 3,
    'timepoints': timepoints,
    'pct_P_decreasing': float(pct_P_decreasing),
    'pct_D_increasing': float(pct_D_increasing),
    'mean_rho_P': float(mean_rho_P),
    'mean_rho_D': float(mean_rho_D),
    'population_P_change': float(median_P[2] - median_P[0]),
    'population_D_change': float(median_D[2] - median_D[0])
}

validation_path = output_dir / "trajectory_validation_metrics.pkl"
with open(validation_path, 'wb') as f:
    pickle.dump(validation, f)
print(f"[OK] Validation metrics: {validation_path}")

print("\n" + "=" * 80)
if pct_P_decreasing > 50 and pct_D_increasing > 50:
    print("SUCCESS: Trajectories show expected biological trends!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Retrain models on corrected trajectories")
    print("2. Update temporal extrapolation split (train: D11->D30, test: D30->D52)")
    print("3. Regenerate all figures")
    print("4. Update METHODOLOGY.md to describe timepoint-based approach")
else:
    print("WARNING: Biological trends weak or wrong direction")
    print("=" * 80)
    print("\nPossible issues:")
    print("- Marker genes don't capture differentiation")
    print("- Need to use full transcriptome, not just 11 genes")
    print("- Consider different dataset")
