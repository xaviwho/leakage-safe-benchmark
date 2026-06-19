"""
Fix marker scoring and regenerate pseudotime trajectories with correct biological direction.

Problem: Current trajectories use scanpy's score_genes() which produces relative scores
that don't guarantee biological direction (pluripotency should decline, differentiation
should increase during development).

Solution: Compute raw mean expression scores and validate biological direction before
building pseudotime trajectories.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import pickle
from scipy.stats import spearmanr

print("=" * 80)
print("FIXING MARKER SCORES AND REGENERATING PSEUDOTIME TRAJECTORIES")
print("=" * 80)

# ========================================
# Step 1: Load raw data
# ========================================
print("\n[1/6] Loading raw data...")
data_path = Path("data/raw/dopaminergic_all_timepoints.h5")

if not data_path.exists():
    print(f"ERROR: Data not found at {data_path}")
    print("Please ensure the dopaminergic dataset is downloaded.")
    exit(1)

adata = sc.read_h5ad(data_path)
print(f"   Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Identify time column
time_col = None
for col in ['day', 'timepoint', 'development_stage', 'time']:
    if col in adata.obs.columns:
        time_col = col
        break

if time_col:
    print(f"   Time column: '{time_col}'")
    print(f"   Timepoints: {sorted(adata.obs[time_col].unique())}")
else:
    print("   WARNING: No time column found")

# ========================================
# Step 2: Preprocess (standard pipeline)
# ========================================
print("\n[2/6] Preprocessing...")

# Normalize to 10,000 counts per cell
print("   - Normalizing to 10,000 counts per cell")
sc.pp.normalize_total(adata, target_sum=1e4)

# Log-transform
print("   - Log-transforming: log1p(x)")
sc.pp.log1p(adata)

# ========================================
# Step 3: Compute marker scores (CORRECTED METHOD)
# ========================================
print("\n[3/6] Computing marker scores (corrected method)...")

# Define marker genes
pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']
differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']

# Check availability
pluri_found = [g for g in pluripotency_genes if g in adata.var_names]
diff_found = [g for g in differentiation_genes if g in adata.var_names]

print(f"   Pluripotency genes found: {len(pluri_found)}/{len(pluripotency_genes)}")
print(f"      {pluri_found}")
print(f"   Differentiation genes found: {len(diff_found)}/{len(differentiation_genes)}")
print(f"      {diff_found}")

if len(pluri_found) == 0 or len(diff_found) == 0:
    print("   ERROR: Insufficient marker genes found!")
    exit(1)

# Compute RAW MEAN EXPRESSION (not relative to control genes)
print("\n   Computing raw mean expression scores...")

# Extract expression matrix for marker genes
pluri_expr = adata[:, pluri_found].X
diff_expr = adata[:, diff_found].X

# Convert sparse to dense if needed
if hasattr(pluri_expr, 'toarray'):
    pluri_expr = pluri_expr.toarray()
if hasattr(diff_expr, 'toarray'):
    diff_expr = diff_expr.toarray()

# Compute mean across genes (already log1p transformed)
P_raw = pluri_expr.mean(axis=1)
D_raw = diff_expr.mean(axis=1)

# Store raw scores
adata.obs['P_raw'] = P_raw
adata.obs['D_raw'] = D_raw

print(f"   P_raw range: [{P_raw.min():.3f}, {P_raw.max():.3f}]")
print(f"   D_raw range: [{D_raw.min():.3f}, {D_raw.max():.3f}]")

# ========================================
# Step 4: Validate biological direction
# ========================================
print("\n[4/6] Validating biological direction...")

if time_col:
    timepoints = sorted(adata.obs[time_col].unique())
    print(f"\n   Population-level trends across timepoints:")
    print(f"   {'Timepoint':<15} {'P_median':<12} {'D_median':<12} {'n_cells':<10}")
    print(f"   {'-'*50}")

    time_stats = []
    for tp in timepoints:
        mask = adata.obs[time_col] == tp
        P_med = np.median(P_raw[mask])
        D_med = np.median(D_raw[mask])
        n = mask.sum()
        time_stats.append({'timepoint': tp, 'P_med': P_med, 'D_med': D_med, 'n': n})
        print(f"   {str(tp):<15} {P_med:>11.3f} {D_med:>11.3f} {n:>9,}")

    # Check if biological expectation is met
    P_meds = [s['P_med'] for s in time_stats]
    D_meds = [s['D_med'] for s in time_stats]

    P_decreasing = P_meds[0] > P_meds[-1]  # Early > Late
    D_increasing = D_meds[0] < D_meds[-1]  # Early < Late

    print(f"\n   Biological expectation check:")
    print(f"   [OK] P decreasing? {P_decreasing} (early={P_meds[0]:.3f}, late={P_meds[-1]:.3f})")
    print(f"   [OK] D increasing? {D_increasing} (early={D_meds[0]:.3f}, late={D_meds[-1]:.3f})")

    if not P_decreasing or not D_increasing:
        print("\n   [WARNING]  WARNING: Biological trends not as expected!")
        print("   This may indicate:")
        print("     - Marker genes don't capture developmental progression")
        print("     - Time labels are incorrect")
        print("     - Gene expression preprocessing issues")
        print("\n   Proceeding anyway for trajectory construction...")
else:
    print("   No time column - skipping population validation")

# ========================================
# Step 5: Normalize scores to [0, 1]
# ========================================
print("\n[5/6] Normalizing scores to [0, 1]...")

# Min-max normalization (global across all cells)
P_min, P_max = P_raw.min(), P_raw.max()
D_min, D_max = D_raw.min(), D_raw.max()

P_norm = (P_raw - P_min) / (P_max - P_min)
D_norm = (D_raw - D_min) / (D_max - D_min)

adata.obs['P'] = P_norm
adata.obs['D'] = D_norm

print(f"   P normalized: [{P_norm.min():.3f}, {P_norm.max():.3f}]")
print(f"   D normalized: [{D_norm.min():.3f}, {D_norm.max():.3f}]")
print(f"   Normalization bounds stored for test set clipping")

# ========================================
# Step 6: Compute pseudotime and build trajectories
# ========================================
print("\n[6/6] Computing pseudotime and building trajectories...")

# PCA for diffusion map
print("   - Running PCA (50 components)...")
sc.tl.pca(adata, n_comps=50)

# Neighbor graph
print("   - Building neighbor graph (k=15)...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

# Diffusion pseudotime
print("   - Computing diffusion pseudotime...")
sc.tl.diffmap(adata, n_comps=15)

# Use first diffusion component as pseudotime
pseudotime = adata.obsm['X_diffmap'][:, 0]
adata.obs['pseudotime'] = pseudotime

# Normalize pseudotime to [0, 1]
pt_min, pt_max = pseudotime.min(), pseudotime.max()
pseudotime_norm = (pseudotime - pt_min) / (pt_max - pt_min)
adata.obs['pseudotime_norm'] = pseudotime_norm

print(f"   Pseudotime range: [{pseudotime_norm.min():.3f}, {pseudotime_norm.max():.3f}]")

# Check correlation between pseudotime and markers
rho_P, _ = spearmanr(pseudotime_norm, P_norm)
rho_D, _ = spearmanr(pseudotime_norm, D_norm)

print(f"\n   Pseudotime correlation with markers:")
print(f"   - rho(tau, P) = {rho_P:.3f} (expect negative if P declines)")
print(f"   - rho(tau, D) = {rho_D:.3f} (expect positive if D increases)")

# If correlations have wrong sign, reverse pseudotime
if rho_P > 0 or rho_D < 0:
    print("\n   WARNING: Reversing pseudotime direction to match biological expectation...")
    pseudotime_norm = 1.0 - pseudotime_norm
    adata.obs['pseudotime_norm'] = pseudotime_norm

    # Recheck correlations
    rho_P, _ = spearmanr(pseudotime_norm, P_norm)
    rho_D, _ = spearmanr(pseudotime_norm, D_norm)
    print(f"   After reversal:")
    print(f"   - rho(tau, P) = {rho_P:.3f}")
    print(f"   - rho(tau, D) = {rho_D:.3f}")

# Build pseudo-trajectories by binning pseudotime
print("\n   Building pseudo-trajectories...")
n_trajectories = 200
n_bins = 10

# Create pseudotime bins
pseudotime_bins = np.linspace(0, 1.0, n_bins + 1)
bin_centers = (pseudotime_bins[:-1] + pseudotime_bins[1:]) / 2

trajectories = []

for traj_idx in range(n_trajectories):
    trajectory = np.zeros((n_bins, 2))  # (10 bins, 2 features [P, D])

    for bin_idx in range(n_bins):
        # Find cells in this pseudotime bin
        bin_start = pseudotime_bins[bin_idx]
        bin_end = pseudotime_bins[bin_idx + 1]

        in_bin = (pseudotime_norm >= bin_start) & (pseudotime_norm < bin_end)

        if in_bin.sum() > 0:
            # Sample one cell from this bin
            cells_in_bin = np.where(in_bin)[0]
            sampled_cell = np.random.choice(cells_in_bin)

            trajectory[bin_idx, 0] = P_norm[sampled_cell]
            trajectory[bin_idx, 1] = D_norm[sampled_cell]
        else:
            # If no cells in bin, interpolate from neighbors
            if bin_idx > 0:
                trajectory[bin_idx] = trajectory[bin_idx - 1]
            else:
                trajectory[bin_idx] = [P_norm.mean(), D_norm.mean()]

    trajectories.append(trajectory)

trajectories = np.array(trajectories)
print(f"   Generated {len(trajectories)} pseudo-trajectories")
print(f"   Shape: {trajectories.shape} (n_traj, n_bins, n_features)")

# ========================================
# Validate trajectories
# ========================================
print("\n" + "=" * 80)
print("VALIDATION: Trajectory Monotonicity")
print("=" * 80)

all_P = trajectories[:, :, 0]
all_D = trajectories[:, :, 1]

# Population trends
median_P = np.median(all_P, axis=0)
median_D = np.median(all_D, axis=0)

print(f"\nPopulation median trends:")
print(f"P: {median_P[0]:.3f} -> {median_P[-1]:.3f} (change: {median_P[-1] - median_P[0]:+.3f})")
print(f"D: {median_D[0]:.3f} -> {median_D[-1]:.3f} (change: {median_D[-1] - median_D[0]:+.3f})")

# Per-trajectory correlations
rho_P_list = []
rho_D_list = []

for i in range(len(trajectories)):
    tau = np.linspace(0, 0.9, n_bins)
    rho_P, _ = spearmanr(tau, all_P[i])
    rho_D, _ = spearmanr(tau, all_D[i])
    rho_P_list.append(rho_P)
    rho_D_list.append(rho_D)

pct_P_decreasing = 100 * np.sum(np.array(rho_P_list) < -0.8) / len(rho_P_list)
pct_D_increasing = 100 * np.sum(np.array(rho_D_list) > 0.8) / len(rho_D_list)

print(f"\nPer-trajectory monotonicity:")
print(f"  {pct_P_decreasing:.1f}% trajectories have rho(P,tau) < -0.8 (strong decline)")
print(f"  {pct_D_increasing:.1f}% trajectories have rho(D,tau) > +0.8 (strong increase)")

# Mean correlations
mean_rho_P = np.mean(rho_P_list)
mean_rho_D = np.mean(rho_D_list)
print(f"\nMean Spearman correlations:")
print(f"  rho(P,tau) = {mean_rho_P:.3f} (expect negative)")
print(f"  rho(D,tau) = {mean_rho_D:.3f} (expect positive)")

# ========================================
# Save results
# ========================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save trajectories
traj_path = output_dir / "dopaminergic_trajectories_pseudotime_FIXED.pkl"
with open(traj_path, 'wb') as f:
    pickle.dump(trajectories.tolist(), f)
print(f"\n[OK] Trajectories saved: {traj_path}")

# Save normalization bounds (for test set clipping)
bounds = {
    'P_min': float(P_min),
    'P_max': float(P_max),
    'D_min': float(D_min),
    'D_max': float(D_max)
}
bounds_path = output_dir / "normalization_bounds.pkl"
with open(bounds_path, 'wb') as f:
    pickle.dump(bounds, f)
print(f"[OK] Normalization bounds saved: {bounds_path}")

# Save population statistics
if time_col:
    pop_stats = []
    for tp in timepoints:
        mask = adata.obs[time_col] == tp
        pop_stats.append({
            'timepoint': str(tp),
            'n_cells': int(mask.sum()),
            'P_mean': float(P_norm[mask].mean()),
            'P_median': float(np.median(P_norm[mask])),
            'P_std': float(P_norm[mask].std()),
            'D_mean': float(D_norm[mask].mean()),
            'D_median': float(np.median(D_norm[mask])),
            'D_std': float(D_norm[mask].std())
        })

    pop_df = pd.DataFrame(pop_stats)
    pop_path = output_dir / "population_statistics_FIXED.csv"
    pop_df.to_csv(pop_path, index=False)
    print(f"[OK] Population statistics saved: {pop_path}")

# Save validation metrics
validation = {
    'n_trajectories': len(trajectories),
    'n_bins': n_bins,
    'pct_P_decreasing': float(pct_P_decreasing),
    'pct_D_increasing': float(pct_D_increasing),
    'mean_rho_P': float(mean_rho_P),
    'mean_rho_D': float(mean_rho_D),
    'population_P_change': float(median_P[-1] - median_P[0]),
    'population_D_change': float(median_D[-1] - median_D[0])
}

validation_path = output_dir / "trajectory_validation_metrics.pkl"
with open(validation_path, 'wb') as f:
    pickle.dump(validation, f)
print(f"[OK] Validation metrics saved: {validation_path}")

print("\n" + "=" * 80)
print("SUCCESS!")
print("=" * 80)
print("\nNext steps:")
print("1. Review validation metrics above")
print("2. If monotonicity looks good (>70% for both), proceed to:")
print("   - Replace old trajectories with FIXED version")
print("   - Retrain models on corrected data")
print("   - Regenerate figures")
print("\n3. If monotonicity still poor, investigate:")
print("   - Marker gene selection (are these genes actually developmental markers?)")
print("   - Pseudotime quality (does diffusion capture progression?)")
print("   - Dataset appropriateness (is this data suitable for pseudotime?)")
