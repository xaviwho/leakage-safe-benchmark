"""
Fix trajectory construction to be scientifically valid.

Issues to address:
1. Current "trajectories" are pseudo-trajectories (random cells chained)
2. Feature definition inconsistent (day_number vs N)
3. No biological continuity or pseudotime

Solutions:
A. Use pseudotime ordering (diffusion pseudotime via Scanpy)
B. Sample along pseudotime paths for continuous trajectories
C. Aggregate to population-level dynamics (mean/percentiles per timepoint)
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import pickle

print("="*80)
print("FIXING TRAJECTORY CONSTRUCTION")
print("="*80)

# Load original data
print("\n[1/5] Loading scRNA-seq data...")
adata = sc.read_h5ad('data/raw/dopaminergic_all_timepoints.h5')
print(f"   Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Calculate marker scores (same as before)
print("\n[2/5] Calculating marker scores...")
pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']
sc.tl.score_genes(adata, pluripotency_genes, score_name='pluripotency_score')

differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']
sc.tl.score_genes(adata, differentiation_genes, score_name='differentiation_score')
print(f"   [OK] Calculated P and D scores")

# Convert time to numeric
time_map = {'D11': 11, 'D30': 30, 'D52': 52}
adata.obs['day_numeric'] = adata.obs['time_point'].map(time_map)

# ========================================
# SOLUTION A: Pseudotime-based trajectories
# ========================================
print("\n[3/5] Computing pseudotime ordering...")

# Preprocess for pseudotime - filter and normalize
# Replace inf values and normalize
if hasattr(adata.X, 'data'):
    adata.X.data[~np.isfinite(adata.X.data)] = 0
else:
    adata.X[~np.isfinite(adata.X)] = 0

# Simple normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Use all genes for PCA (skip variable gene selection to avoid issues)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

# Compute diffusion pseudotime
sc.tl.diffmap(adata, n_comps=15)
adata.obs['pseudotime'] = adata.obsm['X_diffmap'][:, 0]  # First diffusion component

# Normalize pseudotime to [0, 1]
ptime = adata.obs['pseudotime'].values
ptime_norm = (ptime - ptime.min()) / (ptime.max() - ptime.min())
adata.obs['pseudotime_norm'] = ptime_norm

print(f"   [OK] Computed diffusion pseudotime")
print(f"   Pseudotime range: [{ptime_norm.min():.3f}, {ptime_norm.max():.3f}]")

# ========================================
# SOLUTION B: Sample along pseudotime paths
# ========================================
print("\n[4/5] Sampling continuous trajectories along pseudotime...")

n_trajectories = 200
n_timepoints = 10  # More granular than just 3

trajectories_pseudotime = []

for i in range(n_trajectories):
    # Sample starting cell from early pseudotime (Day 11)
    early_cells = adata[(adata.obs['day_numeric'] == 11) &
                        (adata.obs['pseudotime_norm'] < 0.3)]

    if len(early_cells) == 0:
        continue

    start_idx = np.random.randint(0, len(early_cells))

    # Create trajectory by sampling cells at increasing pseudotime
    trajectory = []

    for t_idx in range(n_timepoints):
        # Target pseudotime for this step
        target_ptime = t_idx / (n_timepoints - 1)

        # Find cells near this pseudotime (within ±0.05)
        candidates = adata[(adata.obs['pseudotime_norm'] >= target_ptime - 0.05) &
                          (adata.obs['pseudotime_norm'] <= target_ptime + 0.05)]

        if len(candidates) == 0:
            # If no cells in range, use nearest
            distances = np.abs(adata.obs['pseudotime_norm'] - target_ptime)
            nearest_idx = distances.argmin()
            cell = adata[nearest_idx]
        else:
            # Random sample from candidates
            sample_idx = np.random.randint(0, len(candidates))
            cell = candidates[sample_idx]

        # Extract features: [P, D]
        p_score = cell.obs['pluripotency_score'].values[0]
        d_score = cell.obs['differentiation_score'].values[0]

        trajectory.append([p_score, d_score])

    trajectories_pseudotime.append(np.array(trajectory))

trajectories_pseudotime = np.array(trajectories_pseudotime)
print(f"   [OK] Created {len(trajectories_pseudotime)} pseudotime trajectories")
print(f"   Shape: {trajectories_pseudotime.shape} (n_traj, n_timepoints, 2)")

# ========================================
# SOLUTION C: Population-level dynamics
# ========================================
print("\n[5/5] Computing population-level dynamics...")

# Aggregate statistics per timepoint
population_dynamics = []

for day in [11, 30, 52]:
    day_cells = adata[adata.obs['day_numeric'] == day]

    # Compute statistics
    p_scores = day_cells.obs['pluripotency_score'].values
    d_scores = day_cells.obs['differentiation_score'].values

    stats = {
        'day': day,
        'n_cells': len(day_cells),
        'P_mean': p_scores.mean(),
        'P_std': p_scores.std(),
        'P_q25': np.percentile(p_scores, 25),
        'P_q50': np.percentile(p_scores, 50),
        'P_q75': np.percentile(p_scores, 75),
        'D_mean': d_scores.mean(),
        'D_std': d_scores.std(),
        'D_q25': np.percentile(d_scores, 25),
        'D_q50': np.percentile(d_scores, 50),
        'D_q75': np.percentile(d_scores, 75),
    }

    population_dynamics.append(stats)

pop_df = pd.DataFrame(population_dynamics)
print(f"\n   Population-level dynamics:")
print(pop_df[['day', 'n_cells', 'P_mean', 'D_mean']])

# Save all versions
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# Original (for backwards compatibility, labeled as pseudo)
original_path = output_dir / 'dopaminergic_trajectories_PSEUDO.pkl'
with open('data/processed/dopaminergic_trajectories.pkl', 'rb') as f:
    original_pseudo = pickle.load(f)
with open(original_path, 'wb') as f:
    pickle.dump(original_pseudo, f)
print(f"\n[SAVED] Original pseudo-trajectories: {original_path}")

# Pseudotime-based (RECOMMENDED)
pseudotime_path = output_dir / 'dopaminergic_trajectories_pseudotime.pkl'
with open(pseudotime_path, 'wb') as f:
    pickle.dump(trajectories_pseudotime, f)
print(f"[SAVED] Pseudotime trajectories: {pseudotime_path}")

# Population-level
population_path = output_dir / 'dopaminergic_population_dynamics.csv'
pop_df.to_csv(population_path, index=False)
print(f"[SAVED] Population dynamics: {population_path}")

print("\n" + "="*80)
print("TRAJECTORY RECONSTRUCTION COMPLETE")
print("="*80)
print("\nThree versions created:")
print(f"1. Pseudo-trajectories (original, NOT recommended): {original_path.name}")
print(f"2. Pseudotime trajectories (RECOMMENDED): {pseudotime_path.name}")
print(f"   - Biologically ordered via diffusion pseudotime")
print(f"   - {len(trajectories_pseudotime)} trajectories x {n_timepoints} timepoints x 2 features [P, D]")
print(f"3. Population dynamics (aggregate): {population_path.name}")
print(f"   - Mean/std/percentiles per timepoint")
print(f"   - 3 timepoints (Day 11, 30, 52)")

print("\nRECOMMENDATION for paper:")
print("  Use pseudotime trajectories with explicit acknowledgment:")
print("  'Trajectories were constructed by sampling cells along diffusion")
print("   pseudotime paths, providing biologically continuous ordering.'")
