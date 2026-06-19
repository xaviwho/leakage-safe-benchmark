"""
Extract differentiation trajectories from dopaminergic dataset.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import pickle

print("="*80)
print("EXTRACTING DOPAMINERGIC DIFFERENTIATION TRAJECTORIES")
print("="*80)

# 1. Load data
print("\n[STEP 1/5] Loading dataset...")
adata = sc.read_h5ad('data/raw/dopaminergic_all_timepoints.h5')
print(f"   Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# 2. Calculate marker scores
print("\n[STEP 2/5] Calculating pluripotency and differentiation scores...")

# Pluripotency markers
pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']
sc.tl.score_genes(adata, pluripotency_genes, score_name='pluripotency_score')

# Differentiation markers
differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']
sc.tl.score_genes(adata, differentiation_genes, score_name='differentiation_score')

print(f"   [OK] Calculated scores")

# Convert time_point to numeric
print("\n[STEP 3/5] Converting timepoints to numeric...")
time_map = {'D11': 11, 'D30': 30, 'D52': 52}
adata.obs['day_numeric'] = adata.obs['time_point'].map(time_map)
print(f"   [OK] Timepoints: {sorted(adata.obs['day_numeric'].unique())}")

# 3. Extract trajectories
print("\n[STEP 4/5] Extracting differentiation trajectories...")
n_trajectories = 200
trajectories = []

# Sample cells from each timepoint
for i in range(n_trajectories):
    trajectory = []

    # Sample one cell from each timepoint
    for day in [11, 30, 52]:
        cells_at_day = adata[adata.obs['day_numeric'] == day]

        # Randomly sample one cell
        idx = np.random.randint(0, cells_at_day.n_obs)
        cell = cells_at_day[idx]

        # Extract features: [pluripotency_score, differentiation_score, day]
        pluri = cell.obs['pluripotency_score'].values[0]
        diff = cell.obs['differentiation_score'].values[0]

        trajectory.append([pluri, diff, day])

    trajectories.append(np.array(trajectory))

print(f"   [OK] Extracted {len(trajectories)} trajectories")
print(f"   Each trajectory: {trajectories[0].shape[0]} timepoints x {trajectories[0].shape[1]} features")

# Show example
ex = trajectories[0]
print(f"\n   Example trajectory:")
print(f"   Day 11: P={ex[0,0]:.3f}, D={ex[0,1]:.3f}")
print(f"   Day 30: P={ex[1,0]:.3f}, D={ex[1,1]:.3f}")
print(f"   Day 52: P={ex[2,0]:.3f}, D={ex[2,1]:.3f}")

# 4. Save trajectories
print("\n[STEP 5/5] Saving processed data...")
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'dopaminergic_trajectories.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(trajectories, f)

print(f"   [OK] Saved to: {output_path}")

# Also save summary statistics
print("\n" + "="*80)
print("TRAJECTORY EXTRACTION COMPLETE")
print("="*80)
print(f"Dataset: Dopaminergic Neuron Differentiation")
print(f"Cells: {adata.n_obs:,}")
print(f"Timepoints: Day 11, 30, 52")
print(f"Trajectories: {len(trajectories)}")
print(f"Features per timepoint: 3 (pluripotency, differentiation, day)")
print(f"Output: {output_path}")
print("="*80)

print("\n[SUCCESS] Ready for ML training!")
print("\nNext steps:")
print("  1. Train LSTM: python experiments/train_predictor.py --load_data data/processed/dopaminergic_trajectories.pkl --model lstm")
print("  2. Train Transformer: python experiments/train_predictor.py --load_data data/processed/dopaminergic_trajectories.pkl --model transformer")
