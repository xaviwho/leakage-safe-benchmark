"""
Process dopaminergic differentiation dataset.
"""
import sys
import numpy as np
import scanpy as sc
from pathlib import Path

print("="*80)
print("PROCESSING DOPAMINERGIC DIFFERENTIATION DATASET")
print("="*80)

# Load data
print("\n1. Loading dataset...")
filepath = "data/raw/dopaminergic_all_timepoints.h5"
adata = sc.read_h5ad(filepath)

print(f"   [OK] Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Check metadata
print("\n2. Analyzing metadata...")
print(f"   Observation columns: {list(adata.obs.columns[:15])}")

# Find time column
time_cols = ['day', 'timepoint', 'time', 'days', 'development_stage', 'age']
time_col = None

for col in time_cols:
    if col in adata.obs.columns:
        time_col = col
        unique_vals = sorted(adata.obs[col].unique())
        print(f"\n   [OK] Found time column: '{col}'")
        print(f"   Timepoints: {unique_vals}")
        print(f"   Cells per timepoint:")
        for val in unique_vals:
            count = (adata.obs[col] == val).sum()
            print(f"      {val}: {count:,} cells")
        break

if not time_col:
    print("\n   [WARNING] No standard time column found")
    print(f"   Available columns: {list(adata.obs.columns)}")

# Check for marker genes
print("\n3. Checking for marker genes...")
pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']
differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']

pluri_found = [g for g in pluripotency_genes if g in adata.var_names]
diff_found = [g for g in differentiation_genes if g in adata.var_names]

print(f"   Pluripotency genes found: {len(pluri_found)}/{len(pluripotency_genes)}")
print(f"      {pluri_found}")
print(f"   Differentiation genes found: {len(diff_found)}/{len(differentiation_genes)}")
print(f"      {diff_found}")

# Basic preprocessing
print("\n4. Basic preprocessing...")
print(f"   Original: {adata.n_obs} cells, {adata.n_vars} genes")

# Calculate QC metrics if needed
if 'n_genes' not in adata.obs.columns:
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    print(f"   [OK] Calculated QC metrics")

print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"Cells: {adata.n_obs:,}")
print(f"Genes: {adata.n_vars:,}")
print(f"Time column: {time_col}")
print(f"Marker genes available: {len(pluri_found + diff_found)}/{len(pluripotency_genes + differentiation_genes)}")
print("="*80)

print("\n[SUCCESS] Dataset inspection complete!")
print(f"\nNext step: Extract trajectories for ML training")
