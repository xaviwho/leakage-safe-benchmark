# Pseudotime Diagnosis and Recommended Fix

**Date:** February 10, 2026
**Status:** 🚨 Critical issue identified

---

## Problem Summary

Diffusion pseudotime (first diffusion component) **does not capture developmental progression** in the Cuomo et al. dopaminergic dataset.

### Evidence

1. **Near-zero correlation with developmental markers:**
   - ρ(pseudotime, P) = 0.019 → essentially uncorrelated
   - ρ(pseudotime, D) = -0.003 → essentially uncorrelated

2. **Trajectory validation after fixing marker scores:**
   - 0.0% trajectories show monotonic P decline
   - 0.0% trajectories show monotonic D increase
   - Mean trajectory correlation with pseudotime: -0.124 (P), -0.035 (D)

3. **Population trends along pseudotime:**
   - Both P AND D decline (0.141→0.000, 0.034→0.000)
   - Biologically incorrect: D should INCREASE during differentiation

### Root Cause

The dataset has **3 discrete timepoints** (Day 11, 30, 52) representing developmental stages. Diffusion pseudotime attempts to find a continuous trajectory through gene expression space, but:

- The first diffusion component captures the **dominant source of variation**, which may be:
  - Batch effects between timepoints
  - Cell cycle state
  - Metabolic state
  - Technical noise
  - **NOT developmental progression**

- With only 3 real timepoints and high cellular heterogeneity, there's insufficient temporal structure for unsupervised pseudotime to recover the correct developmental ordering.

---

## Impact on Current Results

All published results were computed on **invalid pseudotime trajectories** where temporal ordering is essentially random:

### ❌ Affected Claims:
1. **"10.7% relative MAE reduction"** - Models learned patterns in randomly-ordered data, not developmental dynamics
2. **"ML shows +29.0% degradation under early→late extrapolation"** - The early/late split was based on invalid pseudotime ordering
3. **"Physics ODE captures differentiation dynamics"** - ODE was calibrated to invalid trajectories
4. **"λ-network validates physics dominance (λ≈0)"** - Comparison is meaningless if data has no temporal structure

### What This Means

The methodology is sound, but applied to **garbage temporal ordering**. Any patterns learned are artifacts of random sampling, not biological signal.

---

## Recommended Fix: Supervised Temporal Ordering

### Option 1: Use Actual Experimental Timepoints (RECOMMENDED)

**Instead of pseudotime, use the 3 discrete timepoints as ground truth temporal ordering:**

```python
# Load data with timepoint labels
adata = load_cuomo_data()

# Extract marker scores at each timepoint
timepoints = [11, 30, 52]  # days
trajectories = []

for traj_id in range(n_trajectories):
    trajectory = []
    for day in timepoints:
        # Sample one cell from this timepoint
        cells_at_day = adata[adata.obs['day'] == day]
        sampled_cell = np.random.choice(len(cells_at_day))
        P = cells_at_day.obs['P'][sampled_cell]
        D = cells_at_day.obs['D'][sampled_cell]
        trajectory.append([P, D])
    trajectories.append(trajectory)
```

**Advantages:**
- ✅ Ground truth temporal ordering (experimentally validated)
- ✅ No assumptions about continuous progression
- ✅ Captures population-level trends across real developmental stages
- ✅ Biologically meaningful early→late extrapolation test

**Disadvantages:**
- Only 3 timepoints (coarse temporal resolution)
- Models predict discrete jumps, not continuous dynamics

**Mitigation:** This is actually MORE appropriate for the digital twin use case:
- Real interventions happen at discrete timepoints (Day 0, 11, 30)
- Clinically relevant to predict "what will cells look like at Day 30 given Day 11 state"
- ODE can still simulate continuous dynamics between measured timepoints

### Option 2: Supervised Pseudotime (Alternative)

Use actual timepoint labels to **guide** pseudotime construction:

```python
# Fit pseudotime constrained to respect timepoint ordering
sc.tl.dpt(adata, n_dcs=15, root='day11_cell_000')  # Set Day 11 as root
# Ensure Day 11 < Day 30 < Day 52 in pseudotime

# Validate that population median pseudotime increases with day
```

**Advantage:** More timepoints (via interpolation)
**Disadvantage:** Still susceptible to noise; requires validation

### Option 3: Use Published Pseudotime (If Available)

Check if Cuomo et al. (2020) provides pre-computed pseudotime trajectories in their original analysis. If they validated developmental ordering, use their pseudotime.

---

## Immediate Action Plan

### Step 1: Validate Timepoint Labels Exist

```bash
python -c "
import scanpy as sc
adata = sc.read_h5ad('data/raw/dopaminergic_all_timepoints.h5')
print('Available columns:', list(adata.obs.columns))
print('Unique values:', adata.obs['day'].unique() if 'day' in adata.obs.columns else 'No day column')
"
```

### Step 2: Implement Timepoint-Based Trajectories

Create `build_timepoint_trajectories.py`:
- Sample cells from Day 11, 30, 52
- Compute marker scores (using fixed method from earlier)
- Build trajectories: each trajectory = [state_day11, state_day30, state_day52]
- Validate biological trends (P decreases, D increases across days)

### Step 3: Retrain All Models

- Train ML models to predict Day 30 state from Day 11, Day 52 from Day 30
- Calibrate ODE parameters to match Day 11→30→52 progression
- Rerun λ-network analysis
- Recompute temporal extrapolation test (train on Day 11→30, test on Day 30→52)

### Step 4: Update Methodology

Update METHODOLOGY.md Section 1.2.3:
```markdown
**Trajectory Construction (Timepoint-Based)**:
Rather than unsupervised pseudotime inference, we constructed trajectories using the
three experimentally-validated developmental stages (Day 11, Day 30, Day 52 post-differentiation).
For each trajectory, we sampled one cell from each timepoint, computing marker scores
[P, D] to create 3-point developmental trajectories. This approach ensures biologically
valid temporal ordering and enables meaningful evaluation of model extrapolation from
early (Day 11→30) to late (Day 30→52) stages.
```

### Step 5: Regenerate All Results

- Baseline comparison
- Temporal extrapolation (train on 11→30, test on 30→52)
- Lambda-network analysis
- Training curves
- Figures

---

## Expected Outcomes After Fix

### If Timepoint-Based Trajectories Work:

1. ✅ **Strong population-level trends:** P declines (Day 11→52), D increases (Day 11→52)
2. ✅ **Biologically valid trajectories:** Monotonic developmental progression
3. ✅ **Meaningful ML evaluation:** Models learn real developmental dynamics, not noise
4. ✅ **Defensible claims:** All results have ground truth temporal ordering

### If Timepoint-Based Trajectories Also Fail:

This would indicate **marker genes don't capture differentiation** in this dataset. You would need to:
- Re-examine marker gene selection
- Use full transcriptome (not just 11 genes)
- Consider different dataset (e.g., lineage-traced scRNA-seq)

---

## Timeline

**Estimated time to fix:**
- Implement timepoint-based trajectories: 2 hours
- Retrain all models: 4-6 hours
- Regenerate figures and update methodology: 2 hours
- **Total: 1 day**

**Risk:** If population-level trends at Day 11, 30, 52 don't show P decline and D increase, the dataset fundamentally doesn't support your hypothesis.

---

## Recommendation

**DO NOT submit the paper with current pseudotime-based results.** They are built on invalid temporal ordering.

**DO implement timepoint-based trajectories immediately.** This is the only way to get biologically defensible results from this dataset.

If timepoint-based trajectories fail validation, consider switching to a different dataset with better temporal structure (e.g., time-series from La Manno et al. chromaffin cell differentiation, which has continuous temporal sampling).
