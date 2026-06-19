# Critical Fixes Applied to METHODOLOGY.md

**Date:** February 10, 2026
**Status:** ✅ All must-fix issues resolved

---

## Fixed Issues

### 1. ✅ Hybrid Claims Contradiction **[CRITICAL]**
**Problem:** Claimed "adaptive physics-ML fusion" as contribution, but λ-network converged to λ≈0 (hybrid = physics-only)

**Fix Applied:**
- Reframed λ-network as **analysis tool** rather than performance improvement
- Changed heading from "Learnable Hybrid Weighting" to "Learnable Hybrid Weighting (Analysis)"
- Added explicit result: "λ-network converged to **λ ≈ 0**, indicating calibrated physics model alone is near-optimal"
- New interpretation: "Rather than demonstrating hybrid improvement, this serves as empirical validation that physics-based modeling with proper calibration can be competitive"

**Safe Claim:** "λ-network analysis empirically validates that calibrated physics model dominates on in-distribution test set"

---

### 2. ✅ Invalid Significance Tests **[CRITICAL]**
**Problem:** Claimed p=0.003 for "hybrid vs physics-only" improvement, but if λ≈0 then hybrid≈physics (contradiction)

**Fix Applied:**
- Removed all significance test claims (paired t-test, Wilcoxon, Cohen's d)
- Replaced §8 with honest reporting standard:
  - "All MAE values reported are mean MAE across held-out test pseudo-trajectories"
  - "Our current experimental design (single split, 30 test pseudo-trajectories) provides insufficient power for rigorous hypothesis testing"
  - "We therefore report **descriptive statistics only**"
  - Added comparative rankings instead of p-values

**Safe Claim:** "10.7% relative improvement represents a meaningful practical difference, though formal significance testing is deferred"

---

### 3. ✅ Imprecise Improvement Labeling **[REQUIRED]**
**Problem:** Stated "10.7% improvement" without clarifying it's relative MAE reduction

**Fix Applied:**
- Changed all instances from "10.7% improvement" to "**10.7% relative MAE reduction**"
- Added explicit calculation: "(0.544 - 0.486) / 0.544 = 0.107"
- Specified "evaluated on held-out test set" and "vs. best classical baseline"

**Safe Claim:** "Transformer achieves 10.7% relative MAE reduction over best classical baseline (Linear Regression) on held-out pseudo-trajectories"

---

### 4. ✅ Pseudo-Trajectory Clarification **[REQUIRED]**
**Problem:** Could mislead reviewers into thinking we tracked biological lineages

**Fix Applied:**
- Changed "200 trajectories" → "200 pseudo-trajectories" throughout
- Added explicit disclaimer: "**Important**: These are pseudo-trajectories sampled along diffusion pseudotime and do not represent tracked single-cell lineages"
- Updated notation from x(t) to x(τ) to emphasize pseudotime
- Changed "trajectories" to "pseudo-trajectories" in all test set descriptions

**Safe Claim:** "Pseudotime-aligned pseudo-trajectories enable learning differentiation progression from snapshot scRNA-seq"

---

### 5. ✅ Real-World Validation Overselling **[REQUIRED]**
**Problem:** Claimed r>0.97 as strong validation, but only 3 timepoints (correlation misleading)

**Fix Applied:**
- Changed heading from "Real-World Validation" to "Population-Level Validation"
- Removed "95% CI" claims (unjustified with 3 points)
- Added caveat: "computed on 3 timepoints"
- Toned down interpretation: "Model predictions qualitatively match expected developmental progression; correlation values should be interpreted cautiously given limited temporal resolution"
- Changed from "high fidelity" to "qualitatively match"

**Safe Claim:** "Population-level trends across observed stages are consistent with expected pluripotency decline and dopaminergic marker increase"

---

### 6. ✅ Temporal Extrapolation Narrative **[POLISH]**
**Problem:** Read as side experiment rather than main motivation

**Fix Applied:**
- Added connecting sentence: "**This motivates the physics component for extrapolation stability in late-stage differentiation**, even though λ-network analysis showed physics dominance on the in-distribution test set"
- Ties extrapolation results back to physics modeling motivation

**Safe Claim:** "ML struggles under early→late extrapolation (+29.0% MAE), highlighting out-of-distribution risk"

---

### 7. ✅ Normalization Bounds **[POLISH]**
**Problem:** Didn't mention test value clipping

**Fix Applied:**
- Added to normalization section: "**Test values are clipped to [0,1] using training bounds to avoid out-of-range artifacts.**"

**Safe Claim:** Test values properly bounded using training statistics

---

### 8. ✅ Summary Section Rewrite **[CRITICAL]**
**Problem:** Summary made unsupported claims about hybrid fusion

**Fix Applied:**
- Completely rewrote summary with **"Safe Claims (supported by results)"** section
- Removed: "adaptive physics-ML fusion" claim
- Removed: "validates against experimental trends (r > 0.97)" overselling
- Added: Honest framing as **"physics-informed ML framework"** not "hybrid digital twin"
- New focus: Proper calibration makes physics competitive, ML extrapolation fails, λ-network validates physics dominance

**New Safe Claims:**
1. Pseudotime-aligned pseudo-trajectories enable learning from scRNA-seq
2. Calibrated ODE provides strong baseline competitive with classical ML
3. Transformer achieves 10.7% relative MAE reduction vs. best baseline
4. Temporal extrapolation reveals ML out-of-distribution risk (+29.0%)
5. λ-network empirically validates calibrated physics model dominance (λ≈0)

---

## Claims You Can Confidently Make

### ✅ Safe to Claim:
- "Pseudotime-aligned pseudo-trajectories enable learning progression from snapshot scRNA-seq"
- "Calibrated 2-state mechanistic model provides strong baseline"
- "Transformer improves MAE over classical baselines by 10.7% on held-out pseudo-trajectories"
- "ML struggles under early→late extrapolation (+29% MAE), highlighting out-of-distribution risk"
- "λ-network analysis validates physics model dominance on in-distribution test set"
- "Proper ODE calibration (39.2% improvement) makes mechanistic models competitive"

### ❌ Do NOT Claim:
- "Hybrid fusion improves performance" (λ≈0 contradicts this)
- "Significant improvement of hybrid over physics-only" (no valid statistical test)
- "Adaptive physics-ML fusion" as working contribution (λ≈0 = pure physics)
- "Strong temporal validation" based on r>0.97 (only 3 timepoints)
- "Hybrid helps with extrapolation" (no evidence for this regime)

---

## What Changed in Practice

**Before (unsafe):**
> "This methodology developed a hybrid physics-ML digital twin that achieves 10.7% improvement through adaptive physics-ML fusion with significant improvement (p=0.003) validated against experimental trends (r>0.97)."

**After (safe):**
> "This methodology developed a physics-informed ML framework that achieves 10.7% relative MAE reduction over classical baselines on held-out pseudo-trajectories. λ-network analysis empirically validates that calibrated physics modeling is competitive, while temporal extrapolation reveals ML out-of-distribution limitations."

---

## Paper Positioning

**Old Framing (flawed):** "We built a hybrid system that outperforms both physics and ML alone"

**New Framing (defensible):** "We provide a rigorous comparative benchmark showing:
1. Proper ODE calibration makes physics competitive with ML
2. Deep learning beats classical baselines by 10.7%
3. ML fails under temporal extrapolation (+29%)
4. Learned gating empirically validates physics dominance (λ→0)"

This is still a strong contribution - it's just honest about what you actually found.

---

## Reviewer-Proofing Complete

All contradictions eliminated. Every claim is now:
- ✅ Supported by experimental results
- ✅ Quantitatively precise
- ✅ Properly caveated
- ✅ Statistically honest

**Status:** Ready for submission with defensible claims.
