# Project Report: Auditing and Reframing a Transformer "Digital Twin" for Stem-Cell Differentiation Forecasting

**Author:** Victor Ikenna Kanu — vkanu@kumoh.ac.kr
**Status:** investigation complete, manuscript drafted (`paper/manuscript.tex`)

---

## 1. Starting point and objective

We began from `journal_expansion_implementation.md`, a spec to upgrade a conference paper (a Transformer "digital twin" for iPSC→dopaminergic-neuron differentiation) into a methods-journal submission. The plan's governing principle was explicit: **kill the riskiest claim first.** That claim was the headline result — a **+19.4%** accuracy gain for an 827K-parameter Transformer over classical baselines, established on a **single seed** over **n=30** randomly-paired pseudo-trajectories.

The plan defined a decision gate: if the Transformer edge survived multi-seed statistics → proceed as a "method wins" paper; if not → pivot to a methodological-null thesis, itself publishable. Everything below was ordered to resolve that gate before investing in prose or figures.

---

## 2. Methodology of the investigation

We worked **gate-driven and de-risking-first**, in six numbered phases, each producing a reproducible script and a JSON result. Three methodological commitments governed all of it:

1. **Leakage safety.** One canonical train/val/test split per seed, shared identically by every model; normalization bounds computed from training data only.
2. **Multi-seed estimation.** Every point estimate replaced by a 20-seed distribution, where a seed perturbs **both** sources of randomness — pseudo-trajectory construction *and* model training.
3. **Paired statistics with effect sizes.** Model comparisons use the per-seed paired difference, summarized by paired-bootstrap 95% CIs, Wilcoxon signed-rank tests, Cohen's d_z, and Holm–Bonferroni correction across baselines. Magnitude (d_z) weighted over p-values.

**Hard reproducibility rule:** no number is hand-typed; every figure traces to a JSON in `experiments/results/` produced by a named script.

---

## 3. Technical methodology

### 3.1 Datasets and acquisition

Two independent iPSC directed-differentiation scRNA-seq datasets were used, chosen to span two distinct lineages (neural and cardiac) so that any conclusion could be tested for cross-lineage generalization.

**Dataset A — Neural (dopaminergic).** The dopaminergic-neuron differentiation series used by the original study (`data/raw/dopaminergic_all_timepoints.h5`), profiled at three discrete developmental stages: **Day 11, Day 30, Day 52**. The raw file contains 205,416 cells across two treatment conditions; we retained only untreated control cells (`treatment == 'NONE'`), giving **161,584 cells** (D11 = 50,661; D30 = 50,169; D52 = 60,754). Expression is indexed by HGNC gene symbols.

**Dataset B — Cardiac (GSE175634).** An iPSC→cardiomyocyte differentiation time-course (Elorbany et al., *PLoS Genetics* 2022; GEO accession **GSE175634**), acquired from GEO for this work as a clean cross-lineage replication. The deposited MatrixMarket triplet (`GSE175634_cell_counts.mtx.gz`, gene-index and cell-index tables, and per-cell metadata) was downloaded (≈1 GB counts matrix) and streamed once to extract only the marker-gene counts plus per-cell library totals, avoiding loading the full 230,786 × 38,943 matrix. The dataset contains **230,786 cells** across **7 timepoints** (day 0, 1, 3, 5, 7, 11, 15), 12+ iPSC donor lines, under a single directed protocol (no treatment confound). Cell-type annotation confirms the expected trajectory IPSC → mesoderm (MES/CMES) → progenitor (PROG) → cardiomyocyte (CM). For the three-stage forecasting task we used **day 0 / day 5 / day 11** (early iPSC, mid-transition, committed CM; day0 = 41,330, day5 = 19,508, day11 = 39,342 cells), avoiding the post-day-7 plateau where the differentiation score is near-constant.

### 3.2 Marker-gene panels

The 2-D state **x = (P, D)** is defined by a pluripotency/progenitor panel (P) and a lineage-differentiation panel (D). The exact panels are:

**Dataset A — Neural (dopaminergic):**
- **Pluripotency P (5 genes):** `POU5F1`, `NANOG`, `SOX2`, `UTF1`, `TDGF1`
- **Differentiation D (6 genes, dopaminergic):** `TH`, `DDC`, `SLC6A3`, `DRD2`, `LMX1A`, `FOXA2`

**Dataset B — Cardiac (GSE175634):**
- **Pluripotency P (7 genes):** `POU5F1`, `NANOG`, `SOX2`, `LIN28A`, `DPPA4`, `UTF1`, `TDGF1`
- **Differentiation D (13 genes, cardiac):** `TNNT2`, `MYH6`, `MYH7`, `ACTC1`, `NPPA`, `MYL7`, `TNNI1`, `TNNI3`, `RYR2`, `NKX2-5`, `GATA4`, `TBX5`, `MYL2`

**Expanded panels (used only in the Phase-5 "larger panel" sensitivity variant, Dataset A; all genes confirmed present in the data):**
- **Pluripotency extras (+5 → 10 total):** `LIN28A`, `DPPA4`, `ZFP42`, `PRDM14`, `POU5F1B`
- **Differentiation extras (+8 → 14 total):** `NR4A2`, `PITX3`, `EN1`, `KCNJ6`, `SLC18A2`, `OTX2`, `TUBB3`, `MAP2`

All marker genes in the base panels were present in their respective datasets (no missing markers).

### 3.3 Preprocessing and state construction

Per-cell expression was library-size normalized to 10,000 counts (CP10K) and log-transformed, `log(1 + CP10K)`. For Dataset A this used Scanpy's `normalize_total(target_sum=1e4)` + `log1p`; for Dataset B the identical transform was computed during the streaming pass (per-cell total UMIs for CP10K scaling; marker counts log-transformed). The pluripotency score P and differentiation score D for a cell are the **mean of log(1+CP10K) expression across the genes of the respective panel**.

The two raw scores were then min–max scaled to [0,1] **per feature using training-set bounds only**, with values clipped to [0,1]:
`x_norm = clip((x_raw − min_train) / (max_train − min_train), 0, 1)`.
Computing bounds from training trajectories only (rather than the global population) makes the normalization itself leakage-safe; because all models share the same per-seed bounds, the affine scaling is identical across models and relative comparisons are invariant to it.

Biological validity was confirmed on both datasets: pluripotency declines and lineage-differentiation rises monotonically across stages (Dataset B medians: P 0.86 → 0.15 → 0.00 over day0/day5/day11; D 0.00 → 0.19 → 0.22).

### 3.4 Pseudo-trajectory construction (limitation owned)

For each of **200 pseudo-trajectories** per seed, one cell was sampled (with replacement) from each of the three stage populations and reduced to its (P, D) state, yielding a 3 × 2 array `[[P_t0, D_t0], [P_t1, D_t1], [P_t2, D_t2]]`. **These are not tracked single-cell lineages**: the three cells in a pseudo-trajectory are statistically independent draws from their respective stage distributions. This is the standard construction in the field and is the central limitation we make explicit — there is no genuine within-trajectory temporal dependency.

### 3.5 Train/validation/test split

A single canonical split was used per seed and shared identically by every model: trajectory indices were permuted with the seed's RNG and partitioned **70 / 15 / 15** (140 train / 30 validation / 30 test). This replaces the original pipeline's inconsistent splitting (permuted indices for training, contiguous slice for evaluation), which was the source of the leakage in Phase 0.

### 3.6 Forecasting task

Given the first two stages, predict the third: **`[x_t0, x_t1] → x_t2`** (a 2-timepoint input sequence → 1-timepoint target, each in ℝ²). This is the same task across all datasets and models.

### 3.7 Models

**Ridge regression.** `α = 1.0`; input the flattened `[x_t0, x_t1] ∈ ℝ⁴`, output `x_t2 ∈ ℝ²`. 8 learned parameters.

**Random forest.** Per-seed 3-fold-CV-tuned over depth ∈ {2,3,5,None} and min-samples-leaf ∈ {1,5,10} (300 trees); same input/output. A fixed depth-10 RF overfits this small, near-linear (140-sample) task and is an unfair strawman — CV-tuning lets it regularize to its data-appropriate complexity (it then ties Ridge), matching the fairness standard applied to the ODE.

**Transformer** (827,010 parameters): linear input projection ℝ² → d_model = 128; sinusoidal positional encoding; 4 encoder layers, 8 attention heads, feedforward dim 512, GELU, dropout 0.1; MLP output head (128 → 256 → 2). Training: Adam (lr 1e-3, weight-decay 1e-5), MSE loss, batch size 32, gradient-norm clipping at 1.0, ReduceLROnPlateau (factor 0.5, patience 10), up to 100 epochs with best-model selection on the validation split. The prediction is taken from the final sequence position.

**Mechanistic ODE** in normalized (P, D), integrated over stage time in days:

```
dP/dt = k_ps·(1 − P) − k_pd·P − r·P·D
dD/dt = k_b + k_ds·P·D + r·P·D − k_dd·D
```

with `k_ps = 1.0` and `k_dd = 0.3` fixed, and free parameters `[k_pd, k_ds, r (diff_rate), k_b (k_basal)]` bounded to `[(0,2), (0,2), (0,2), (0,1)]`. Integration used `scipy.integrate.solve_ivp` (RK45, rtol 1e-4, atol 1e-6), outputs clipped to [0,1]. Stage-time coordinates: Dataset A `t = [0, 19, 41]` days (D11→D30→D52); Dataset B `t = [0, 5, 11]` days (day0→day5→day11). Three calibration regimes (Phase 2):
1. **Population-mean** — fit the 4 parameters once (differential evolution) so that integrating from the training-set mean t0 state matches the training mean states at t1 and t2.
2. **Distributional** — fit (least-squares) so the simulated ensemble (propagating a 40-IC subsample of the training t0 distribution) matches both the mean **and** standard deviation of the training population at t1 and t2.
3. **Per-trajectory** — for each test trajectory independently, fit the 4 parameters (least-squares, initialized at the population fit) to reproduce that trajectory's own t0→t1 step, then integrate onward to forecast t2 (the fairest sample-level use).

### 3.8 Evaluation metric

Mean absolute error (MAE) on the predicted late-stage state x_t2, averaged over the two features, reported on the normalized [0,1] scale. Per-feature MAE (P, D) was also logged.

### 3.9 Statistical analysis

Each experiment was run over **20 seeds** (Phases 1–4) or 20 seeds per representation variant (Phase 5). A seed jointly reseeds pseudo-trajectory sampling, the split permutation, and model initialization/training, so it perturbs **both** data construction and model stochasticity. Reported per model: mean ± s.d. and a percentile-bootstrap 95% CI over the seed distribution. Models were compared to the Transformer by the **paired per-seed difference** (same seed ⇒ same trajectory draw ⇒ genuinely paired), summarized by: a paired-bootstrap 95% CI on the mean difference (10,000 resamples), a Wilcoxon signed-rank test, and Cohen's `d_z` on the paired differences. Wilcoxon p-values across the (≥2) baselines were corrected with Holm–Bonferroni. Effect size (`d_z`) was weighted over the p-value given the sample size.

### 3.10 Architectural ablations (Phase 4)

Two controls were applied to the Transformer on the identical task and seeds:
- **Temporal-order shuffle** — the order of the two input timepoints was permuted per sample, using a fixed per-sample permutation applied to **both** training and test, destroying stage order while preserving the values.
- **No positional encoding (no-PE)** — the sinusoidal positional-encoding buffer was zeroed, removing all positional signal.

Each was compared to the ordered baseline by the paired per-seed difference (Wilcoxon). Non-significance is the *positive* result: it demonstrates the temporal machinery is unused.

### 3.11 Representation-sensitivity variants (Phase 5)

Holding the comparison fixed, the (P, D) state construction was varied; for each variant the full Ridge-vs-Transformer comparison was re-run over 20 seeds, with the **same sampled cells reused across variants within a seed** (clean paired design). Variants:
- **baseline** — mean over the base panel (Sec. 3.2).
- **leave-one-out** — drop one marker gene and re-aggregate; one variant per base-panel gene (5 P + 6 D = 11 for Dataset A; 7 P + 13 D = 20 for Dataset B).
- **agg_median** — median (instead of mean) over the panel.
- **agg_zscore** — per-gene z-score (using each gene's global mean/s.d. over all stages), then mean.
- **larger_panel** (Dataset A only) — base + expanded panels (10 pluripotency, 14 differentiation genes; Sec. 3.2).

This yields **15 variants for Dataset A and 23 for Dataset B (38 total)**. The headline metric is the paired Transformer−Ridge gap; robustness is established if no variant produces a significant Transformer advantage.

### 3.12 Software and reproducibility

Python 3.14; NumPy, SciPy, scikit-learn, PyTorch (CPU), Scanpy. Every experiment is a single self-contained script that writes a JSON to `experiments/results/`; no number in this report or the manuscript is hand-entered. The mapping of scripts → results is given in Section 6.

---

## 4. Results — phase by phase

### Phase 0 — Leakage discovery and repair
The original pipeline trained the Transformer on a *permuted* split but evaluated it on a *contiguous* tail slice, ignoring the permutation. Because the permuted training set draws ~70% of indices, **22 of 30 "test" trajectories had been used in training** — while baselines were scored leakage-free. Repaired on the identical frozen data:

| Model | Reported (leaky) | Leakage-free |
|---|---|---|
| Ridge | 0.100 | 0.0843 |
| Random Forest | 0.102 | 0.0954 |
| Transformer | 0.081 | 0.0861 |
| ODE | 0.433 | 0.4147 |
| **Transformer vs. best baseline** | **+19.4%** | **−2.0%** |

The headline gain was entirely a leakage artifact.

### Phase 1 — Multi-seed null (neural)
20 seeds, leakage-free, both randomness sources perturbed:

| Model | MAE (mean ± sd) | Δ vs Transformer | d_z | p (Holm) |
|---|---|---|---|---|
| Ridge | 0.1774 ± 0.0204 | −0.0013 | −0.14 | 0.996 (n.s.) |
| Random Forest (CV-tuned) | 0.1794 ± 0.0208 | +0.0007 | +0.07 | 0.996 (n.s.) |
| Transformer | 0.1787 ± 0.0203 | — | — | — |

**Gate decision: the Transformer edge does not survive.** With a fair CV-tuned RF, all three models are statistically indistinguishable — the Transformer ties 8-parameter ridge regression *and* the random forest. → Pivot thesis confirmed. (An under-regularized depth-10 RF appears ~0.009 worse, but that is an overfitting artifact, not a Transformer advantage; see §3.7.)

### Phase 3 — Cross-lineage generalization (cardiac)
Same protocol on the independent cardiomyocyte dataset:

| Model | MAE (mean ± sd) | Δ vs Transformer | p (Holm) |
|---|---|---|---|
| Ridge | 0.1100 ± 0.0155 | +0.0000 | 1.000 (n.s.) |
| Random Forest (CV-tuned) | 0.1121 ± 0.0175 | +0.0021 | 1.000 (n.s.) |
| Transformer | 0.1100 ± 0.0198 | — | — |

A dead-even tie — Ridge, the CV-tuned random forest, and the Transformer are all indistinguishable. **The null generalizes across lineage** — now a two-dataset, 40-seed result.

### Phase 4 — Why: temporal-structure controls
20 seeds, both datasets. Shuffling stage order or removing positional encoding:

| Dataset | Ordered | Shuffle (Δ, p) | No-PE (Δ, p) |
|---|---|---|---|
| Neural | 0.1787 | −0.0002 (0.87) | −0.0006 (0.50) |
| Cardiac | 0.1100 | −0.0002 (0.90) | −0.0002 (0.99) |

Destroying temporal order changes MAE by ~1–3% of the seed noise, n.s. on both. **The architecture extracts no temporal structure** — because randomly paired pseudo-trajectories contain none.

### Phase 2 — Fair mechanistic ODE
Three calibration regimes on the same test sets:

| Calibration | Neural (vs Ridge) | Cardiac (vs Ridge) |
|---|---|---|
| Population-mean | 0.1984 (+12%) | 0.2483 (+126%) |
| Distributional | 0.2950 (+66%) | 0.2584 (+135%) |
| Per-trajectory | 0.3697 (+108%) | 0.3335 (+203%) |

Monotone on both: **the fairer the sample-level calibration, the worse it gets.** Not a strawman — mechanistic priors add no sample-level predictive value, and per-trajectory calibration fits the noise of independently paired cells.

### Phase 5 — Representation sensitivity
38 variants (every marker dropped one at a time; median / z-score aggregation; larger neural panel), 20 seeds each:

| Dataset | # variants | mean gap (T−Ridge) | gap range | max \|d_z\| | sig. TF wins |
|---|---|---|---|---|---|
| Neural | 15 | +0.0004 | [−0.0016, +0.0017] | 0.19 | 0/15 |
| Cardiac | 23 | +0.0004 | [−0.0009, +0.0028] | 0.19 | 0/23 |

**The null is robust** — it does not hinge on marker panel or aggregation rule.

---

## 5. Synthesis — what it means

The evidence chain is consistent and mutually reinforcing:

> The reported advantage was **leakage** (P0); with leakage removed the Transformer **ties ridge on two lineages** (P1, P3); controls show it **uses no temporal structure** (P4); a **fair mechanistic model does not help** (P2); and the null is **robust to representation** (P5).

The common cause is the pseudo-trajectory construction: pairing **independently sampled** cells across stages destroys within-trajectory temporal dependency, so the conditional p(x_t2 | x_t0, x_t1) is essentially a smooth near-linear map and a memoryless regressor is optimal. This is not an indictment of Transformers but of *evaluating* them on data lacking the structure they model.

**Recommendations** (in the manuscript): report leakage-safe, multi-seed, paired statistics with effect sizes; include a temporal-shuffle control whenever "forecasting" is claimed on pseudo-trajectories; use linear regression as the baseline-of-record; and use lineage-resolved data for genuine temporal-model tests.

---

## 6. Deliverables

| Artifact | Purpose |
|---|---|
| `fair_comparison.py` | Leakage repair / single-seed fair comparison (Table 1) |
| `run_multiseed.py`, `run_multiseed_cardio.py` | Phase 1 / 3 multi-seed nulls |
| `run_ablations.py` | Phase 4 shuffle + no-PE controls |
| `run_ode_fair.py` | Phase 2 fair ODE calibration |
| `run_sensitivity.py` | Phase 5 representation sensitivity |
| `build_cardio_scores.py` | Streamed GSE175634 marker-score extraction |
| `experiments/results/*.json` | All numbers (single source of truth) |
| `paper/manuscript.tex` → `manuscript.pdf` | Reframed 6-page manuscript, compiles cleanly |

---

## 7. Open items

1. **Neural-dataset citation** — repo docs are internally inconsistent (Cuomo 2020 vs 2021 vs likely Jerber 2021); a marked placeholder remains in the manuscript bibliography to confirm.
2. **Figures** — the manuscript is currently table-driven; bar-plots / CI plots could be scripted from the JSONs if desired.
3. **Optional extensions** the plan lists but we deferred: a third dataset, model-size sweep, learned-latent-state comparison.
