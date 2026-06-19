# A Leakage-Safe Benchmark for Single-Cell Differentiation Forecasting

A leakage-safe, multi-seed, equivalence-tested benchmark for **discrete-stage single-cell
differentiation forecasting**. The repository accompanies the manuscript *"A Leakage-Safe
Benchmark for Single-Cell Differentiation Forecasting"* (Kanu et al., under review at
*BMC Bioinformatics*) and contains everything needed to regenerate every figure and table
from per-seed result files.

## What this study found

Expressive temporal models (Transformers, neural ODEs, physics-informed networks) are
increasingly used to "forecast" cell-differentiation state from cross-sectional scRNA-seq.
Because no cell is observed at more than one timepoint, such forecasts are evaluated on
**pseudo-trajectories** that concatenate *independently sampled* cells under a ground-truth
stage order. Under a single leakage-safe protocol applied to two independent lineages
(dopaminergic neuron and cardiomyocyte) over 20 seeds each:

- A Transformer is **statistically equivalent** to an eight-parameter linear (Ridge) model
  and to a complexity-matched random forest, on both lineages (paired bootstrap CI within an
  equivalence margin of |Δ| < 0.01).
- A previously apparent **+19.4 % advantage was an artifact of train–test leakage**; it
  inverts to −2 % once the split is corrected (Table 3).
- **Shuffling stage order or removing positional encoding leaves accuracy unchanged** — the
  sequence model extracts no temporal structure (Fig. 5).
- **Fairer ODE calibration monotonically worsens** prediction (Fig. 6).

One mechanism explains all of it: pseudo-trajectories built from independently sampled cells
contain no within-trajectory temporal dependency, so a memoryless predictor is optimal and
added model capacity fits noise. The lasting contribution is the **evaluation protocol**, not
any single model.

## Repository map

Each script reads/writes only `experiments/results/*.json`; **every reported number is
generated programmatically — nothing is hand-entered.**

| Script | Produces | Paper artifact |
|---|---|---|
| `fair_comparison.py` | `results/fair_comparison_seed42.json` | **Table 3** — leakage demonstration & repair |
| `run_multiseed.py` | `results/multiseed_n20.json` | **Fig. 3/4** — neural equivalence (20 seeds) |
| `run_multiseed_cardio.py` | `results/multiseed_cardio_n20.json` | **Fig. 3/4** — cardiac replication |
| `run_ablations.py` | `results/ablations_n20.json` | **Fig. 5** — stage-order shuffle & no-PE controls |
| `run_ode_fair.py` | `results/ode_fair_n20.json` | **Fig. 6** — three-regime ODE calibration |
| `run_sensitivity.py` | `results/sensitivity_{dopaminergic,cardio}_n20.json` | **Fig. 7 / S2** — 38-variant representation sweep |
| `build_cardio_scores.py` | `data/processed/cardio_cell_scores.csv.gz`, `cardio_marker_log.npz` | cardiac marker-score extraction from GSE175634 |
| `paper/figures/make_figures.py` | `paper/figures/*.{png,pdf}` | **all figures**, regenerated from `results/` |

Model implementations (Transformer, trainer, ODE simulator) live in `src/`; the manuscript and
generated figures are in `paper/`.

## Data availability

The two datasets are **public and referenced, not redistributed** (raw counts are ~26 GB and
are git-ignored). See [`data/README.md`](data/README.md) for exact accessions and download steps.

- **Neural (dopaminergic):** Jerber et al. 2021, *Nature Genetics* — population-scale
  iPSC→dopaminergic-neuron differentiation. The pipeline reads `data/raw/dopaminergic_all_timepoints.h5`.
- **Cardiac (cardiomyocyte):** GEO accession **GSE175634** (Elorbany et al. 2022, *PLoS Genetics*).
  Raw counts are reduced to marker scores by `build_cardio_scores.py`.

The small **derived per-seed result files** (`experiments/results/*.json`) **are committed** —
so you can regenerate every figure without re-running the full pipeline or downloading raw data.

## Installation

CPU-only; no GPU required.

```bash
pip install -r requirements.txt
```

Core stack: Python 3.14, PyTorch (CPU), Scanpy, SciPy, scikit-learn, NumPy.

## Reproduce the results

**Fastest path — regenerate all figures from the committed results (no raw data needed):**

```bash
python paper/figures/make_figures.py    # writes paper/figures/*.{png,pdf}
```

**Full path — regenerate the per-seed metrics, then the figures** (requires the raw datasets,
see `data/README.md`):

```bash
# 1. (cardiac only) build marker scores from raw GSE175634 counts
python build_cardio_scores.py

# 2. regenerate per-seed metrics (writes experiments/results/*.json)
python run_multiseed.py --seeds 20
python run_multiseed_cardio.py --seeds 20
python run_ablations.py --seeds 20
python run_ode_fair.py --seeds 20
python run_sensitivity.py --dataset dopaminergic --seeds 20
python run_sensitivity.py --dataset cardio --seeds 20
python fair_comparison.py --seed 42        # Table 3 leakage demonstration

# 3. regenerate every figure from results/
python paper/figures/make_figures.py
```

Each script accepts `--seeds` (default 20, except `run_sensitivity.py` which defaults to 10) and
`--epochs`. A quick smoke run uses `--seeds 2` (matching the committed `*_n2.json` files).

### Expected runtime & hardware

CPU-only on a laptop. Figure regeneration from committed JSON takes **seconds**. The full
multi-seed pipeline (small Transformer, ~827K params; 200 pseudo-trajectories per seed) runs in
well under an hour.

## Evaluation protocol (in one paragraph)

For each seed: 200 stage-ordered pseudo-trajectories are assembled by sampling one cell
independently per stage; a single 70/15/15 trajectory-level split is drawn and shared identically
by every model; min–max normalization bounds are estimated on the **training partition only**.
Each baseline is compared to the Transformer by the per-seed paired difference Δ = MAE_baseline −
MAE_Transformer, summarized with a percentile paired-bootstrap 95 % CI (10⁴ resamples), a Wilcoxon
signed-rank test, Cohen's dₓ, and Holm–Bonferroni correction. A CI bracketing zero inside the
pre-specified margin |Δ| < 0.01 is read as **statistical equivalence**, not mere non-significance.

## License & citation

Released under the **MIT License** (see [`LICENSE`](LICENSE)).

If you use this benchmark, please cite the manuscript (Kanu et al., *A Leakage-Safe Benchmark for
Single-Cell Differentiation Forecasting*, under review at *BMC Bioinformatics*, 2026). Machine-readable
citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button); the volume/DOI will be filled in once assigned.
