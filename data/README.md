# Data

The two datasets analysed in this study are **public** and are **referenced, not redistributed**.
Raw single-cell counts are large (~26 GB total) and are **git-ignored** — obtain them from the
accessions below. The small derived inputs the pipeline actually consumes are kept under
`data/processed/`.

```
data/
├── raw/         # public raw counts — NOT committed (git-ignored); download per below
└── processed/   # small derived inputs consumed by the pipeline (committed)
```

## 1. Neural (dopaminergic) lineage

- **Source:** Jerber et al., *Population-scale single-cell RNA-seq profiling across dopaminergic
  neuron differentiation*, **Nature Genetics** 53(3):304–312 (2021).
- **Accession:** ArrayExpress / INSDC — `PRJEB38269` / `ERP121676` *(confirm against the
  publication before release)*.
- **Stages used:** day 11, day 30, day 52; untreated controls only (`treatment = NONE`).
- **Pipeline input:** `data/raw/dopaminergic_all_timepoints.h5`, read directly by
  `run_multiseed.py`, `run_ablations.py`, `run_ode_fair.py`, and `run_sensitivity.py`.

## 2. Cardiac (cardiomyocyte) lineage

- **Source:** Elorbany et al., *Single-cell sequencing reveals lineage-specific dynamic genetic
  regulation of gene expression during human cardiomyocyte differentiation*, **PLoS Genetics**
  18(1):e1009666 (2022).
- **Accession:** GEO **GSE175634** —
  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175634>
- **Download** the supplementary files into `data/raw/cardio_gse175634/`:
  `GSE175634_cell_counts.mtx.gz`, `GSE175634_cell_metadata.tsv.gz`,
  `GSE175634_cell_indices.tsv.gz`, `GSE175634_gene_indices_counts.tsv.gz`.
- **Build the marker-score inputs** (a single streaming pass over the counts matrix):

  ```bash
  python build_cardio_scores.py
  ```

  This writes `data/processed/cardio_cell_scores.csv.gz` and `data/processed/cardio_marker_log.npz`,
  consumed by `run_multiseed_cardio.py`, `run_ablations.py`, `run_ode_fair.py`, and
  `run_sensitivity.py --dataset cardio`.

## State construction

For each cell, raw counts are library-size normalized to 10,000 (CP10K) and log-transformed;
two state scores are the per-panel means of the log-normalized expression of a pluripotency panel
(P) and a lineage-differentiation panel (D). Each score is min–max scaled to [0, 1] using bounds
estimated **on the training partition only** (leakage-safe), then clipped. Marker panels and the
full procedure are given in §2.1–2.2 of the manuscript.

## Reproducing figures without raw data

The committed per-seed result files in `experiments/results/*.json` let you regenerate every
figure without downloading any raw data:

```bash
python paper/figures/make_figures.py
```

No new sequencing data were generated in this study. When using these datasets, cite the original
publications above.
