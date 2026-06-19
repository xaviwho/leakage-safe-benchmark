# Data Directory

This directory contains datasets for training and validating the digital twin models.

## Directory Structure

```
data/
├── raw/          # Original, unprocessed datasets
├── processed/    # Preprocessed and cleaned datasets
└── simulated/    # Synthetic data from simulators
```

## Recommended Datasets

### 1. Single-Cell RNA-seq Datasets

#### CellxGene (Curated Stem Cell Data)
- **Source**: https://cellxgene.cziscience.com/
- **Search for**: "iPSC differentiation", "pluripotent stem cells"
- **File format**: .h5ad (AnnData)
- **Usage**: Training trajectory prediction models

**Recommended collections:**
- Human iPSC differentiation time-series
- Embryonic stem cell developmental trajectories
- Differentiation to specific lineages (cardiac, neural, hepatic)

#### GEO (Gene Expression Omnibus)
- **Source**: https://www.ncbi.nlm.nih.gov/geo/
- **Search terms**:
  - "iPSC differentiation scRNA-seq"
  - "pluripotent stem cell time course"
  - "directed differentiation single cell"

**Example datasets:**
- GSE75748 - Human iPSC to definitive endoderm
- GSE122662 - iPSC cardiac differentiation
- GSE134355 - Neural differentiation time-course

#### Human Cell Atlas
- **Source**: https://www.humancellatlas.org/
- **Focus**: Developmental biology datasets
- **File format**: Various (.h5ad, .loom, .h5)

### 2. Bulk RNA-seq Datasets

For validating population-level predictions:
- **ArrayExpress**: https://www.ebi.ac.uk/arrayexpress/
- **Search**: "stem cell differentiation time series"

### 3. Proteomic Data

For multi-omics integration:
- **PRIDE Archive**: https://www.ebi.ac.uk/pride/
- **Search**: "stem cell differentiation proteomics"

## Data Download Instructions

### Option 1: Manual Download

1. Visit the data source (e.g., CellxGene, GEO)
2. Search for relevant datasets
3. Download to `data/raw/`
4. Run preprocessing scripts in `notebooks/`

### Option 2: Programmatic Download

```python
# Example: Download from CellxGene
import cellxgene_census
import anndata

# Load dataset
adata = cellxgene_census.download_dataset("dataset_id")

# Save to raw directory
adata.write("data/raw/dataset_name.h5ad")
```

### Option 3: Use Simulation Data

For initial development and testing, use the simulator to generate synthetic data:

```python
from src.models.simulators import iPSCDifferentiationSimulator
import numpy as np

# Generate training data
simulator = iPSCDifferentiationSimulator()

# Multiple runs with different parameters
datasets = []
for i in range(100):
    # Randomize parameters
    growth_factors = {
        'fgf2': np.random.uniform(0, 1),
        'retinoic_acid': np.random.uniform(0, 1)
    }

    time, states = simulator.run_simulation(
        duration=14,
        timesteps=100,
        growth_factors=growth_factors
    )

    datasets.append({'time': time, 'states': states, 'gf': growth_factors})

# Save
import pickle
with open('data/simulated/training_data.pkl', 'wb') as f:
    pickle.dump(datasets, f)
```

## Data Preprocessing

Once downloaded, preprocess data using:

```bash
# From project root
python src/data/preprocess.py --input data/raw/ --output data/processed/
```

Or use the provided Jupyter notebooks:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`

## Data Format

### Expected Format for Training

Processed data should be in AnnData format (.h5ad) with:

**Observations (cells):**
- `obs['timepoint']`: Time point in hours/days
- `obs['condition']`: Culture condition
- `obs['cell_type']`: Cell type annotation (if available)
- `obs['pluripotency_score']`: Computed pluripotency score
- `obs['differentiation_score']`: Computed differentiation score

**Variables (genes):**
- Gene expression matrix in `adata.X`
- Normalized and log-transformed

**Metadata:**
- `adata.uns['protocol']`: Differentiation protocol details
- `adata.uns['growth_factors']`: Growth factor concentrations

## Example Data Loading

```python
import anndata
import scanpy as sc

# Load preprocessed data
adata = anndata.read_h5ad('data/processed/ipsc_diff_timecourse.h5ad')

# Quick exploration
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
print(f"Timepoints: {adata.obs['timepoint'].unique()}")

# Visualize
sc.pl.umap(adata, color=['timepoint', 'pluripotency_score'])
```

## Citation

When using public datasets, please cite the original publications. Citations can be found:
- In the dataset metadata
- On the source database webpage
- In the `data/citations.txt` file

## Data Privacy & Ethics

- Only use publicly available datasets or datasets with proper permissions
- Follow data use agreements and licenses
- For clinical data, ensure proper de-identification and IRB approval
- Do not include any private or sensitive data in this repository

## Need Help?

See `notebooks/00_data_guide.ipynb` for detailed tutorials on:
- Finding relevant datasets
- Downloading and loading data
- Preprocessing pipelines
- Quality control
- Data integration

---

**Last Updated**: February 2026
