# CellxGene Real Data Integration Guide

## Overview

You can now train your ML models on **real stem cell datasets** from CellxGene instead of synthetic data! 🎉

---

## 🚀 Quick Start

### Step 1: Find a Dataset on CellxGene

Visit: **https://cellxgene.cziscience.com/datasets**

Search for:
- `stem cell differentiation`
- `iPSC`
- `pluripotent`
- `cardiomyocyte` or `neuron` (specific lineages)

**Look for datasets with**:
- ✓ Time-series data (Day 0, Day 3, Day 7, etc.)
- ✓ Multiple developmental stages
- ✓ Clear differentiation progression

### Step 2: Get the Download URL

1. Click on a dataset
2. Find the **"Download"** button
3. Copy the direct download URL (ends with `.h5ad`)

### Step 3: Download and Process

Run our helper script:

```bash
python scripts/prepare_cellxgene_data.py --url <PASTE_URL_HERE>
```

**Example**:
```bash
python scripts/prepare_cellxgene_data.py --url "https://datasets.cellxgene.cziscience.com/abc123/dataset.h5ad"
```

This will:
1. Download the dataset (~100MB-5GB)
2. Load and analyze it
3. Extract pluripotency/differentiation markers
4. Generate trajectories
5. Save processed data to `data/processed/cellxgene_trajectories.pkl`

### Step 4: Train on Real Data

```bash
python experiments/train_predictor.py --load_data data/processed/cellxgene_trajectories.pkl --model lstm --epochs 100
```

---

## 🔍 Example: Finding iPSC→Neuron Dataset

### Search on CellxGene:

1. Go to https://cellxgene.cziscience.com/datasets
2. Search: **"iPSC differentiation neuron"**
3. Look for datasets with:
   - Multiple timepoints
   - Clear progression from iPSC to neurons
   - Good cell coverage (>10,000 cells)

### Example Datasets to Try:

**Recommended searches**:
- "iPSC cardiomyocyte differentiation"
- "neural differentiation time series"
- "organoid development"
- "stem cell lineage"

---

## 📊 What the Loader Does

### 1. Downloads Dataset
```python
loader = StemCellDataLoader()
filepath = loader.download_from_cellxgene(url)
```

### 2. Loads h5ad File
```python
adata = loader.load_h5ad(filepath)
# Shows: number of cells, genes, metadata
```

### 3. Extracts Marker Genes

**Pluripotency markers** (automatically detected):
- POU5F1 (OCT4)
- SOX2
- NANOG
- KLF4
- MYC

**Differentiation markers**:
- Mesoderm: T, MIXL1, EOMES
- Ectoderm: SOX1, PAX6, NES
- Endoderm: SOX17, FOXA2, GATA6

### 4. Computes Cell State Scores

For each cell:
- **Pluripotency score**: Mean expression of pluripotency markers
- **Differentiation score**: Mean expression of differentiation markers
- **Population**: Cell count proxy

### 5. Extracts Trajectories

Creates pseudobulk trajectories by:
- Grouping cells by timepoint
- Averaging states at each timepoint
- Generating multiple replicates with sampling

---

## 🛠️ Advanced Usage

### Custom Marker Genes

```python
from src.data.cellxgene_loader import StemCellDataLoader

loader = StemCellDataLoader()
adata = loader.load_h5ad('data/raw/dataset.h5ad')

# Use custom markers
marker_df = loader.extract_stem_cell_markers(
    adata,
    pluripotency_genes=['POU5F1', 'SOX2', 'NANOG'],
    differentiation_genes=['MAP2', 'TUBB3', 'SYP']  # Neuron-specific
)
```

### Extract More Trajectories

```bash
python scripts/prepare_cellxgene_data.py \
    --url <URL> \
    --n-trajectories 500 \
    --time-column "day"
```

### Use Different Time Column

If the dataset uses "day", "age", or "development_stage":

```bash
python scripts/prepare_cellxgene_data.py \
    --url <URL> \
    --time-column "day"
```

---

## 📝 Python API

### Full Control in Python:

```python
from src.data.cellxgene_loader import StemCellDataLoader

# 1. Create loader
loader = StemCellDataLoader(data_dir='data/raw')

# 2. Download dataset
filepath = loader.download_from_cellxgene(
    dataset_url='YOUR_CELLXGENE_URL_HERE'
)

# 3. Load data
adata = loader.load_h5ad(filepath)

# 4. Extract trajectories
trajectories = loader.extract_trajectories(
    adata,
    time_column='timepoint',  # or 'day', 'age', etc.
    n_trajectories=200
)

# 5. Save for training
loader.save_processed_data(
    trajectories,
    'data/processed/my_real_data.pkl'
)

# 6. Train models
from src.models.predictors.trainer import CellStateDataset
dataset = CellStateDataset(trajectories, sequence_length=20, prediction_horizon=10)
# ... proceed with training
```

---

## 🎯 Recommended Datasets

### Good Starting Points:

1. **iPSC → Cardiomyocyte**
   - Search: "cardiomyocyte differentiation"
   - Look for: TNNT2, MYH6, NKX2-5 markers

2. **iPSC → Neuron**
   - Search: "neural differentiation"
   - Look for: MAP2, TUBB3, SYP markers

3. **Organoid Development**
   - Search: "organoid time series"
   - Look for: multi-day progression

4. **Early Development**
   - Search: "embryonic development"
   - Look for: gastrulation, lineage specification

---

## 🔧 Troubleshooting

### "No time column found"
**Solution**: The loader will automatically compute pseudotime using diffusion maps.

Or specify a different column:
```bash
--time-column "development_stage"
```

### "No pluripotency markers found"
**Solution**: The dataset might use different gene names or not include stem cells.
- Check what genes are available: `list(adata.var_names)`
- Use PCA components as fallback (automatic)

### "Download failed"
**Solutions**:
1. Check the URL is correct (must end with .h5ad)
2. Check your internet connection
3. Try downloading manually from browser, then use:
   ```python
   adata = loader.load_h5ad('path/to/manually/downloaded/file.h5ad')
   ```

### "Out of memory"
**Solution**: Process smaller batches:
```bash
--n-trajectories 50  # Reduce from default 200
```

---

## 📈 Training with Real Data

### Compare Synthetic vs Real Data:

**Train on synthetic**:
```bash
python experiments/train_predictor.py --model lstm --n_train 1000
```

**Train on real CellxGene data**:
```bash
python experiments/train_predictor.py --model lstm --load_data data/processed/cellxgene_trajectories.pkl
```

**Compare results** - real data should give:
- Better generalization
- More realistic dynamics
- Validation against actual biology

---

## 🎓 Publication Impact

Using real CellxGene data:
- ✅ Validates your model on actual experimental data
- ✅ Shows generalization beyond synthetic simulations
- ✅ Enables comparison with published datasets
- ✅ Strengthens publication claims

**For your ICUFN paper**, include:
1. Training on real scRNA-seq data
2. Validation metrics (accuracy on held-out cells)
3. Comparison: synthetic-trained vs real-trained models
4. Case study: predicting differentiation outcomes

---

## 📚 Resources

- **CellxGene**: https://cellxgene.cziscience.com
- **Scanpy Documentation**: https://scanpy.readthedocs.io
- **AnnData Format**: https://anndata.readthedocs.io

---

## ✅ Checklist

- [ ] Found relevant dataset on CellxGene
- [ ] Downloaded and processed with prepare_cellxgene_data.py
- [ ] Verified trajectories look reasonable
- [ ] Trained LSTM/Transformer on real data
- [ ] Compared with synthetic-trained model
- [ ] Generated figures for paper

---

**Now you can train on REAL stem cell data! 🎉**

This completes your hybrid Physics-ML digital twin with real data integration!
