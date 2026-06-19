# Dataset Information

## Jerber et al. (2021) - Dopaminergic Neuron Differentiation

### Publication Details

**Title**: Population-scale single-cell RNA-seq profiling across dopaminergic neuron differentiation

**Journal**: Nature Genetics (2021)

**DOI**: 10.1038/s41588-021-00801-6

**Authors**: Julie Jerber, Daniel D. Seaton, et al.

### Dataset Description

This dataset contains single-cell RNA-sequencing data from a large-scale study of iPSC differentiation towards dopaminergic neurons, a cell type relevant for Parkinson's disease research.

**Key Features:**
- **215 iPSC lines** from different donors
- **3 differentiation timepoints**: Day 0 (iPSC), Day 11 (neural progenitors), Day 30 (dopaminergic neurons)
- **>1 million cells** profiled across all conditions
- **Single-cell resolution** capturing heterogeneity and dynamics
- **Multiplexed design** enabling efficient large-scale profiling

### Data Access

#### 1. Processed Data (Recommended)
**Zenodo**: https://zenodo.org/record/4333872

Files available:
- Count matrices for each timepoint (.h5ad format)
- Cell metadata
- Gene annotations
- eQTL summary statistics

#### 2. Raw Sequencing Data
**European Nucleotide Archive (ENA)**: ERP121676
- Raw FASTQ files
- Requires alignment and processing

#### 3. Analysis Code
**GitHub**: https://github.com/single-cell-genetics/singlecell_neuroseq_paper
- Data processing scripts
- Analysis notebooks
- Figure generation code

### Downloading Data

#### Automatic Download (Recommended)

```bash
# Download processed data
python src/data/download_data.py

# Show download info only
python src/data/download_data.py --info-only
```

#### Manual Download

1. Visit: https://zenodo.org/record/4333872
2. Download the following files:
   - `day0_counts.h5ad` (iPSCs)
   - `day11_counts.h5ad` (Neural progenitors)
   - `day30_counts.h5ad` (Dopaminergic neurons)
3. Place files in `data/raw/` directory

### Processing Data

After downloading:

```bash
# Process and combine timepoints
python src/data/load_data.py

# Or in Python:
from src.data import DopaminergicDataLoader

loader = DopaminergicDataLoader()
loader.load_all_timepoints()
adata = loader.combine_timepoints()
adata = loader.preprocess()
adata = loader.calculate_scores(adata)
loader.save_processed(adata)
```

### Data Structure

**Timepoints:**
- **Day 0**: Pluripotent iPSCs (high OCT4, NANOG, SOX2)
- **Day 11**: Neural progenitors (neural markers emerging)
- **Day 30**: Dopaminergic neurons (TH, DDC, SLC6A3 expression)

**Key Cell Markers:**

| Cell Type | Marker Genes |
|-----------|--------------|
| iPSC (Pluripotent) | POU5F1 (OCT4), NANOG, SOX2, UTF1 |
| Neural Progenitor | SOX1, PAX6, NESTIN |
| Dopaminergic Neuron | TH, DDC, SLC6A3, LMX1A, FOXA2 |
| Pan-neuronal | MAP2, TUBB3, DCX, NCAM1 |

### Why This Dataset?

1. **Large scale**: 215 cell lines provide population-level variation
2. **Temporal resolution**: 3 timepoints capture differentiation dynamics
3. **Well-characterized**: Published in high-impact journal with thorough validation
4. **Relevant biology**: Dopaminergic neurons are clinically important (Parkinson's)
5. **Accessible**: Data freely available with clear documentation
6. **Quality**: High-quality scRNA-seq with good coverage

### Dataset Statistics

```
Day 0 (iPSC):           ~300,000 cells
Day 11 (Progenitor):    ~350,000 cells
Day 30 (DA Neuron):     ~400,000 cells
Total:                  >1,000,000 cells

Genes:                  ~25,000 (before filtering)
Cell lines:             215 donors
```

### Citation

If you use this dataset, please cite:

```bibtex
@article{jerber2021population,
  title={Population-scale single-cell RNA-seq profiling across dopaminergic neuron differentiation},
  author={Jerber, Julie and Seaton, Daniel D and Cuomo, Anna SE and others},
  journal={Nature Genetics},
  volume={53},
  number={3},
  pages={304--312},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### Data License

The data is released under CC-BY 4.0 license, allowing:
- Free use for research and commercial purposes
- Redistribution and modification
- Requires attribution to original authors

### Related Datasets

Other iPSC differentiation datasets from the same group:
- **GSE118723**: iPSC heterogeneity study
- **HCA Portal**: Human Cell Atlas developmental datasets

### Support & Questions

- **GitHub Issues**: https://github.com/single-cell-genetics/singlecell_neuroseq_paper/issues
- **Paper Contact**: Corresponding authors listed in publication
- **Zenodo**: Comments on Zenodo record

### File Formats

**AnnData (.h5ad)**:
- Standard format for single-cell data
- Readable by scanpy, scvi-tools, cellxgene
- Contains:
  - Count matrix (`.X`)
  - Cell metadata (`.obs`)
  - Gene metadata (`.var`)
  - Embeddings (`.obsm`)

**CSV files**:
- Cell metadata: Cell IDs, timepoints, QC metrics
- Gene annotations: Gene IDs, names, biotypes

---

**Last Updated**: February 2026

**Next Steps**:
1. Download data: `python src/data/download_data.py`
2. Process data: `python src/data/load_data.py`
3. Train models: See `notebooks/` for examples
