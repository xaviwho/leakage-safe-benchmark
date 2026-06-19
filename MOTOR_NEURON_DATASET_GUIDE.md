# Motor Neuron Differentiation Dataset Guide

## 🎯 Your Choice: iPSC → Motor Neuron Differentiation

Perfect choice! This is one of the best-characterized differentiation systems with:
- Clear developmental trajectory
- Strong marker genes at each stage
- Clinical relevance (ALS, SMA research)
- Multiple available datasets

---

## 🔍 Step-by-Step: Finding Your Dataset

### **Step 1: Go to CellxGene**
Visit: **https://cellxgene.cziscience.com/datasets**

### **Step 2: Search Terms (Try in Order)**

**Search 1**: `motor neuron differentiation`
- Most direct search
- Should return 5-20 datasets

**Search 2**: `spinal cord organoid`
- Organoids include motor neuron differentiation
- Often has time-series

**Search 3**: `MNX1 OLIG2 ISL1`
- Motor neuron marker genes
- Finds datasets with these markers

**Search 4**: `ALS iPSC`
- ALS studies always have control iPSC→MN differentiation
- High quality datasets

**Search 5**: `Wichterle` or `Jessell` or `Novitch`
- Pioneer labs in motor neuron biology
- Gold standard datasets

### **Step 3: What to Look For**

**Perfect Dataset Has**:
```
✓ Title: Contains "motor neuron" or "spinal cord"
✓ Cells: >10,000 (ideally 20,000+)
✓ Timepoints: Minimum 3 (Day 0, 14, 28), ideally 5+ (Day 0, 3, 7, 14, 21, 28)
✓ Description: Mentions "differentiation protocol" or "time course"
✓ Year: 2019 or later (more recent = better QC)
✓ Journal: Nature, Science, Cell, Neuron (higher quality)
```

**Red Flags to Avoid**:
```
✗ Only mature neurons (no iPSC starting point)
✗ Single timepoint
✗ Disease-only (want controls)
✗ <5,000 cells (might be noisy)
```

---

## 🎓 Specific Dataset Recommendations

### **Top Papers to Find on CellxGene:**

#### **1. Wichterle Lab Studies** ⭐⭐⭐⭐⭐
- **Search**: "Wichterle motor neuron"
- **Expected timepoints**: Day 0, 7, 14, 28
- **Quality**: Excellent - pioneers of MN differentiation

#### **2. Spinal Cord Organoid Studies** ⭐⭐⭐⭐⭐
- **Search**: "spinal cord organoid development"
- **Expected timepoints**: Multiple (weeks)
- **Quality**: Very good - recent method

#### **3. ALS Control Datasets** ⭐⭐⭐⭐
- **Search**: "ALS iPSC" then filter for "control"
- **Expected**: Control lines with time-series
- **Quality**: Good - well-validated protocols

#### **4. Protocol Paper Datasets** ⭐⭐⭐⭐⭐
- **Search**: "motor neuron protocol" or "method"
- **Expected**: Very detailed time course
- **Quality**: Excellent - optimized for teaching/reproduction

---

## 📊 Motor Neuron Differentiation Stages

Your dataset should show this progression:

### **Day 0: iPSC (Pluripotent)**
**Markers**: POU5F1/OCT4, SOX2, NANOG
- High pluripotency score
- No differentiation markers

### **Day 3-7: Neural Induction**
**Markers**: PAX6, SOX1, NES
- Losing pluripotency
- Becoming neural progenitors

### **Day 7-14: Motor Neuron Progenitors**
**Markers**: OLIG2, NKX6-1
- Neural identity established
- Ventral spinal cord specification

### **Day 14-21: Immature Motor Neurons**
**Markers**: ISL1, ISL2, MNX1 (HB9)
- Motor neuron identity
- Still maturing

### **Day 21-35: Mature Motor Neurons**
**Markers**: CHAT, SLC18A3, PRPH, TUBB3, MAP2
- Functional motor neurons
- Express acetylcholine machinery

---

## 💻 Once You Find a Dataset

### **Download It**:
1. Click on the dataset
2. Click "Download" button
3. Choose "Download .h5ad file"
4. Save to `data/raw/motor_neuron_dataset.h5ad`

### **Process It**:
```bash
# Run our specialized motor neuron processor
python scripts/prepare_motor_neuron_data.py --file data/raw/motor_neuron_dataset.h5ad
```

**What this does**:
- ✅ Loads the h5ad file
- ✅ Checks for motor neuron markers
- ✅ Extracts pluripotency→differentiation trajectories
- ✅ Generates 200 pseudobulk trajectories
- ✅ Saves to `data/processed/motor_neuron_trajectories.pkl`

### **Train Models**:
```bash
# Train LSTM
python experiments/train_predictor.py \
    --load_data data/processed/motor_neuron_trajectories.pkl \
    --model lstm \
    --epochs 100

# Train Transformer
python experiments/train_predictor.py \
    --load_data data/processed/motor_neuron_trajectories.pkl \
    --model transformer \
    --epochs 100
```

---

## 🎯 Example: Step-by-Step Walkthrough

### **Scenario: You Found a Dataset**

**Example Dataset**:
- Title: "Single-cell analysis of human iPSC-derived motor neuron differentiation"
- Cells: 45,000
- Timepoints: Day 0, 7, 14, 21, 28
- Size: 850 MB

**What to Do**:

1. **Download**:
   ```bash
   # Download from CellxGene (click Download button)
   # Save as: data/raw/motor_neuron_45k.h5ad
   ```

2. **Quick Check**:
   ```python
   from src.data.cellxgene_loader import StemCellDataLoader

   loader = StemCellDataLoader()
   adata = loader.load_h5ad('data/raw/motor_neuron_45k.h5ad')

   print(f"Cells: {adata.n_obs:,}")
   print(f"Genes: {adata.n_vars:,}")
   print(f"Metadata: {list(adata.obs.columns)}")

   # Find time column
   if 'day' in adata.obs.columns:
       print(f"Timepoints: {sorted(adata.obs['day'].unique())}")
   ```

3. **Process**:
   ```bash
   python scripts/prepare_motor_neuron_data.py \
       --file data/raw/motor_neuron_45k.h5ad \
       --time-column day \
       --n-trajectories 200
   ```

4. **Train**:
   ```bash
   python experiments/train_predictor.py \
       --load_data data/processed/motor_neuron_trajectories.pkl \
       --model lstm \
       --epochs 100 \
       --batch_size 32
   ```

5. **Validate**:
   ```bash
   python examples/hybrid_ml_demo.py
   ```

---

## 🔬 Motor Neuron-Specific Markers

Our processor automatically looks for these genes:

### **Pluripotency** (Should decrease):
- POU5F1, SOX2, NANOG, LIN28A, DPPA4

### **Neural Progenitor** (Increases Days 3-7):
- PAX6, SOX1, NES, HES5

### **Motor Neuron Progenitor** (Increases Days 7-14):
- OLIG2, NKX6-1, DBX1, NKX2-2

### **Mature Motor Neuron** (Increases Days 14-35):
- ISL1, ISL2, MNX1, CHAT, SLC18A3, PRPH, TUBB3, MAP2

---

## 📈 Expected Results

### **What Good Data Looks Like**:

**Pluripotency Score**:
- Day 0: 0.9-1.0 (high)
- Day 7: 0.5-0.7 (decreasing)
- Day 14: 0.2-0.4 (low)
- Day 28: 0.0-0.2 (very low)

**Differentiation Score**:
- Day 0: 0.0-0.1 (low)
- Day 7: 0.2-0.4 (increasing)
- Day 14: 0.5-0.7 (high)
- Day 28: 0.8-1.0 (very high)

### **Training Performance**:
- **Synthetic data**: MSE ~0.01-0.02
- **Real motor neuron data**: MSE ~0.02-0.05 (more variability)
- **Hybrid model**: Best of both worlds

---

## 🎓 For Your ICUFN Paper

### **What to Show**:

1. **Dataset Description**:
   - "We obtained iPSC→motor neuron scRNA-seq data from CellxGene"
   - "Dataset contains X cells across Y timepoints (Day 0-28)"
   - Cite original paper

2. **Model Training**:
   - "Trained on 200 differentiation trajectories"
   - "LSTM achieved X accuracy, Transformer achieved Y"
   - Compare with physics-only baseline

3. **Validation**:
   - "Predicted differentiation outcomes on held-out cells"
   - "Hybrid physics-ML outperformed pure ML by Z%"
   - Show prediction accuracy over time

4. **Biological Validation**:
   - "Model correctly predicts marker gene expression"
   - "Captures known biology (OLIG2→ISL1 transition)"
   - Show correlation with known differentiation stages

---

## ✅ Checklist

- [ ] Searched CellxGene for motor neuron datasets
- [ ] Found dataset with 3+ timepoints
- [ ] Downloaded .h5ad file
- [ ] Ran `prepare_motor_neuron_data.py`
- [ ] Verified trajectories look reasonable
- [ ] Trained LSTM model
- [ ] Trained Transformer model
- [ ] Compared with synthetic data baseline
- [ ] Generated figures for paper

---

## 🆘 Troubleshooting

### **"Can't find any good datasets"**
→ Try broader searches: "spinal", "ventral", "neural"
→ Look at ALS study controls
→ Check recent papers (2021-2024)

### **"Dataset has no timepoints"**
→ Use pseudotime (automatic fallback)
→ Or find a different dataset

### **"Markers not found"**
→ Genes might have different names
→ Check `adata.var_names` for alternatives
→ Example: MNX1 = HB9, POU5F1 = OCT4

### **"Too much memory"**
→ Reduce `--n-trajectories` to 50
→ Process in smaller batches

---

**Ready to start? Go find your motor neuron dataset on CellxGene! 🚀**

Next: Come back with the dataset name/file and I'll help you process it!
