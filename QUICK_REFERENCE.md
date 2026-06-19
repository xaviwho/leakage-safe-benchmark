# Quick Reference: Motor Neuron Project

## 🎯 Your Project: iPSC → Motor Neuron Digital Twin

---

## 🚀 **3-Step Workflow**

### **STEP 1: Find Dataset (5 minutes)**
```
1. Go to: https://cellxgene.cziscience.com/datasets
2. Search: "motor neuron differentiation"
3. Pick dataset with 3+ timepoints
4. Download .h5ad file → save to data/raw/
```

### **STEP 2: Process Dataset (2-5 minutes)**
```bash
python scripts/prepare_motor_neuron_data.py --file data/raw/YOUR_FILE.h5ad
```

### **STEP 3: Train Models (30-60 minutes)**
```bash
# LSTM
python experiments/train_predictor.py \
    --load_data data/processed/motor_neuron_trajectories.pkl \
    --model lstm --epochs 100

# Transformer
python experiments/train_predictor.py \
    --load_data data/processed/motor_neuron_trajectories.pkl \
    --model transformer --epochs 100
```

---

## 🔍 **CellxGene Search Terms**

**Best searches** (in order of priority):
1. `motor neuron differentiation`
2. `spinal cord organoid`
3. `MNX1 OLIG2 ISL1` (marker genes)
4. `ALS iPSC control`
5. `Wichterle motor neuron`

**What you're looking for**:
- ✓ Multiple timepoints (Day 0, 7, 14, 21, 28)
- ✓ >10,000 cells
- ✓ "Differentiation" or "development" in title
- ✓ Contains iPSC and motor neurons

---

## 📊 **Motor Neuron Markers**

### **Stage 1: iPSC (Day 0)**
POU5F1, SOX2, NANOG → **High pluripotency**

### **Stage 2: Neural Progenitor (Day 3-7)**
PAX6, SOX1, NES → **Becoming neural**

### **Stage 3: MN Progenitor (Day 7-14)**
OLIG2, NKX6-1 → **Motor neuron fate**

### **Stage 4: Mature Motor Neuron (Day 14-35)**
ISL1, MNX1, CHAT → **Functional motor neurons**

---

## 📁 **File Locations**

```
data/raw/                           # Downloaded datasets (.h5ad)
data/processed/motor_neuron_trajectories.pkl  # Processed for training
experiments/results/                # Trained models & results
```

---

## 🎓 **For Your Paper**

### **What You'll Show**:
1. ✅ Real motor neuron differentiation data from CellxGene
2. ✅ LSTM and Transformer trained on real trajectories
3. ✅ Hybrid physics-ML predictions
4. ✅ Validation on held-out cells
5. ✅ Comparison with synthetic data baseline

### **Key Results**:
- Training accuracy on real data
- Prediction accuracy vs physics-only
- Hybrid model performance
- Biological validation (marker progression)

---

## 💻 **Quick Commands**

### **Check Dataset**
```python
from src.data.cellxgene_loader import StemCellDataLoader

loader = StemCellDataLoader()
adata = loader.load_h5ad('data/raw/dataset.h5ad')
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
print(f"Timepoints: {adata.obs['day'].unique()}")
```

### **Process Data**
```bash
python scripts/prepare_motor_neuron_data.py --file data/raw/dataset.h5ad
```

### **Train LSTM**
```bash
python experiments/train_predictor.py \
    --load_data data/processed/motor_neuron_trajectories.pkl \
    --model lstm --epochs 100
```

### **Run Complete Demo**
```bash
python examples/hybrid_ml_demo.py
```

### **Validate System**
```bash
python tests/test_system.py
```

---

## 📚 **Documentation**

- **[MOTOR_NEURON_DATASET_GUIDE.md](MOTOR_NEURON_DATASET_GUIDE.md)** - Complete guide
- **[CELLXGENE_GUIDE.md](CELLXGENE_GUIDE.md)** - General CellxGene usage
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - System overview
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was built

---

## ✅ **Your Checklist**

### **Today**:
- [ ] Find motor neuron dataset on CellxGene
- [ ] Download .h5ad file
- [ ] Run `prepare_motor_neuron_data.py`
- [ ] Verify processing succeeded

### **This Week**:
- [ ] Train LSTM model (100 epochs)
- [ ] Train Transformer model (100 epochs)
- [ ] Compare performance
- [ ] Generate figures

### **For Paper**:
- [ ] Train on full dataset (200+ trajectories)
- [ ] Run validation experiments
- [ ] Compare with synthetic baseline
- [ ] Write results section
- [ ] Generate publication figures

---

## 🆘 **Need Help?**

### **Common Issues**:

**"Can't find dataset"**
→ See [MOTOR_NEURON_DATASET_GUIDE.md](MOTOR_NEURON_DATASET_GUIDE.md)

**"Processing failed"**
→ Check time column name with `--time-column`

**"Training too slow"**
→ Reduce `--batch_size` or `--epochs`

**"Out of memory"**
→ Reduce `--n-trajectories` to 50-100

---

## 🎯 **Current Status**

✅ **Complete**:
- ODE simulator
- ML models (LSTM, Transformer)
- Training pipeline
- Motor neuron data loader
- Hybrid digital twin
- All examples and tests

🔄 **Next Step**:
- Download motor neuron dataset from CellxGene
- Process and train on real data
- Generate results for paper

---

**You're ready to go! Start by finding a dataset on CellxGene.** 🚀

**Next command to run**:
```bash
# After downloading dataset
python scripts/prepare_motor_neuron_data.py --file data/raw/YOUR_DATASET.h5ad
```
