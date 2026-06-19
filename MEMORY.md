# Stem Cell Digital Twin - Key Learnings

## Data Processing
- Real trajectory data needs proper train/val split from same distribution
- Never mix real data (training) with synthetic ODE data (validation)
- Feature scales must match: day numbers vs population counts caused 117B validation loss
- With only 3 timepoints per trajectory, use sequence_length=2, prediction_horizon=1

## Model Performance on Dopaminergic Data
- **Transformer > LSTM**: 38% lower validation loss (0.209 vs 0.338)
- Transformer better at differentiation prediction (MAE 0.311 vs 0.378)
- 200 trajectories split 80/20 gives 160 train, 40 validation samples

## Best ML Approaches for Stem Cell Trajectories
1. **Neural ODEs** - Combines physics equations with neural networks (state-of-the-art)
2. **Transformers** - Best general-purpose sequence model
3. **Graph Neural Networks** - For cell-cell interaction modeling
4. **LSTM** - Good baseline but outperformed by above

## Digital Twin Architecture
- **Hybrid = Physics (ODE) + ML (Transformer/NeuralODE)**
- ODE simulator: Known biological mechanisms
- ML model: Learn from real data patterns
- Combine: ODE provides structure, ML corrects residuals

## For ICUFN 2026 Paper
- Dataset: Cuomo et al. (2021) dopaminergic differentiation, 205K cells
- Timepoints: Day 11, 30, 52
- All 11 marker genes present (5 pluripotency + 6 differentiation)
- Transformer achieves ~4% error on differentiation prediction
