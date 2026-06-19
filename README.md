# Hybrid Physics-ML Digital Twin for Predicting Stem Cell Differentiation Dynamics

A computational framework that integrates mechanistic ODE models with machine learning to predict induced pluripotent stem cell (iPSC) differentiation dynamics in real-time.

## 🎯 Project Overview

This project implements a **pure software digital twin** that:
- Simulates iPSC differentiation dynamics using mechanistic models
- Predicts differentiation outcomes using ML on single-cell RNA-seq data
- Enables virtual experimentation and protocol optimization
- Provides real-time state tracking and prediction

**Target Conference:** ICUFN (International Conference on Ubiquitous and Future Networks)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Digital Twin Engine                   │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Physical   │  │  Biological  │  │   ML-Based   │  │
│  │  Simulator   │→ │    Models    │→ │  Predictors  │  │
│  │  (ODE/SDE)   │  │    (GRN)     │  │ (LSTM/Trans) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         ↓                  ↓                  ↓          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         State Estimation & Prediction            │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
              ┌────────────────────────┐
              │  Visualization Layer   │
              │  (Interactive Dashboard)│
              └────────────────────────┘
```

## 📁 Project Structure

```
stem-cell-digital-twin/
├── data/
│   ├── raw/              # Original scRNA-seq datasets
│   ├── processed/        # Preprocessed data
│   └── simulated/        # Synthetic data from simulators
├── models/
│   ├── simulators/       # ODE/SDE cell models
│   ├── digital_twin/     # Core digital twin engine
│   ├── predictors/       # ML prediction models
│   └── grn/              # Gene regulatory networks
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model implementations
│   ├── visualization/    # Dashboard and plotting
│   └── utils/            # Helper functions
├── experiments/
│   ├── configs/          # Experiment configurations
│   └── results/          # Experiment outputs
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository (or navigate to project directory)
cd stem-cell-digital-twin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.digital_twin import DigitalTwin
from src.models.simulators import iPSCDifferentiationSimulator

# Initialize simulator
simulator = iPSCDifferentiationSimulator()

# Create digital twin
twin = DigitalTwin(simulator)

# Run simulation
results = twin.run_simulation(duration=14, timesteps=100)

# Visualize
twin.plot_trajectory()
```

### Run Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

## 🔬 Key Features

### 1. **Mechanistic Simulator**
- ODE-based population dynamics
- Stochastic single-cell behavior
- Gene regulatory network modeling
- Culture condition effects (nutrients, growth factors)

### 2. **Data-Driven Models**
- Train on public scRNA-seq datasets
- Trajectory inference algorithms
- Cell fate prediction
- RNA velocity integration

### 3. **Digital Twin Engine**
- Real-time state estimation
- Predictive forecasting (24-48h ahead)
- Uncertainty quantification
- Adaptive protocol recommendations

### 4. **ML Predictors**
- LSTM/Transformer for time-series
- Graph Neural Networks for cell interactions
- Variational Autoencoders for state space
- Reinforcement Learning for optimization

### 5. **Visualization**
- Interactive dashboard (Streamlit)
- Real-time simulation playback
- 3D cell state space (UMAP/t-SNE)
- Comparative analysis tools

## 📊 Datasets

We use publicly available single-cell RNA-seq datasets:
- **CellxGene**: Curated stem cell datasets
- **GEO**: Differentiation time-series
- **Human Cell Atlas**: Developmental trajectories

See `data/README.md` for dataset details and download instructions.

## 🧪 Experiments

Run predefined experiments:

```bash
# Basic differentiation simulation
python experiments/run_simulation.py --config configs/basic_diff.yaml

# ML model training
python experiments/train_predictor.py --config configs/lstm_config.yaml

# Virtual protocol optimization
python experiments/optimize_protocol.py
```

## 📈 Results

Results will be saved in `experiments/results/` including:
- Simulation trajectories
- Prediction accuracy metrics
- Optimized protocols
- Visualizations

## 🛠️ Technologies

- **Python 3.9+**
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Machine Learning**: PyTorch, Scikit-learn
- **Single-Cell Analysis**: Scanpy, AnnData, scVelo
- **Visualization**: Matplotlib, Plotly, Streamlit
- **Modeling**: PyDSTool (ODEs), Gillespie (stochastic)

## 📝 Citation

If you use this work, please cite:

```bibtex
@inproceedings{kanu2026digitaltwin,
  title={Hybrid Physics-ML Digital Twin for Predicting Stem Cell Differentiation Dynamics},
  author={Kanu, Xavier},
  booktitle={International Conference on Ubiquitous and Future Networks (ICUFN)},
  year={2026}
}
```

## 🤝 Contributing

Contributions welcome! Please see `CONTRIBUTING.md` for guidelines.

## 📄 License

MIT License - see `LICENSE` file for details.

## 📧 Contact

Xavier Kanu - [@kanuxvi](https://twitter.com/kanuxvi)

Project Link: [https://github.com/xaviwho/stem-cell-digital-twin](https://github.com/xaviwho/stem-cell-digital-twin)

---

**Status:** 🚧 Under Active Development

**Last Updated:** February 2026
