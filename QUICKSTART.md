# Quick Start Guide

Get up and running with the iPSC Digital Twin in 5 minutes!

## Installation

### 1. Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment tool

### 2. Clone/Download Project

```bash
cd C:\Users\Xavie\Downloads\stem-cell-digital-twin
```

### 3. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Note**: Installation may take 5-10 minutes depending on your connection.

## Run Your First Simulation

### Option 1: Basic Example Script

```bash
python examples/basic_simulation.py
```

This will:
- Run 3 different differentiation scenarios
- Generate visualizations
- Save results to `experiments/results/`
- Display plots

### Option 2: Interactive Python

```python
import sys
sys.path.insert(0, 'C:/Users/Xavie/Downloads/stem-cell-digital-twin')

from src.models.simulators import iPSCDifferentiationSimulator
from src.visualization import DigitalTwinPlotter
from src.utils import load_config

# Load config
config = load_config()

# Create simulator
simulator = iPSCDifferentiationSimulator(config)

# Run simulation (14 days, maintaining pluripotency)
time, states = simulator.run_simulation(
    duration=14,
    timesteps=100,
    growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
)

# Check final state
final_state = simulator.get_current_state()
print(f"Pluripotency: {final_state['pluripotency']:.3f}")
print(f"Differentiation: {final_state['differentiation']:.3f}")
print(f"Population: {final_state['population']:.0f}")

# Visualize
plotter = DigitalTwinPlotter()
plotter.plot_trajectory(time, states)

import matplotlib.pyplot as plt
plt.show()
```

### Option 3: Digital Twin Demo

```python
from src.models.simulators import iPSCDifferentiationSimulator
from src.models.digital_twin import DigitalTwinEngine
from src.utils import load_config

# Setup
config = load_config()
simulator = iPSCDifferentiationSimulator(config)
twin = DigitalTwinEngine(simulator, config=config)

# Initialize
twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

# Update over 3 days
for day in range(1, 4):
    state = twin.update(duration=24)  # 24 hours
    print(f"Day {day}: P={state['pluripotency']:.3f}, "
          f"D={state['differentiation']:.3f}")

    # Get recommendations
    recs = twin.recommend_action()
    print(f"  → {recs['actions'][0]['message']}")

# Predict next 48 hours
prediction = twin.predict(
    horizon=48,
    growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
)

final_pred = prediction['predicted_states'][-1]
print(f"\nPredicted (48h): P={final_pred['pluripotency']:.3f}, "
      f"D={final_pred['differentiation']:.3f}")
```

## Understanding the Results

### State Variables

- **Pluripotency (P)**: 0-1 scale, measures expression of OCT4/NANOG/SOX2
  - > 0.8: High pluripotency, suitable for expansion
  - 0.4-0.8: Intermediate state
  - < 0.4: Loss of pluripotency

- **Differentiation (D)**: 0-1 scale, measures lineage-specific markers
  - > 0.7: Well-differentiated cells
  - 0.3-0.7: Differentiating
  - < 0.3: Undifferentiated

- **Population (N)**: Total cell count

### Growth Factors

- **FGF2**: Maintains pluripotency (higher = more pluripotent)
- **Retinoic Acid**: Induces differentiation (higher = more differentiation)

## Next Steps

### 1. Explore Configurations

Edit `config.yaml` to customize:
- Simulation parameters
- Model parameters
- Visualization settings

### 2. Try Different Protocols

```python
# Cardiomyocyte differentiation
time, states = simulator.run_simulation(
    duration=14,
    growth_factors={'fgf2': 0.1, 'activin_a': 1.0, 'bmp4': 0.8}
)

# Neural differentiation
time, states = simulator.run_simulation(
    duration=14,
    growth_factors={'fgf2': 0.5, 'retinoic_acid': 0.7}
)
```

### 3. Explore Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Notebooks cover:
- Data exploration
- Model training
- Advanced visualizations
- Custom experiments

### 4. Run Interactive Dashboard (Coming Soon)

```bash
streamlit run src/visualization/dashboard.py
```

### 5. Work with Real Data

See `data/README.md` for:
- Downloading public datasets
- Preprocessing pipelines
- Training ML models on real data

## Common Issues

### Import Errors

If you get import errors:

```python
import sys
sys.path.insert(0, '/path/to/stem-cell-digital-twin')
```

Or install as package:
```bash
pip install -e .
```

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Plots Not Showing

```python
import matplotlib.pyplot as plt
plt.show()  # Add this after plotting
```

## Project Structure

```
stem-cell-digital-twin/
├── src/
│   ├── models/
│   │   ├── simulators/       # ODE-based cell models
│   │   ├── digital_twin/     # Digital twin engine
│   │   └── predictors/       # ML models (coming soon)
│   ├── visualization/        # Plotting utilities
│   ├── data/                 # Data loaders
│   └── utils/                # Helper functions
├── examples/                 # Example scripts
├── notebooks/                # Jupyter notebooks
├── data/                     # Datasets
├── experiments/              # Results
└── config.yaml              # Configuration
```

## Getting Help

- **Documentation**: See `docs/` folder (coming soon)
- **Examples**: Check `examples/` directory
- **Issues**: Report bugs on GitHub
- **Configuration**: See `config.yaml` with inline comments

## What's Next?

Now that you have the basics working, you can:

1. **Customize models**: Modify ODE parameters in simulator
2. **Add ML models**: Implement LSTM/Transformer predictors
3. **Use real data**: Download and train on scRNA-seq data
4. **Build dashboard**: Create interactive visualization
5. **Optimize protocols**: Use RL to find optimal conditions
6. **Publish**: Prepare for ICUFN conference submission

## Quick Reference

### Run Simulation
```python
simulator.run_simulation(duration=14, timesteps=100, growth_factors={...})
```

### Initialize Twin
```python
twin.initialize(growth_factors={...})
```

### Update Twin
```python
twin.update(duration=24, growth_factors={...})
```

### Make Prediction
```python
twin.predict(horizon=48, growth_factors={...})
```

### Visualize
```python
plotter.plot_trajectory(time, states)
plotter.plot_phase_space(states)
```

---

**You're all set! Start experimenting with the digital twin! 🚀**
