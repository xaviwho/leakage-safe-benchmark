# Getting Started in 10 Minutes

## Step 1: Navigate to Project (30 seconds)

```bash
cd "C:\Users\Xavie\Downloads\stem-cell-digital-twin"
```

---

## Step 2: Install Dependencies (5 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install packages (this takes a few minutes)
pip install -r requirements.txt
```

**Note**: If you get errors, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 3: Run Your First Simulation (2 minutes)

```bash
python examples\basic_simulation.py
```

This will:
- ✅ Run 3 differentiation scenarios
- ✅ Create beautiful visualizations
- ✅ Save results to `experiments/results/`
- ✅ Show you the digital twin in action

**Expected Output**:
```
============================================================
iPSC Digital Twin - Basic Simulation Example
============================================================
Configuration loaded
1. Initializing simulator...
2. Scenario 1: Maintaining Pluripotency (High FGF2)
...
Simulation complete!
```

---

## Step 4: View Results (1 minute)

Results are saved in:
```
experiments/results/
├── scenario1_pluripotency.png
├── scenario2_differentiation.png
├── scenario3_sequential.png
├── comparison_all_scenarios.png
└── phase_space_differentiation.png
```

**Open them** to see your simulations!

---

## Step 5: Try Interactive Mode (2 minutes)

```python
# Run Python
python

# Import libraries
from src.models.simulators import iPSCDifferentiationSimulator
from src.models.digital_twin import DigitalTwinEngine
from src.visualization import DigitalTwinPlotter
from src.utils import load_config

# Load config
config = load_config()

# Create simulator
simulator = iPSCDifferentiationSimulator(config)

# Run 14-day simulation
time, states = simulator.run_simulation(
    duration=14,
    timesteps=100,
    growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
)

# Check results
print(f"Final pluripotency: {states[-1, 0]:.3f}")
print(f"Final differentiation: {states[-1, 1]:.3f}")
print(f"Final population: {states[-1, 2]:.0f}")

# Visualize
plotter = DigitalTwinPlotter()
plotter.plot_trajectory(time, states)

import matplotlib.pyplot as plt
plt.show()
```

---

## What Just Happened?

### Scenario 1: High FGF2 (Pluripotency Maintenance)
- Cells maintain pluripotency markers
- Low differentiation
- Good for expansion

### Scenario 2: Retinoic Acid (Differentiation)
- Pluripotency decreases
- Differentiation increases
- Cells commit to lineage

### Scenario 3: Sequential Protocol
- Phase 1: Expand with FGF2 (7 days)
- Phase 2: Differentiate with RA (7 days)
- Mimics real lab protocols

---

## Next Steps

### Option A: Download Real Data (Optional)

```bash
# See what's available
python src\data\download_data.py --info-only

# Download (may take time, files are large)
python src\data\download_data.py

# Process data
python src\data\load_data.py
```

### Option B: Customize Parameters

Edit `config.yaml` to change:
- Simulation duration
- Growth factor concentrations
- Cell cycle rates
- Differentiation thresholds

### Option C: Explore the Code

**Key Files**:
- `src/models/simulators/ipsc_simulator.py` - ODE model
- `src/models/digital_twin/twin_engine.py` - Digital twin
- `src/visualization/plotter.py` - Plotting functions
- `examples/basic_simulation.py` - Full example

---

## Troubleshooting

### Import Errors

```python
import sys
sys.path.insert(0, 'C:/Users/Xavie/Downloads/stem-cell-digital-twin')
```

### Plots Not Showing

```python
import matplotlib.pyplot as plt
plt.show()  # Add this after plotting
```

### Package Installation Fails

```bash
# Install packages one by one
pip install numpy scipy pandas
pip install matplotlib seaborn
pip install torch scikit-learn
pip install scanpy anndata
```

### Need Help?

1. Check `README.md` for full documentation
2. See `QUICKSTART.md` for detailed guide
3. Look at `examples/` for more examples
4. Read `PROJECT_SUMMARY.md` for overview

---

## Understanding the Output

### State Variables

**Pluripotency (P)**: 0-1 scale
- 1.0 = Fully pluripotent (iPSC)
- 0.0 = Fully differentiated

**Differentiation (D)**: 0-1 scale
- 0.0 = Undifferentiated
- 1.0 = Fully differentiated

**Population (N)**: Cell count
- Grows over time
- Affected by division and death rates

### Growth Factors

**FGF2**: Maintains pluripotency
- High FGF2 → High P, Low D
- Cells stay as iPSCs

**Retinoic Acid (RA)**: Induces differentiation
- High RA → Low P, High D
- Cells become neurons

---

## Quick Reference

### Run Simulation
```python
time, states = simulator.run_simulation(
    duration=14,           # days
    timesteps=100,         # data points
    growth_factors={       # concentrations
        'fgf2': 1.0,
        'retinoic_acid': 0.0
    }
)
```

### Create Digital Twin
```python
twin = DigitalTwinEngine(simulator, config=config)
twin.initialize()
state = twin.update(duration=24)  # 24 hours
prediction = twin.predict(horizon=48)  # 48 hours ahead
```

### Visualize
```python
plotter = DigitalTwinPlotter()
plotter.plot_trajectory(time, states)
plotter.plot_phase_space(states)
```

---

## Success! 🎉

You now have:
- ✅ A working digital twin system
- ✅ Simulation results
- ✅ Visualization tools
- ✅ Framework for ML integration

**Ready for the next step?** See `PROJECT_SUMMARY.md` for the full roadmap!

---

**Project**: Hybrid Physics-ML Digital Twin for Predicting Stem Cell Differentiation Dynamics
**Author**: Xavier Kanu
**Conference**: ICUFN 2026
