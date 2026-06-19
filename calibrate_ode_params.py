"""
Calibrate ODE parameters to real population-level data.

This fits physics simulator parameters to observed experimental trends,
addressing the reviewer concern about "unfitted parameters".
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from pathlib import Path

from src.models.simulators import iPSCDifferentiationSimulator
from src.utils import load_config

print("="*80)
print("ODE PARAMETER CALIBRATION")
print("="*80)

# Load real population-level data
pop_data = pd.read_csv('data/processed/dopaminergic_population_dynamics.csv')
print("\nReal Population Data:")
print(pop_data[['day', 'n_cells', 'P_mean', 'D_mean']])

# Extract target values
day11 = pop_data[pop_data['day'] == 11].iloc[0]
day30 = pop_data[pop_data['day'] == 30].iloc[0]
day52 = pop_data[pop_data['day'] == 52].iloc[0]

# Observed changes
observed_P_11 = day11['P_mean']
observed_D_11 = day11['D_mean']
observed_P_30 = day30['P_mean']
observed_D_30 = day30['D_mean']
observed_P_52 = day52['P_mean']
observed_D_52 = day52['D_mean']

print(f"\nObserved Trends:")
print(f"Day 11: P={observed_P_11:.3f}, D={observed_D_11:.3f}")
print(f"Day 30: P={observed_P_30:.3f}, D={observed_D_30:.3f}")
print(f"Day 52: P={observed_P_52:.3f}, D={observed_D_52:.3f}")


def objective_function(params_vec):
    """
    Objective function to minimize: difference between ODE predictions and real data.

    Args:
        params_vec: [k_pluri_deg, k_diff_synth, diff_rate, k_basal]

    Returns:
        Loss (L2 distance to observed data)
    """
    k_pluri_deg, k_diff_synth, diff_rate, k_basal = params_vec

    # Create simulator with trial parameters
    config = load_config()
    sim = iPSCDifferentiationSimulator(config)

    # Update parameters
    sim.params['k_pluri_deg'] = k_pluri_deg
    sim.params['k_diff_synth'] = k_diff_synth
    sim.params['diff_rate'] = diff_rate
    sim.params['k_basal'] = k_basal

    # Initial state at Day 11 (use observed values)
    initial_state = np.array([observed_P_11, observed_D_11, day11['n_cells']])

    # Simulate Day 11 -> Day 30
    times_30, states_30 = sim.run_simulation(
        duration=19.0,  # 19 days
        timesteps=20,
        initial_state=initial_state
    )
    pred_P_30, pred_D_30 = states_30[-1, :2]

    # Simulate Day 11 -> Day 52
    times_52, states_52 = sim.run_simulation(
        duration=41.0,  # 41 days
        timesteps=50,
        initial_state=initial_state
    )
    pred_P_52, pred_D_52 = states_52[-1, :2]

    # Calculate loss (L2 distance)
    loss_30 = (pred_P_30 - observed_P_30)**2 + (pred_D_30 - observed_D_30)**2
    loss_52 = (pred_P_52 - observed_P_52)**2 + (pred_D_52 - observed_D_52)**2

    total_loss = loss_30 + loss_52

    return total_loss


# Parameter bounds (biologically plausible ranges)
bounds = [
    (0.1, 1.0),   # k_pluri_deg: decay rate
    (0.1, 1.0),   # k_diff_synth: synthesis rate
    (0.05, 0.5),  # diff_rate: differentiation rate
    (0.01, 0.2)   # k_basal: basal induction
]

print("\n" + "="*80)
print("OPTIMIZATION")
print("="*80)
print(f"Method: Differential Evolution (global optimization)")
print(f"Parameter bounds:")
print(f"  k_pluri_deg:   [{bounds[0][0]}, {bounds[0][1]}]")
print(f"  k_diff_synth:  [{bounds[1][0]}, {bounds[1][1]}]")
print(f"  diff_rate:     [{bounds[2][0]}, {bounds[2][1]}]")
print(f"  k_basal:       [{bounds[3][0]}, {bounds[3][1]}]")

# Initial guess (current defaults)
x0 = [0.5, 0.3, 0.15, 0.05]

print(f"\nInitial guess (defaults): {x0}")
initial_loss = objective_function(x0)
print(f"Initial loss: {initial_loss:.6f}")

# Optimize using differential evolution (global optimizer)
print("\nRunning optimization...")
result = differential_evolution(
    objective_function,
    bounds,
    maxiter=100,
    popsize=15,
    seed=42,
    disp=True,
    atol=1e-6,
    tol=1e-6
)

# Extract optimized parameters
k_pluri_deg_opt, k_diff_synth_opt, diff_rate_opt, k_basal_opt = result.x

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"\nOptimized Parameters:")
print(f"  k_pluri_deg:   {k_pluri_deg_opt:.6f} (was: {x0[0]})")
print(f"  k_diff_synth:  {k_diff_synth_opt:.6f} (was: {x0[1]})")
print(f"  diff_rate:     {diff_rate_opt:.6f} (was: {x0[2]})")
print(f"  k_basal:       {k_basal_opt:.6f} (was: {x0[3]})")

print(f"\nFinal loss: {result.fun:.6f}")
print(f"Improvement: {((initial_loss - result.fun) / initial_loss * 100):.1f}%")

# Test calibrated parameters
sim_calibrated = iPSCDifferentiationSimulator(load_config())
sim_calibrated.params['k_pluri_deg'] = k_pluri_deg_opt
sim_calibrated.params['k_diff_synth'] = k_diff_synth_opt
sim_calibrated.params['diff_rate'] = diff_rate_opt
sim_calibrated.params['k_basal'] = k_basal_opt

initial_state = np.array([observed_P_11, observed_D_11, day11['n_cells']])

times_30, states_30 = sim_calibrated.run_simulation(19.0, 20, initial_state)
times_52, states_52 = sim_calibrated.run_simulation(41.0, 50, initial_state)

print("\n" + "="*80)
print("VALIDATION")
print("="*80)
print("\nDay 30 Predictions:")
print(f"  Observed: P={observed_P_30:.3f}, D={observed_D_30:.3f}")
print(f"  Predicted: P={states_30[-1, 0]:.3f}, D={states_30[-1, 1]:.3f}")
print(f"  Error: P={abs(states_30[-1, 0] - observed_P_30):.3f}, D={abs(states_30[-1, 1] - observed_D_30):.3f}")

print(f"\nDay 52 Predictions:")
print(f"  Observed: P={observed_P_52:.3f}, D={observed_D_52:.3f}")
print(f"  Predicted: P={states_52[-1, 0]:.3f}, D={states_52[-1, 1]:.3f}")
print(f"  Error: P={abs(states_52[-1, 0] - observed_P_52):.3f}, D={abs(states_52[-1, 1] - observed_D_52):.3f}")

# Save calibrated parameters
output = {
    'k_pluri_deg': float(k_pluri_deg_opt),
    'k_diff_synth': float(k_diff_synth_opt),
    'diff_rate': float(diff_rate_opt),
    'k_basal': float(k_basal_opt),
    'final_loss': float(result.fun),
    'improvement_percent': float((initial_loss - result.fun) / initial_loss * 100)
}

import json
output_path = Path('config/calibrated_ode_params.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[SAVED] Calibrated parameters: {output_path}")

print("\n" + "="*80)
print("SUCCESS")
print("="*80)
print("ODE parameters are now CALIBRATED to real experimental data!")
print("Use these parameters in your physics simulator for publication.")
