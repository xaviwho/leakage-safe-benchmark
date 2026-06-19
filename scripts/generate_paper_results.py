"""
Generate results for ICUFN 2026 paper.

Creates all figures and metrics needed for:
"Hybrid Physics-ML Digital Twin for Predicting Stem Cell Differentiation Dynamics"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.models.simulators import iPSCDifferentiationSimulator, StochasticiPSCSimulator
from src.models.digital_twin import DigitalTwinEngine
from src.models.predictors import LSTMPredictor, TransformerPredictor
from src.visualization import DigitalTwinPlotter
from src.utils import load_config, setup_logger

logger = setup_logger("paper_results", level="INFO")


def generate_all_figures(output_dir: Path):
    """Generate all figures for paper."""

    config = load_config()
    plotter = DigitalTwinPlotter()

    logger.info("="*80)
    logger.info("GENERATING ICUFN 2026 PAPER RESULTS")
    logger.info("="*80)

    # Figure 1: System Architecture (diagram - manual)
    logger.info("\n📊 Figure 1: System architecture diagram")
    logger.info("   → Create manually with architecture overview")

    # Figure 2: ODE Simulation Results
    logger.info("\n📊 Figure 2: ODE simulation of differentiation protocols")

    simulator = iPSCDifferentiationSimulator(config)

    # Three scenarios
    scenarios = [
        ("Pluripotency Maintenance", {'fgf2': 1.0, 'retinoic_acid': 0.0}),
        ("Differentiation Induction", {'fgf2': 0.0, 'retinoic_acid': 1.0}),
        ("Sequential Protocol", None)  # Special handling
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (name, gf) in enumerate(scenarios[:2]):
        simulator.reset()
        time, states = simulator.run_simulation(
            duration=14, timesteps=100, growth_factors=gf
        )

        axes[0, i].plot(time, states[:, 0], 'b-', linewidth=2)
        axes[0, i].set_title(f"{name}\nPluripotency")
        axes[0, i].set_xlabel("Time (days)")
        axes[0, i].set_ylabel("Pluripotency Score")
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(time, states[:, 1], 'r-', linewidth=2)
        axes[1, i].set_title(f"Differentiation")
        axes[1, i].set_xlabel("Time (days)")
        axes[1, i].set_ylabel("Differentiation Score")
        axes[1, i].grid(True, alpha=0.3)

    # Sequential protocol
    simulator.reset()
    time1, states1 = simulator.run_simulation(
        duration=7, timesteps=50, growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
    )
    time2, states2 = simulator.add_perturbation(
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}, duration=7, timesteps=50
    )

    time_seq = np.concatenate([time1, time1[-1] + time2])
    states_seq = np.concatenate([states1, states2])

    axes[0, 2].plot(time_seq, states_seq[:, 0], 'b-', linewidth=2)
    axes[0, 2].axvline(7, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].set_title("Sequential Protocol\nPluripotency")
    axes[0, 2].set_xlabel("Time (days)")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 2].plot(time_seq, states_seq[:, 1], 'r-', linewidth=2)
    axes[1, 2].axvline(7, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].set_title("Differentiation")
    axes[1, 2].set_xlabel("Time (days)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Figure 2: ODE-Based Simulation of Differentiation Protocols",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_ode_simulation.png", dpi=300, bbox_inches='tight')
    logger.info(f"   ✓ Saved: figure2_ode_simulation.png")

    # Figure 3: Stochastic Simulation with Uncertainty
    logger.info("\n📊 Figure 3: Stochastic simulation with uncertainty quantification")

    stochastic_sim = StochasticiPSCSimulator(config)
    time, ensemble_states = stochastic_sim.run_simulation(
        duration=14, timesteps=100,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0},
        n_realizations=50
    )

    stats = stochastic_sim.get_ensemble_statistics(ensemble_states)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pluripotency
    axes[0].plot(time, stats['mean'][:, 0], 'b-', linewidth=2, label='Mean')
    axes[0].fill_between(time, stats['q05'][:, 0], stats['q95'][:, 0],
                         alpha=0.3, color='blue', label='90% CI')
    axes[0].fill_between(time, stats['q25'][:, 0], stats['q75'][:, 0],
                         alpha=0.5, color='blue', label='50% CI')
    axes[0].set_xlabel("Time (days)", fontsize=12)
    axes[0].set_ylabel("Pluripotency Score", fontsize=12)
    axes[0].set_title("Pluripotency with Uncertainty", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Differentiation
    axes[1].plot(time, stats['mean'][:, 1], 'r-', linewidth=2, label='Mean')
    axes[1].fill_between(time, stats['q05'][:, 1], stats['q95'][:, 1],
                         alpha=0.3, color='red', label='90% CI')
    axes[1].fill_between(time, stats['q25'][:, 1], stats['q75'][:, 1],
                         alpha=0.5, color='red', label='50% CI')
    axes[1].set_xlabel("Time (days)", fontsize=12)
    axes[1].set_ylabel("Differentiation Score", fontsize=12)
    axes[1].set_title("Differentiation with Uncertainty", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Figure 3: Stochastic Simulation with Uncertainty Quantification (n=50)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_uncertainty.png", dpi=300, bbox_inches='tight')
    logger.info(f"   ✓ Saved: figure3_uncertainty.png")

    # Figure 4: Phase space trajectory
    logger.info("\n📊 Figure 4: Phase space analysis")
    plotter.plot_phase_space(
        states2,
        title="Figure 4: Phase Space Trajectory (Pluripotency vs Differentiation)",
        save_path=output_dir / "figure4_phase_space.png"
    )
    logger.info(f"   ✓ Saved: figure4_phase_space.png")

    # Figure 5 & 6: ML results (need trained models)
    logger.info("\n📊 Figures 5-6: ML model results")
    logger.info("   → Train models first with train_predictor.py")
    logger.info("   → Then run hybrid_ml_demo.py for ML figures")

    plt.close('all')


def generate_results_table():
    """Generate results table for paper."""

    logger.info("\n📊 Table 1: Model performance comparison")

    # Template - fill in after training
    results = {
        'Model': ['Physics-only (ODE)', 'LSTM', 'Transformer', 'Hybrid (50-50)'],
        'MSE': ['N/A', 'TBD', 'TBD', 'TBD'],
        'MAE': ['N/A', 'TBD', 'TBD', 'TBD'],
        'R²': ['N/A', 'TBD', 'TBD', 'TBD'],
        'Parameters': ['15', 'TBD', 'TBD', 'TBD']
    }

    df = pd.DataFrame(results)

    print("\nTable 1: Model Performance on Cortical Neuron Differentiation")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    print("\n(TBD = To Be Determined after training)")

    # Save
    output_path = Path("experiments/results/paper_results/table1_performance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"   ✓ Saved: {output_path}")


def main():
    """Generate all paper results."""

    output_dir = Path("experiments/results/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir}")

    # Generate figures
    generate_all_figures(output_dir)

    # Generate tables
    generate_results_table()

    logger.info("\n" + "="*80)
    logger.info("✅ PAPER RESULTS GENERATED")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nGenerated:")
    logger.info("  • Figure 2: ODE simulation results")
    logger.info("  • Figure 3: Uncertainty quantification")
    logger.info("  • Figure 4: Phase space analysis")
    logger.info("  • Table 1: Performance comparison (template)")
    logger.info("\nNext steps:")
    logger.info("  1. Train LSTM and Transformer models")
    logger.info("  2. Run hybrid_ml_demo.py for ML figures")
    logger.info("  3. Update Table 1 with actual metrics")
    logger.info("="*80)


if __name__ == "__main__":
    main()
