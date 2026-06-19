"""
Basic simulation example demonstrating the iPSC Digital Twin.

This script shows how to:
1. Initialize the simulator
2. Run a differentiation simulation
3. Visualize results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.models.simulators import iPSCDifferentiationSimulator
from src.models.digital_twin import DigitalTwinEngine
from src.visualization import DigitalTwinPlotter
from src.utils import load_config, setup_logger

# Setup logging
logger = setup_logger("basic_simulation", level="INFO")


def main():
    """Run basic simulation example."""
    logger.info("=" * 60)
    logger.info("iPSC Digital Twin - Basic Simulation Example")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")

    # Initialize simulator
    logger.info("\n1. Initializing simulator...")
    simulator = iPSCDifferentiationSimulator(config)

    # Initialize plotter
    plotter = DigitalTwinPlotter()

    # =========================================================================
    # Scenario 1: Maintain Pluripotency
    # =========================================================================
    logger.info("\n2. Scenario 1: Maintaining Pluripotency (High FGF2)")
    logger.info("-" * 60)

    time1, states1 = simulator.run_simulation(
        duration=14,
        timesteps=100,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
    )

    logger.info(f"Final state - P: {states1[-1, 0]:.3f}, D: {states1[-1, 1]:.3f}, "
               f"N: {states1[-1, 2]:.0f}")

    # Reset for next scenario
    simulator.reset()

    # =========================================================================
    # Scenario 2: Induce Differentiation
    # =========================================================================
    logger.info("\n3. Scenario 2: Inducing Differentiation (Retinoic Acid)")
    logger.info("-" * 60)

    time2, states2 = simulator.run_simulation(
        duration=14,
        timesteps=100,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
    )

    logger.info(f"Final state - P: {states2[-1, 0]:.3f}, D: {states2[-1, 1]:.3f}, "
               f"N: {states2[-1, 2]:.0f}")

    # Reset for next scenario
    simulator.reset()

    # =========================================================================
    # Scenario 3: Sequential Protocol (Expansion then Differentiation)
    # =========================================================================
    logger.info("\n4. Scenario 3: Sequential Protocol")
    logger.info("-" * 60)
    logger.info("Phase 1: Expansion (7 days with FGF2)")

    time3a, states3a = simulator.run_simulation(
        duration=7,
        timesteps=50,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
    )

    logger.info(f"After expansion - P: {states3a[-1, 0]:.3f}, D: {states3a[-1, 1]:.3f}, "
               f"N: {states3a[-1, 2]:.0f}")

    logger.info("Phase 2: Differentiation (7 days with Retinoic Acid)")

    time3b, states3b = simulator.add_perturbation(
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0},
        duration=7,
        timesteps=50
    )

    # Combine phases
    time3 = np.concatenate([time3a, time3a[-1] + time3b])
    states3 = np.concatenate([states3a, states3b])

    logger.info(f"Final state - P: {states3[-1, 0]:.3f}, D: {states3[-1, 1]:.3f}, "
               f"N: {states3[-1, 2]:.0f}")

    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\n5. Creating visualizations...")
    logger.info("-" * 60)

    # Create output directory
    output_dir = project_root / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual scenarios
    logger.info("Plotting Scenario 1: Pluripotency maintenance...")
    plotter.plot_trajectory(
        time1, states1,
        title="Scenario 1: Maintaining Pluripotency (High FGF2)",
        save_path=output_dir / "scenario1_pluripotency.png"
    )

    logger.info("Plotting Scenario 2: Differentiation...")
    plotter.plot_trajectory(
        time2, states2,
        title="Scenario 2: Inducing Differentiation (Retinoic Acid)",
        save_path=output_dir / "scenario2_differentiation.png"
    )

    logger.info("Plotting Scenario 3: Sequential protocol...")
    plotter.plot_trajectory(
        time3, states3,
        title="Scenario 3: Sequential Protocol (Expansion → Differentiation)",
        save_path=output_dir / "scenario3_sequential.png"
    )

    # Compare all scenarios
    logger.info("Creating comparison plot...")
    plotter.plot_comparison(
        time1,
        [states1, states2, states3],
        labels=["High FGF2 (Pluripotency)", "Retinoic Acid (Differentiation)",
                "Sequential Protocol"],
        title="Comparison of Differentiation Protocols",
        save_path=output_dir / "comparison_all_scenarios.png"
    )

    # Phase space plots
    logger.info("Creating phase space plots...")
    plotter.plot_phase_space(
        states2,
        title="Phase Space: Differentiation Trajectory",
        save_path=output_dir / "phase_space_differentiation.png"
    )

    logger.info(f"\nAll plots saved to: {output_dir}")

    # =========================================================================
    # Digital Twin Demo
    # =========================================================================
    logger.info("\n6. Digital Twin Demo")
    logger.info("-" * 60)

    # Reset simulator
    simulator.reset()

    # Create digital twin
    twin = DigitalTwinEngine(simulator, config=config)

    # Initialize
    logger.info("Initializing digital twin...")
    twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    # Simulate 3 days of updates
    logger.info("\nSimulating real-time updates...")
    for day in range(1, 4):
        state = twin.update(
            duration=24,  # 24 hours
            growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
        )
        logger.info(f"Day {day}: P={state['pluripotency']:.3f}, "
                   f"D={state['differentiation']:.3f}, N={state['population']:.0f}")

        # Get recommendations
        recs = twin.recommend_action()
        logger.info(f"  → {recs['actions'][0]['message']}")

    # Make prediction
    logger.info("\nGenerating 48-hour prediction...")
    prediction = twin.predict(
        horizon=48,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
    )

    final_pred = prediction['predicted_states'][-1]
    logger.info(f"Predicted state (48h): P={final_pred['pluripotency']:.3f}, "
               f"D={final_pred['differentiation']:.3f}")

    # Get metrics
    logger.info("\nDigital Twin Metrics:")
    metrics = twin.get_metrics()
    logger.info(f"  Duration: {metrics['duration']:.1f} hours")
    logger.info(f"  Pluripotency change: {metrics['pluripotency']['change']:.3f}")
    logger.info(f"  Differentiation change: {metrics['differentiation']['change']:.3f}")
    logger.info(f"  Population fold-change: {metrics['population']['fold_change']:.2f}x")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nKey Findings:")
    logger.info("  • High FGF2 maintains pluripotency")
    logger.info("  • Retinoic acid induces differentiation")
    logger.info("  • Sequential protocols enable controlled differentiation")
    logger.info("  • Digital twin provides real-time tracking and prediction")
    logger.info("=" * 60)

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
