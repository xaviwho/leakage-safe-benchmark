"""
Hybrid Physics-ML Digital Twin Demo.

This script demonstrates the complete hybrid system:
1. Generate training data from ODE simulator
2. Train an LSTM predictor
3. Integrate ML with digital twin
4. Compare physics, ML, and hybrid predictions
5. Demonstrate uncertainty quantification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from src.models.simulators import iPSCDifferentiationSimulator, StochasticiPSCSimulator
from src.models.predictors import LSTMPredictor
from src.models.predictors.trainer import PredictorTrainer, CellStateDataset
from src.models.digital_twin import DigitalTwinEngine
from src.data.data_generator import SyntheticDataGenerator
from src.visualization import DigitalTwinPlotter
from src.utils import load_config, setup_logger

# Setup logging
logger = setup_logger("hybrid_ml_demo", level="INFO")


def main():
    """Run hybrid ML-physics demonstration."""
    logger.info("=" * 70)
    logger.info("HYBRID PHYSICS-ML DIGITAL TWIN DEMONSTRATION")
    logger.info("=" * 70)

    # Load configuration
    config = load_config()
    plotter = DigitalTwinPlotter()

    # Create output directory
    output_dir = project_root / "experiments" / "results" / "hybrid_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Generate Training Data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Generating Training Data")
    logger.info("=" * 70)

    simulator = iPSCDifferentiationSimulator(config)
    data_generator = SyntheticDataGenerator(simulator, config)

    logger.info("Generating 200 training trajectories...")
    train_trajectories = data_generator.generate_dataset(
        n_trajectories=200,
        timesteps=100,
        add_noise=True,
        noise_level=0.02
    )

    logger.info("Generating 50 validation trajectories...")
    val_trajectories = data_generator.generate_validation_set(n_trajectories=50)

    # Create datasets
    train_states = [states for _, states in train_trajectories]
    val_states = [states for _, states in val_trajectories]

    train_dataset = CellStateDataset(train_states, sequence_length=20, prediction_horizon=10)
    val_dataset = CellStateDataset(val_states, sequence_length=20, prediction_horizon=10)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # =========================================================================
    # STEP 2: Train LSTM Predictor
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Training LSTM Predictor")
    logger.info("=" * 70)

    # Create LSTM model
    lstm_model = LSTMPredictor(input_size=3, output_size=3, config=config)
    logger.info(f"LSTM model created with {lstm_model.count_parameters():,} parameters")

    # Update config for quick training (reduced epochs for demo)
    config['ml_models']['lstm']['epochs'] = 30
    config['ml_models']['lstm']['batch_size'] = 32

    # Create trainer
    trainer = PredictorTrainer(
        model=lstm_model,
        config=config,
        experiment_dir=output_dir / "lstm_training"
    )

    # Train
    logger.info("Training LSTM (30 epochs for demo)...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_best=True
    )

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches='tight')
    logger.info(f"Training history saved to {output_dir / 'training_history.png'}")

    # =========================================================================
    # STEP 3: Create Hybrid Digital Twin
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Creating Hybrid Digital Twin")
    logger.info("=" * 70)

    # Reset simulator
    simulator.reset()

    # Create digital twin with trained ML predictor
    twin = DigitalTwinEngine(simulator, predictor=lstm_model, config=config)

    logger.info("Digital twin created with ML predictor integrated")

    # Initialize twin
    twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})
    logger.info(f"Twin initialized: {twin.current_state}")

    # =========================================================================
    # STEP 4: Simulate Real-Time Updates
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Real-Time Updates (3 days)")
    logger.info("=" * 70)

    # Run for 3 days with updates
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

    # =========================================================================
    # STEP 5: Compare Prediction Methods
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Comparing Prediction Methods")
    logger.info("=" * 70)

    # Compare physics, ML, and hybrid predictions
    logger.info("Generating predictions with different methods...")
    comparison = twin.compare_prediction_methods(
        horizon=48,  # 48 hours
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
    )

    # Extract predictions for plotting
    time_physics = comparison['physics']['time_points']
    states_physics = np.array([
        [s['pluripotency'], s['differentiation'], s['population']]
        for s in comparison['physics']['predicted_states']
    ])

    time_ml = comparison['ml']['time_points']
    states_ml = np.array([
        [s['pluripotency'], s['differentiation'], s['population']]
        for s in comparison['ml']['predicted_states']
    ])

    time_hybrid = comparison['hybrid']['time_points']
    states_hybrid = np.array([
        [s['pluripotency'], s['differentiation'], s['population']]
        for s in comparison['hybrid']['predicted_states']
    ])

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pluripotency
    axes[0, 0].plot(time_physics / 24, states_physics[:, 0], 'b-', linewidth=2, label='Physics')
    axes[0, 0].plot(time_ml / 24, states_ml[:, 0], 'r-', linewidth=2, label='ML')
    axes[0, 0].plot(time_hybrid / 24, states_hybrid[:, 0], 'g-', linewidth=2, label='Hybrid')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Pluripotency')
    axes[0, 0].set_title('Pluripotency Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Differentiation
    axes[0, 1].plot(time_physics / 24, states_physics[:, 1], 'b-', linewidth=2, label='Physics')
    axes[0, 1].plot(time_ml / 24, states_ml[:, 1], 'r-', linewidth=2, label='ML')
    axes[0, 1].plot(time_hybrid / 24, states_hybrid[:, 1], 'g-', linewidth=2, label='Hybrid')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Differentiation')
    axes[0, 1].set_title('Differentiation Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Population
    axes[1, 0].plot(time_physics / 24, states_physics[:, 2], 'b-', linewidth=2, label='Physics')
    axes[1, 0].plot(time_ml / 24, states_ml[:, 2], 'r-', linewidth=2, label='ML')
    axes[1, 0].plot(time_hybrid / 24, states_hybrid[:, 2], 'g-', linewidth=2, label='Hybrid')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cell Number')
    axes[1, 0].set_title('Population Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Phase space
    axes[1, 1].plot(states_physics[:, 0], states_physics[:, 1], 'b-', linewidth=2, label='Physics', alpha=0.7)
    axes[1, 1].plot(states_ml[:, 0], states_ml[:, 1], 'r-', linewidth=2, label='ML', alpha=0.7)
    axes[1, 1].plot(states_hybrid[:, 0], states_hybrid[:, 1], 'g-', linewidth=2, label='Hybrid', alpha=0.7)
    axes[1, 1].set_xlabel('Pluripotency')
    axes[1, 1].set_ylabel('Differentiation')
    axes[1, 1].set_title('Phase Space Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Hybrid Physics-ML Predictions: Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"Method comparison saved to {output_dir / 'method_comparison.png'}")

    # =========================================================================
    # STEP 6: Uncertainty Quantification
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Uncertainty Quantification")
    logger.info("=" * 70)

    # Use stochastic simulator for ensemble predictions
    stochastic_sim = StochasticiPSCSimulator(config)
    twin_stochastic = DigitalTwinEngine(stochastic_sim, predictor=lstm_model, config=config)
    twin_stochastic.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    # Update for 3 days
    for _ in range(3):
        twin_stochastic.update(duration=24, growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    # Predict with uncertainty
    prediction_with_uncertainty = twin_stochastic.predict(
        horizon=48,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0},
        use_ml=True,
        fusion_weight=0.5
    )

    # Extract uncertainty
    pred_states_unc = prediction_with_uncertainty['predicted_states']
    time_unc = prediction_with_uncertainty['time_points']

    # Plot with uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pluripotency with uncertainty
    pluri_vals = [s['pluripotency'] for s in pred_states_unc]
    pluri_unc = [s.get('uncertainty', {}).get('pluripotency', 0) for s in pred_states_unc]

    axes[0].plot(time_unc / 24, pluri_vals, 'b-', linewidth=2, label='Prediction')
    axes[0].fill_between(
        time_unc / 24,
        np.array(pluri_vals) - 2 * np.array(pluri_unc),
        np.array(pluri_vals) + 2 * np.array(pluri_unc),
        alpha=0.3, color='blue', label='95% CI'
    )
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Pluripotency')
    axes[0].set_title('Pluripotency Prediction with Uncertainty')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Differentiation with uncertainty
    diff_vals = [s['differentiation'] for s in pred_states_unc]
    diff_unc = [s.get('uncertainty', {}).get('differentiation', 0) for s in pred_states_unc]

    axes[1].plot(time_unc / 24, diff_vals, 'r-', linewidth=2, label='Prediction')
    axes[1].fill_between(
        time_unc / 24,
        np.array(diff_vals) - 2 * np.array(diff_unc),
        np.array(diff_vals) + 2 * np.array(diff_unc),
        alpha=0.3, color='red', label='95% CI'
    )
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Differentiation')
    axes[1].set_title('Differentiation Prediction with Uncertainty')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_quantification.png", dpi=300, bbox_inches='tight')
    logger.info(f"Uncertainty plot saved to {output_dir / 'uncertainty_quantification.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nKey Achievements:")
    logger.info("  ✓ Generated synthetic training data from ODE simulator")
    logger.info("  ✓ Trained LSTM predictor on differentiation trajectories")
    logger.info("  ✓ Integrated ML with physics-based digital twin")
    logger.info("  ✓ Demonstrated hybrid prediction (physics + ML fusion)")
    logger.info("  ✓ Quantified prediction uncertainty")
    logger.info("\nThis demonstrates a true Hybrid Physics-ML Digital Twin!")
    logger.info("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
