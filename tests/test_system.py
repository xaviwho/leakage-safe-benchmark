"""
System-level tests to validate the complete framework.

Tests:
1. ODE simulator functionality
2. Data generation
3. ML model creation and forward pass
4. Digital twin integration
5. Stochastic simulation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.models.simulators import iPSCDifferentiationSimulator, StochasticiPSCSimulator
from src.models.digital_twin import DigitalTwinEngine
from src.models.predictors import LSTMPredictor, TransformerPredictor
from src.data.data_generator import SyntheticDataGenerator
from src.utils import load_config


def test_ode_simulator():
    """Test ODE simulator."""
    print("Testing ODE Simulator...")

    config = load_config()
    simulator = iPSCDifferentiationSimulator(config)

    # Run simulation
    time, states = simulator.run_simulation(
        duration=7,
        timesteps=50,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
    )

    # Validate output
    assert len(time) == 50, "Time array length mismatch"
    assert states.shape == (50, 3), "States shape mismatch"
    assert np.all(states[:, 0] >= 0) and np.all(states[:, 0] <= 1), "Pluripotency out of bounds"
    assert np.all(states[:, 1] >= 0) and np.all(states[:, 1] <= 1), "Differentiation out of bounds"
    assert np.all(states[:, 2] > 0), "Population should be positive"

    # Check final state
    final_state = simulator.get_current_state()
    assert 'pluripotency' in final_state
    assert 'differentiation' in final_state
    assert 'population' in final_state

    print("  ✓ ODE Simulator works correctly")


def test_data_generation():
    """Test synthetic data generation."""
    print("Testing Data Generation...")

    config = load_config()
    simulator = iPSCDifferentiationSimulator(config)
    data_generator = SyntheticDataGenerator(simulator, config)

    # Generate single trajectory
    time_points, states = data_generator.generate_trajectory(timesteps=50)

    assert len(time_points) > 0, "No time points generated"
    assert states.shape[1] == 3, "States should have 3 variables"

    # Generate dataset
    dataset = data_generator.generate_dataset(n_trajectories=10)

    assert len(dataset) == 10, "Dataset size mismatch"
    for time, state in dataset:
        assert len(time) > 0, "Empty trajectory"
        assert state.shape[1] == 3, "States shape mismatch"

    print("  ✓ Data Generation works correctly")


def test_lstm_model():
    """Test LSTM predictor."""
    print("Testing LSTM Predictor...")

    config = load_config()
    model = LSTMPredictor(input_size=3, output_size=3, config=config)

    # Test forward pass
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 3)

    output = model(x)

    assert output.shape == (batch_size, seq_len, 3), "Output shape mismatch"

    # Test prediction
    history = np.random.randn(20, 3)
    predictions, uncertainty = model.predict(history, horizon=10, return_uncertainty=True)

    assert predictions.shape == (10, 3), "Predictions shape mismatch"
    assert uncertainty.shape == (10, 3), "Uncertainty shape mismatch"

    print("  ✓ LSTM Predictor works correctly")


def test_transformer_model():
    """Test Transformer predictor."""
    print("Testing Transformer Predictor...")

    config = load_config()
    model = TransformerPredictor(input_size=3, output_size=3, config=config)

    # Test forward pass
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 3)

    output = model(x)

    assert output.shape == (batch_size, seq_len, 3), "Output shape mismatch"

    # Test prediction
    history = np.random.randn(20, 3)
    predictions, _ = model.predict(history, horizon=10, return_uncertainty=False)

    assert predictions.shape == (10, 3), "Predictions shape mismatch"

    print("  ✓ Transformer Predictor works correctly")


def test_digital_twin():
    """Test digital twin functionality."""
    print("Testing Digital Twin...")

    config = load_config()
    simulator = iPSCDifferentiationSimulator(config)
    twin = DigitalTwinEngine(simulator, config=config)

    # Initialize
    twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    assert twin.current_state is not None, "Twin not initialized"
    assert len(twin.state_history) > 0, "No state history"

    # Update
    state = twin.update(duration=24, growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    assert 'pluripotency' in state
    assert 'differentiation' in state
    assert 'population' in state

    # Predict
    prediction = twin.predict(horizon=48, growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0})

    assert 'predicted_states' in prediction
    assert len(prediction['predicted_states']) > 0

    # Recommendations
    recs = twin.recommend_action()

    assert 'actions' in recs
    assert len(recs['actions']) > 0

    # Metrics
    metrics = twin.get_metrics()

    assert 'pluripotency' in metrics
    assert 'differentiation' in metrics
    assert 'population' in metrics

    print("  ✓ Digital Twin works correctly")


def test_stochastic_simulator():
    """Test stochastic simulator."""
    print("Testing Stochastic Simulator...")

    config = load_config()
    stochastic_sim = StochasticiPSCSimulator(config)

    # Single realization
    time, states = stochastic_sim.run_simulation(
        duration=7,
        timesteps=50,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0},
        add_noise=True,
        n_realizations=1
    )

    assert states.shape == (50, 3), "States shape mismatch"

    # Ensemble
    time, ensemble_states = stochastic_sim.run_simulation(
        duration=7,
        timesteps=50,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0},
        add_noise=True,
        n_realizations=5
    )

    assert ensemble_states.shape == (50, 5, 3), "Ensemble states shape mismatch"

    # Statistics
    stats = stochastic_sim.get_ensemble_statistics(ensemble_states)

    assert 'mean' in stats
    assert 'std' in stats
    assert stats['mean'].shape == (50, 3)

    print("  ✓ Stochastic Simulator works correctly")


def test_hybrid_prediction():
    """Test hybrid physics-ML prediction."""
    print("Testing Hybrid Prediction...")

    config = load_config()
    simulator = iPSCDifferentiationSimulator(config)

    # Create ML predictor
    lstm_model = LSTMPredictor(input_size=3, output_size=3, config=config)

    # Create twin with ML
    twin = DigitalTwinEngine(simulator, predictor=lstm_model, config=config)

    # Initialize and update
    twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    for _ in range(3):
        twin.update(duration=24, growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    # Test hybrid prediction
    prediction = twin.predict(
        horizon=48,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0},
        use_ml=True,
        fusion_weight=0.5
    )

    assert prediction['method'] in ['physics', 'ml', 'hybrid']
    assert 'predicted_states' in prediction
    assert len(prediction['predicted_states']) > 0

    # Test comparison
    comparison = twin.compare_prediction_methods(
        horizon=48,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
    )

    assert 'physics' in comparison
    assert 'ml' in comparison
    assert 'hybrid' in comparison

    print("  ✓ Hybrid Prediction works correctly")


def run_all_tests():
    """Run all system tests."""
    print("=" * 70)
    print("SYSTEM VALIDATION TESTS")
    print("=" * 70)

    tests = [
        test_ode_simulator,
        test_data_generation,
        test_lstm_model,
        test_transformer_model,
        test_digital_twin,
        test_stochastic_simulator,
        test_hybrid_prediction
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 70)
    if not failed:
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe system is fully functional and ready for use:")
        print("  • ODE simulator ✓")
        print("  • Data generation ✓")
        print("  • ML predictors (LSTM & Transformer) ✓")
        print("  • Digital twin with ML integration ✓")
        print("  • Stochastic simulation ✓")
        print("  • Hybrid physics-ML predictions ✓")
        print("  • Uncertainty quantification ✓")
        return True
    else:
        print(f"FAILED: {len(failed)} test(s)")
        print("=" * 70)
        for test_name in failed:
            print(f"  ✗ {test_name}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
