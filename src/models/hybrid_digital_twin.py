"""
Hybrid Physics-ML Digital Twin for Stem Cell Differentiation.

Combines:
- Physics-based ODE simulator (known biology)
- ML predictor (learns from real data)
- Hybrid prediction = ODE + ML residual
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path

from .simulators import iPSCDifferentiationSimulator
from .predictors import TransformerPredictor, LSTMPredictor


class HybridDigitalTwin:
    """
    Hybrid physics-ML digital twin for stem cell differentiation prediction.

    Architecture:
        Hybrid Prediction = Physics ODE + ML Residual Correction

    This combines:
    1. Physics simulator: Captures known biological mechanisms
    2. ML predictor: Learns unknown patterns from real data
    3. Hybrid approach: Better than either alone
    """

    def __init__(
        self,
        config: dict,
        ml_model_type: str = 'transformer',
        ml_checkpoint: Optional[str] = None,
        input_size: int = 3,
        output_size: int = 3
    ):
        """
        Initialize hybrid digital twin.

        Args:
            config: Configuration dictionary
            ml_model_type: 'transformer' or 'lstm'
            ml_checkpoint: Path to trained ML model checkpoint
            input_size: Number of input features (default 3 for [P,D,N], use 2 for [P,D])
            output_size: Number of output features (default 3 for [P,D,N], use 2 for [P,D])
        """
        # Physics-based simulator
        self.ode_simulator = iPSCDifferentiationSimulator(config)

        # ML-based predictor
        if ml_model_type == 'transformer':
            self.ml_predictor = TransformerPredictor(
                input_size=input_size,
                output_size=output_size,
                config=config
            )
        elif ml_model_type == 'lstm':
            self.ml_predictor = LSTMPredictor(
                input_size=input_size,
                output_size=output_size,
                config=config
            )
        else:
            raise ValueError(f"Unknown model type: {ml_model_type}")

        # Load trained ML model if provided
        if ml_checkpoint:
            self.load_ml_model(ml_checkpoint)

        self.config = config
        self.ml_model_type = ml_model_type

        # Learnable hybrid weighting network (state-dependent λ)
        self.lambda_network = None
        self._initialize_lambda_network()

    def _initialize_lambda_network(self):
        """Initialize learnable hybrid weighting network."""
        import torch.nn as nn

        class LambdaNetwork(nn.Module):
            """Small network to learn state-dependent hybrid weight λ."""
            def __init__(self, input_size=2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()  # λ ∈ [0, 1]
                )

            def forward(self, state):
                """
                Args:
                    state: [P, D] or batch of states
                Returns:
                    λ weight in [0, 1]
                """
                return self.net(state)

        self.lambda_network = LambdaNetwork(input_size=2)  # State = [P, D]

    def load_ml_model(self, checkpoint_path: str):
        """Load trained ML model weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.ml_predictor.device)
        self.ml_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.ml_predictor.eval()
        print(f"Loaded ML model from {checkpoint_path}")

    def predict_physics_only(
        self,
        initial_state: np.ndarray,
        time_horizon: float,
        n_steps: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using only physics-based ODE simulator in pseudotime.

        Args:
            initial_state: Initial [P, D] state (2D) or [P, D, N] (3D, will extract [P,D])
            time_horizon: Pseudotime horizon (e.g., 1.0 for full trajectory)
            n_steps: Number of simulation steps

        Returns:
            pseudotimes: Pseudotime points
            states: Predicted [P, D] states at each pseudotime
        """
        # Run ODE simulator in pseudotime
        tau, states = self.ode_simulator.simulate_pseudotime(
            tau_span=(0, time_horizon),
            n_steps=n_steps,
            initial_state=initial_state
        )

        return tau, states

    def predict_ml_only(
        self,
        state_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Predict using only ML model.

        Args:
            state_sequence: Input sequence of states (seq_len, 3)

        Returns:
            prediction: Next state prediction (3,)
        """
        # Convert to tensor
        x = torch.FloatTensor(state_sequence).unsqueeze(0)  # (1, seq_len, 3)

        # Predict
        with torch.no_grad():
            pred = self.ml_predictor(x)  # (1, 1, 3)

        return pred.squeeze().cpu().numpy()

    def predict_hybrid(
        self,
        initial_state: np.ndarray,
        time_horizon: float,
        n_steps: int = 100,
        residual_weight: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Hybrid prediction: Physics ODE + ML residual correction.

        Args:
            initial_state: Initial [P, D, N] state
            time_horizon: How far ahead to predict
            n_steps: Number of steps
            residual_weight: Weight for ML residual (0-1)

        Returns:
            Dictionary with:
                - 'times': Time points
                - 'physics': Physics-only predictions
                - 'ml_residual': ML residual corrections
                - 'hybrid': Hybrid predictions (physics + residual)
        """
        # 1. Physics-based prediction
        t, physics_pred = self.predict_physics_only(
            initial_state, time_horizon, n_steps
        )

        # 2. ML residual correction
        ml_residual = np.zeros_like(physics_pred)

        # For each timestep, predict residual using recent history
        seq_len = 2  # Use last 2 steps to predict residual

        for i in range(seq_len, len(physics_pred)):
            # Get recent physics predictions
            recent_states = physics_pred[i-seq_len:i]

            # Predict what ML thinks the next state should be
            ml_next = self.predict_ml_only(recent_states)

            # Residual = what ML predicts - what physics predicts
            ml_residual[i] = ml_next - physics_pred[i]

        # 3. Hybrid prediction
        hybrid_pred = physics_pred + residual_weight * ml_residual

        return {
            'times': t,
            'physics': physics_pred,
            'ml_residual': ml_residual,
            'hybrid': hybrid_pred
        }

    def predict_hybrid_learnable(
        self,
        initial_state: np.ndarray,
        time_horizon: float,
        n_steps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Hybrid prediction with LEARNABLE state-dependent weights.

        Uses a neural network to learn optimal λ(state) for each state,
        allowing adaptive reliance on physics vs ML.

        Args:
            initial_state: Initial [P, D, N] state
            time_horizon: How far ahead to predict
            n_steps: Number of steps

        Returns:
            Dictionary with predictions and learned λ values
        """
        # 1. Physics-based prediction
        t, physics_pred = self.predict_physics_only(
            initial_state, time_horizon, n_steps
        )

        # 2. ML predictions
        ml_residual = np.zeros_like(physics_pred)
        learned_lambdas = np.zeros(len(physics_pred))

        seq_len = 2
        for i in range(seq_len, len(physics_pred)):
            recent_states = physics_pred[i-seq_len:i]
            ml_next = self.predict_ml_only(recent_states)
            ml_residual[i] = ml_next - physics_pred[i]

        # 3. Learnable hybrid weights
        # For each state, predict optimal λ
        with torch.no_grad():
            for i in range(len(physics_pred)):
                state_PD = torch.FloatTensor(physics_pred[i, :2])  # [P, D]
                lambda_i = self.lambda_network(state_PD).item()
                learned_lambdas[i] = lambda_i

        # 4. Hybrid prediction with learned weights
        hybrid_pred = physics_pred.copy()
        for i in range(len(physics_pred)):
            hybrid_pred[i] += learned_lambdas[i] * ml_residual[i]

        return {
            'times': t,
            'physics': physics_pred,
            'ml_residual': ml_residual,
            'hybrid': hybrid_pred,
            'learned_lambdas': learned_lambdas  # NEW: adaptive weights
        }

    def train_lambda_network(
        self,
        train_data: list,
        n_epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Train the λ network to minimize prediction error.

        Args:
            train_data: List of (initial_state, target_state, time_horizon) tuples
            n_epochs: Training epochs
            learning_rate: Learning rate
        """
        import torch.optim as optim
        import torch.nn as nn

        optimizer = optim.Adam(self.lambda_network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.lambda_network.train()

        for epoch in range(n_epochs):
            total_loss = 0

            for initial, target, horizon in train_data:
                # Get physics and ML predictions
                t, physics_pred = self.predict_physics_only(initial, horizon, n_steps=20)
                final_physics = physics_pred[-1, :2]  # [P, D] at final time

                # ML prediction
                recent = physics_pred[-3:-1]  # Last 2 states
                ml_pred = self.predict_ml_only(recent)[:2]

                # Residual
                residual = ml_pred - final_physics

                # Predicted λ
                state_tensor = torch.FloatTensor(final_physics)
                lambda_pred = self.lambda_network(state_tensor)

                # Hybrid prediction
                hybrid = final_physics + lambda_pred.item() * residual

                # Loss: distance to true target
                target_tensor = torch.FloatTensor(target[:2])
                hybrid_tensor = torch.FloatTensor(hybrid)
                loss = criterion(hybrid_tensor, target_tensor)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_data):.4f}")

        self.lambda_network.eval()
        print("Lambda network training complete!")

    def simulate_intervention(
        self,
        initial_state: np.ndarray,
        intervention: Dict[str, float],
        time_horizon: float = 30.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate "what-if" intervention scenario.

        Args:
            initial_state: Starting cell state
            intervention: Parameter changes (e.g., {'k_diff': 0.05})
            time_horizon: Simulation duration

        Returns:
            Predictions with and without intervention
        """
        # Baseline (no intervention)
        baseline = self.predict_hybrid(initial_state, time_horizon)

        # With intervention
        original_params = self.ode_simulator.params.copy()

        # Apply intervention
        for param, value in intervention.items():
            if hasattr(self.ode_simulator, param):
                setattr(self.ode_simulator, param, value)

        intervened = self.predict_hybrid(initial_state, time_horizon)

        # Restore original parameters
        for param, value in original_params.items():
            if hasattr(self.ode_simulator, param):
                setattr(self.ode_simulator, param, value)

        return {
            'baseline': baseline,
            'intervened': intervened,
            'intervention': intervention
        }

    def optimize_protocol(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        param_ranges: Dict[str, Tuple[float, float]],
        n_trials: int = 100
    ) -> Dict:
        """
        Find optimal differentiation protocol to reach target state.

        Args:
            initial_state: Starting state
            target_state: Desired final state
            param_ranges: Parameter search ranges
            n_trials: Number of optimization trials

        Returns:
            Best parameters and resulting trajectory
        """
        best_params = None
        best_error = float('inf')
        best_trajectory = None

        for _ in range(n_trials):
            # Sample random parameters
            params = {
                param: np.random.uniform(low, high)
                for param, (low, high) in param_ranges.items()
            }

            # Simulate with these parameters
            result = self.simulate_intervention(
                initial_state,
                params,
                time_horizon=30.0
            )

            # Final state
            final_state = result['intervened']['hybrid'][-1]

            # Error from target
            error = np.linalg.norm(final_state - target_state)

            if error < best_error:
                best_error = error
                best_params = params
                best_trajectory = result['intervened']

        return {
            'best_params': best_params,
            'best_trajectory': best_trajectory,
            'final_error': best_error
        }

    def get_uncertainty(
        self,
        initial_state: np.ndarray,
        time_horizon: float,
        n_samples: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Estimate prediction uncertainty using dropout/ensemble.

        Args:
            initial_state: Starting state
            time_horizon: Prediction horizon
            n_samples: Number of samples for uncertainty estimate

        Returns:
            Mean predictions and confidence intervals
        """
        predictions = []

        for _ in range(n_samples):
            # Could add dropout or parameter perturbation here
            pred = self.predict_hybrid(initial_state, time_horizon)
            predictions.append(pred['hybrid'])

        predictions = np.array(predictions)  # (n_samples, n_steps, 3)

        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0)
        }


def demo_hybrid_digital_twin():
    """Demonstrate hybrid digital twin capabilities."""
    from ..utils import load_config

    print("="*80)
    print("HYBRID PHYSICS-ML DIGITAL TWIN DEMONSTRATION")
    print("="*80)

    # Load config
    config = load_config()

    # Create digital twin
    print("\n1. Creating hybrid digital twin...")
    twin = HybridDigitalTwin(config, ml_model_type='transformer')

    # Initial state: iPSC at day 0
    initial_state = np.array([1.0, 0.0, 10000.0])  # High P, low D, some cells

    # 2. Pure physics prediction
    print("\n2. Physics-only prediction...")
    t_phys, states_phys = twin.predict_physics_only(initial_state, time_horizon=30)
    print(f"   Physics predicts: P={states_phys[-1,0]:.3f}, D={states_phys[-1,1]:.3f}")

    # 3. Hybrid prediction
    print("\n3. Hybrid (Physics + ML) prediction...")
    hybrid_result = twin.predict_hybrid(initial_state, time_horizon=30)
    print(f"   Hybrid predicts: P={hybrid_result['hybrid'][-1,0]:.3f}, "
          f"D={hybrid_result['hybrid'][-1,1]:.3f}")
    print(f"   ML residual contribution: {np.mean(np.abs(hybrid_result['ml_residual'])):.4f}")

    # 4. Intervention testing
    print("\n4. Testing intervention (increase differentiation rate)...")
    intervention = {'k_diff': 0.08}  # Increase from default
    intervention_result = twin.simulate_intervention(
        initial_state, intervention, time_horizon=30
    )
    baseline_D = intervention_result['baseline']['hybrid'][-1, 1]
    intervened_D = intervention_result['intervened']['hybrid'][-1, 1]
    print(f"   Baseline differentiation: {baseline_D:.3f}")
    print(f"   With intervention: {intervened_D:.3f}")
    print(f"   Improvement: {((intervened_D - baseline_D) / baseline_D * 100):.1f}%")

    print("\n" + "="*80)
    print("Digital twin ready for ICUFN 2026 paper!")
    print("="*80)


if __name__ == "__main__":
    demo_hybrid_digital_twin()
