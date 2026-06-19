"""
Digital Twin Engine for iPSC Differentiation.

This module implements the core digital twin that:
1. Maintains a virtual replica of the cell culture
2. Tracks real-time state
3. Predicts future outcomes
4. Provides recommendations
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DigitalTwinEngine:
    """
    Digital Twin Engine that integrates simulation and prediction.

    The twin maintains synchronization between the physical system
    (or simulator in our case) and the virtual model.
    """

    def __init__(
        self,
        simulator: Any,
        predictor: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Digital Twin Engine.

        Args:
            simulator: Mechanistic simulator (e.g., iPSCDifferentiationSimulator)
            predictor: ML predictor for future states (optional)
            config: Configuration dictionary
        """
        self.simulator = simulator
        self.predictor = predictor
        self.config = config or {}

        # Twin state
        self.current_state = None
        self.state_history = []
        self.predictions = []
        self.uncertainty = None

        # Time tracking
        self.start_time = None
        self.current_time = 0.0

        # Settings
        self.update_frequency = self.config.get('digital_twin', {}).get('update_frequency', 1.0)
        self.prediction_horizon = self.config.get('digital_twin', {}).get('prediction_horizon', 48)

        logger.info("Digital Twin Engine initialized")

    def initialize(
        self,
        initial_state: Optional[Dict[str, float]] = None,
        growth_factors: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the digital twin.

        Args:
            initial_state: Initial cell state (P, D, N)
            growth_factors: Initial growth factor concentrations
        """
        self.start_time = datetime.now()
        self.current_time = 0.0

        # Initialize simulator
        if initial_state:
            state_array = np.array([
                initial_state.get('pluripotency', 0.95),
                initial_state.get('differentiation', 0.05),
                initial_state.get('population', 10000)
            ])
        else:
            state_array = None

        # Run initial simulation
        time_points, states = self.simulator.run_simulation(
            duration=0.1,  # Very short initial run
            timesteps=2,
            initial_state=state_array,
            growth_factors=growth_factors
        )

        self.current_state = self.simulator.get_current_state()
        self.state_history.append({
            'time': self.current_time,
            'state': self.current_state.copy(),
            'timestamp': self.start_time
        })

        logger.info(f"Digital Twin initialized at t=0: {self.current_state}")

    def update(
        self,
        measured_state: Optional[Dict[str, float]] = None,
        growth_factors: Optional[Dict[str, float]] = None,
        duration: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update digital twin state.

        Args:
            measured_state: Measured state from physical system (if available)
            growth_factors: Current growth factor concentrations
            duration: Time elapsed since last update (in hours)

        Returns:
            Updated state dictionary
        """
        if duration is None:
            duration = self.update_frequency

        # Convert hours to days for simulator
        duration_days = duration / 24.0

        # If we have measurements, use them to correct the twin
        if measured_state:
            logger.info(f"Updating with measured state: {measured_state}")
            # State correction (simple: replace with measurement)
            # In advanced implementation, use Kalman filter or particle filter
            self.current_state.update(measured_state)

        # Run simulator forward
        current_state_array = np.array([
            self.current_state['pluripotency'],
            self.current_state['differentiation'],
            self.current_state['population']
        ])

        time_points, states = self.simulator.run_simulation(
            duration=duration_days,
            timesteps=10,
            initial_state=current_state_array,
            growth_factors=growth_factors
        )

        # Update current state
        self.current_time += duration
        self.current_state = self.simulator.get_current_state()
        self.current_state['time'] = self.current_time

        # Store history
        self.state_history.append({
            'time': self.current_time,
            'state': self.current_state.copy(),
            'timestamp': datetime.now()
        })

        logger.debug(f"Twin updated to t={self.current_time:.2f}h: {self.current_state}")

        return self.current_state

    def predict(
        self,
        horizon: Optional[int] = None,
        growth_factors: Optional[Dict[str, float]] = None,
        use_ml: bool = True,
        fusion_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict future state of the system using hybrid approach.

        Uses both mechanistic simulation and ML predictions (if available),
        combining them based on fusion_weight.

        Args:
            horizon: Prediction horizon in hours (if None, uses default)
            growth_factors: Assumed growth factor profile
            use_ml: Whether to use ML predictor (if available)
            fusion_weight: Weight for ML prediction (0=pure physics, 1=pure ML)

        Returns:
            Dictionary with predictions including:
            - predicted_states: List of future states
            - time_points: Future time points
            - uncertainty: Uncertainty estimates (if available)
            - method: Prediction method used ('physics', 'ml', or 'hybrid')
        """
        if horizon is None:
            horizon = self.prediction_horizon

        # Convert hours to days
        horizon_days = horizon / 24.0

        current_state_array = np.array([
            self.current_state['pluripotency'],
            self.current_state['differentiation'],
            self.current_state['population']
        ])

        # 1. Physics-based prediction (always computed)
        timesteps = max(10, horizon)
        time_points, physics_states = self.simulator.run_simulation(
            duration=horizon_days,
            timesteps=timesteps,
            initial_state=current_state_array,
            growth_factors=growth_factors
        )

        # 2. ML-based prediction (if predictor available and use_ml=True)
        ml_states = None
        uncertainty = None
        method = 'physics'

        if self.predictor is not None and use_ml:
            try:
                # Get historical states for ML input
                history_length = min(20, len(self.state_history))
                if history_length > 0:
                    history_states = np.array([
                        [h['state']['pluripotency'],
                         h['state']['differentiation'],
                         h['state']['population']]
                        for h in self.state_history[-history_length:]
                    ])

                    # Add current state
                    history_states = np.vstack([history_states, current_state_array])

                    # ML prediction
                    ml_pred, uncertainty = self.predictor.predict(
                        history=history_states,
                        horizon=timesteps,
                        return_uncertainty=True
                    )

                    ml_states = ml_pred
                    method = 'ml' if fusion_weight >= 0.99 else 'hybrid'

                    logger.debug(f"ML prediction computed with uncertainty")

            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Using physics only.")
                ml_states = None

        # 3. Fusion of predictions (if both available)
        if ml_states is not None and fusion_weight > 0:
            # Weighted average of physics and ML predictions
            fused_states = (1 - fusion_weight) * physics_states + fusion_weight * ml_states
            final_states = fused_states
            logger.info(f"Hybrid prediction (fusion_weight={fusion_weight:.2f})")
        else:
            final_states = physics_states
            method = 'physics'
            logger.info("Physics-only prediction")

        # Convert to list of state dicts
        predicted_states = []
        for i, (t, s) in enumerate(zip(time_points, final_states)):
            state_dict = {
                'time': self.current_time + t * 24,  # Convert back to hours
                'pluripotency': float(s[0]),
                'differentiation': float(s[1]),
                'population': float(s[2])
            }

            # Add uncertainty if available
            if uncertainty is not None:
                state_dict['uncertainty'] = {
                    'pluripotency': float(uncertainty[i, 0]),
                    'differentiation': float(uncertainty[i, 1]),
                    'population': float(uncertainty[i, 2])
                }

            predicted_states.append(state_dict)

        prediction = {
            'predicted_states': predicted_states,
            'time_points': time_points * 24 + self.current_time,  # Hours from start
            'horizon': horizon,
            'uncertainty': uncertainty,
            'method': method,
            'fusion_weight': fusion_weight if method == 'hybrid' else None
        }

        self.predictions.append(prediction)

        logger.info(f"Generated {method} prediction for {horizon}h horizon")

        return prediction

    def recommend_action(self) -> Dict[str, Any]:
        """
        Recommend actions based on current state and predictions.

        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'timestamp': datetime.now(),
            'current_time': self.current_time,
            'actions': []
        }

        # Check pluripotency level
        pluri = self.current_state['pluripotency']
        diff = self.current_state['differentiation']

        if pluri < 0.4 and diff < 0.5:
            recommendations['actions'].append({
                'type': 'warning',
                'message': 'Cells losing pluripotency without differentiating',
                'suggestion': 'Check culture conditions and consider adding FGF2'
            })

        if pluri > 0.8:
            recommendations['actions'].append({
                'type': 'info',
                'message': 'Cells maintaining high pluripotency',
                'suggestion': 'Good for expansion; add differentiation stimulus when ready'
            })

        if diff > 0.7:
            recommendations['actions'].append({
                'type': 'success',
                'message': 'Differentiation progressing well',
                'suggestion': 'Continue current protocol'
            })

        # Population check
        pop = self.current_state['population']
        if pop > 500000:
            recommendations['actions'].append({
                'type': 'action',
                'message': 'High cell density detected',
                'suggestion': 'Consider passaging cells'
            })

        if not recommendations['actions']:
            recommendations['actions'].append({
                'type': 'info',
                'message': 'All parameters within normal range',
                'suggestion': 'Continue monitoring'
            })

        return recommendations

    def get_state_history(self) -> List[Dict]:
        """Get complete state history."""
        return self.state_history

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with metrics
        """
        if len(self.state_history) < 2:
            return {}

        # Extract time series
        times = [h['time'] for h in self.state_history]
        pluri = [h['state']['pluripotency'] for h in self.state_history]
        diff = [h['state']['differentiation'] for h in self.state_history]
        pop = [h['state']['population'] for h in self.state_history]

        metrics = {
            'duration': times[-1] - times[0],
            'pluripotency': {
                'initial': pluri[0],
                'current': pluri[-1],
                'change': pluri[-1] - pluri[0],
                'mean': np.mean(pluri),
                'std': np.std(pluri)
            },
            'differentiation': {
                'initial': diff[0],
                'current': diff[-1],
                'change': diff[-1] - diff[0],
                'mean': np.mean(diff),
                'std': np.std(diff)
            },
            'population': {
                'initial': pop[0],
                'current': pop[-1],
                'fold_change': pop[-1] / pop[0] if pop[0] > 0 else 0,
                'max': max(pop)
            }
        }

        return metrics

    def load_ml_predictor(self, predictor_path: str, model_type: str = 'lstm'):
        """
        Load a trained ML predictor.

        Args:
            predictor_path: Path to saved model checkpoint
            model_type: Type of model ('lstm' or 'transformer')
        """
        import torch
        from ..predictors import LSTMPredictor, TransformerPredictor

        # Create model instance
        if model_type.lower() == 'lstm':
            predictor = LSTMPredictor(input_size=3, output_size=3, config=self.config)
        elif model_type.lower() == 'transformer':
            predictor = TransformerPredictor(input_size=3, output_size=3, config=self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        checkpoint = torch.load(predictor_path, map_location=predictor.device)
        predictor.load_state_dict(checkpoint['model_state_dict'])
        predictor.eval()

        self.predictor = predictor
        logger.info(f"ML predictor loaded from {predictor_path}")
        logger.info(f"Model type: {model_type}, Parameters: {predictor.count_parameters():,}")

    def compare_prediction_methods(
        self,
        horizon: int,
        growth_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compare physics-only, ML-only, and hybrid predictions.

        Args:
            horizon: Prediction horizon in hours
            growth_factors: Growth factor concentrations

        Returns:
            Dictionary with all three predictions for comparison
        """
        if self.predictor is None:
            logger.warning("No ML predictor loaded. Cannot compare methods.")
            return None

        results = {}

        # Physics-only
        logger.info("Computing physics-only prediction...")
        results['physics'] = self.predict(
            horizon=horizon,
            growth_factors=growth_factors,
            use_ml=False
        )

        # ML-only
        logger.info("Computing ML-only prediction...")
        results['ml'] = self.predict(
            horizon=horizon,
            growth_factors=growth_factors,
            use_ml=True,
            fusion_weight=1.0
        )

        # Hybrid (50-50)
        logger.info("Computing hybrid prediction...")
        results['hybrid'] = self.predict(
            horizon=horizon,
            growth_factors=growth_factors,
            use_ml=True,
            fusion_weight=0.5
        )

        logger.info("Comparison complete")
        return results

    def reset(self):
        """Reset digital twin to initial state."""
        self.current_state = None
        self.state_history = []
        self.predictions = []
        self.start_time = None
        self.current_time = 0.0
        self.simulator.reset()
        logger.info("Digital Twin reset")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from src.models.simulators.ipsc_simulator import iPSCDifferentiationSimulator

    # Create simulator and twin
    simulator = iPSCDifferentiationSimulator()
    twin = DigitalTwinEngine(simulator)

    # Initialize
    print("Initializing digital twin...")
    twin.initialize(growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0})

    # Simulate updates over time
    print("\nSimulating 7 days of culture...")
    for day in range(7):
        # Update every 24 hours
        state = twin.update(
            duration=24,
            growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
        )
        print(f"Day {day+1}: P={state['pluripotency']:.3f}, "
              f"D={state['differentiation']:.3f}, N={state['population']:.0f}")

        # Get recommendations
        if day % 2 == 0:
            recs = twin.recommend_action()
            print(f"  Recommendation: {recs['actions'][0]['message']}")

    # Predict future
    print("\nPredicting next 48 hours...")
    prediction = twin.predict(
        horizon=48,
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0}
    )
    final_pred = prediction['predicted_states'][-1]
    print(f"Predicted state: P={final_pred['pluripotency']:.3f}, "
          f"D={final_pred['differentiation']:.3f}")

    # Get metrics
    print("\nMetrics:")
    metrics = twin.get_metrics()
    print(f"Pluripotency change: {metrics['pluripotency']['change']:.3f}")
    print(f"Population fold change: {metrics['population']['fold_change']:.1f}x")
