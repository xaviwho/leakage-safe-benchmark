"""
Stochastic iPSC Differentiation Simulator.

Adds stochasticity to the ODE model through:
1. Stochastic differential equations (SDEs)
2. Cell-to-cell variability
3. Intrinsic and extrinsic noise
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional
import logging

from .ipsc_simulator import iPSCDifferentiationSimulator

logger = logging.getLogger(__name__)


class StochasticiPSCSimulator(iPSCDifferentiationSimulator):
    """
    Stochastic version of iPSC differentiation simulator.

    Adds Gaussian noise to the deterministic ODE dynamics to model:
    - Intrinsic noise (gene expression variability)
    - Extrinsic noise (environmental fluctuations)
    - Cell-to-cell heterogeneity
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize stochastic simulator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Noise parameters
        self.noise_params = {
            'intrinsic_noise': 0.05,  # Gene expression noise
            'extrinsic_noise': 0.02,  # Environmental noise
            'measurement_noise': 0.01  # Measurement uncertainty
        }

        # Update from config if provided
        if 'stochastic_model' in self.config:
            self.noise_params.update(self.config['stochastic_model'])

        logger.info("Stochastic iPSC Simulator initialized")
        logger.info(f"  Intrinsic noise: {self.noise_params['intrinsic_noise']}")
        logger.info(f"  Extrinsic noise: {self.noise_params['extrinsic_noise']}")

    def run_simulation(
        self,
        duration: float = 14.0,
        timesteps: int = 100,
        initial_state: Optional[np.ndarray] = None,
        growth_factors: Optional[Dict[str, float]] = None,
        add_noise: bool = True,
        n_realizations: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run stochastic differentiation simulation.

        Args:
            duration: Simulation duration in days
            timesteps: Number of time points
            initial_state: Initial conditions [P, D, N]
            growth_factors: Growth factor concentrations
            add_noise: Whether to add stochastic noise
            n_realizations: Number of stochastic realizations (for ensemble)

        Returns:
            Tuple of (time_points, states)
            If n_realizations > 1, states has shape (timesteps, n_realizations, 3)
        """
        if n_realizations == 1:
            # Single realization
            return self._run_single_stochastic(
                duration, timesteps, initial_state, growth_factors, add_noise
            )
        else:
            # Multiple realizations for ensemble
            return self._run_ensemble(
                duration, timesteps, initial_state, growth_factors, add_noise, n_realizations
            )

    def _run_single_stochastic(
        self,
        duration: float,
        timesteps: int,
        initial_state: Optional[np.ndarray],
        growth_factors: Optional[Dict[str, float]],
        add_noise: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single stochastic realization."""
        # Run deterministic ODE first
        time_points, states = super().run_simulation(
            duration=duration,
            timesteps=timesteps,
            initial_state=initial_state,
            growth_factors=growth_factors
        )

        if not add_noise:
            return time_points, states

        # Add stochastic noise using Euler-Maruyama method
        dt = time_points[1] - time_points[0]
        noisy_states = np.copy(states)

        for i in range(1, len(time_points)):
            # Intrinsic noise (proportional to state)
            intrinsic = self.noise_params['intrinsic_noise'] * np.sqrt(dt) * np.random.randn(3)
            intrinsic = intrinsic * noisy_states[i] / (noisy_states[i] + 0.1)  # State-dependent

            # Extrinsic noise (constant)
            extrinsic = self.noise_params['extrinsic_noise'] * np.sqrt(dt) * np.random.randn(3)

            # Add noise
            noisy_states[i] = noisy_states[i] + intrinsic + extrinsic

            # Enforce constraints
            noisy_states[i, 0] = np.clip(noisy_states[i, 0], 0, 1)  # P in [0, 1]
            noisy_states[i, 1] = np.clip(noisy_states[i, 1], 0, 1)  # D in [0, 1]
            noisy_states[i, 2] = np.maximum(noisy_states[i, 2], 100)  # N > 0

        # Add measurement noise
        if self.noise_params['measurement_noise'] > 0:
            measurement_noise = self.noise_params['measurement_noise'] * np.random.randn(*noisy_states.shape)
            noisy_states = noisy_states + measurement_noise

            # Enforce constraints again
            noisy_states[:, 0] = np.clip(noisy_states[:, 0], 0, 1)
            noisy_states[:, 1] = np.clip(noisy_states[:, 1], 0, 1)
            noisy_states[:, 2] = np.maximum(noisy_states[:, 2], 100)

        return time_points, noisy_states

    def _run_ensemble(
        self,
        duration: float,
        timesteps: int,
        initial_state: Optional[np.ndarray],
        growth_factors: Optional[Dict[str, float]],
        add_noise: bool,
        n_realizations: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ensemble of stochastic realizations.

        Returns:
            time_points: Array of time points
            states: Array of shape (timesteps, n_realizations, 3)
        """
        logger.info(f"Running ensemble of {n_realizations} stochastic realizations...")

        ensemble_states = []

        for i in range(n_realizations):
            time_points, states = self._run_single_stochastic(
                duration, timesteps, initial_state, growth_factors, add_noise
            )
            ensemble_states.append(states)

        # Stack realizations
        ensemble_states = np.stack(ensemble_states, axis=1)  # (timesteps, n_realizations, 3)

        logger.info(f"Ensemble simulation complete")

        return time_points, ensemble_states

    def get_ensemble_statistics(
        self,
        ensemble_states: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute statistics across ensemble.

        Args:
            ensemble_states: Array of shape (timesteps, n_realizations, 3)

        Returns:
            Dictionary with mean, std, median, and quantiles
        """
        stats = {
            'mean': np.mean(ensemble_states, axis=1),
            'std': np.std(ensemble_states, axis=1),
            'median': np.median(ensemble_states, axis=1),
            'q25': np.percentile(ensemble_states, 25, axis=1),
            'q75': np.percentile(ensemble_states, 75, axis=1),
            'q05': np.percentile(ensemble_states, 5, axis=1),
            'q95': np.percentile(ensemble_states, 95, axis=1)
        }

        return stats


class GillespieSimulator:
    """
    Gillespie algorithm for exact stochastic simulation.

    Models individual molecular reactions and cell division/death events
    at the single-cell level.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Gillespie simulator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Reaction rates
        self.rates = {
            'pluri_synthesis': 1.0,
            'pluri_degradation': 0.5,
            'diff_synthesis': 0.3,
            'diff_degradation': 0.3,
            'cell_division': 0.8,
            'cell_death': 0.05
        }

        logger.info("Gillespie Simulator initialized")

    def run_simulation(
        self,
        duration: float = 14.0,
        initial_molecules: Optional[Dict[str, int]] = None,
        growth_factors: Optional[Dict[str, float]] = None,
        max_steps: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Gillespie simulation.

        Args:
            duration: Simulation duration in days
            initial_molecules: Initial molecule counts
            growth_factors: Growth factor concentrations
            max_steps: Maximum number of reaction steps

        Returns:
            Tuple of (times, states)
        """
        # Initialize
        if initial_molecules is None:
            initial_molecules = {
                'P_molecules': 1000,  # Pluripotency markers
                'D_molecules': 50,    # Differentiation markers
                'cells': 100          # Number of cells
            }

        state = initial_molecules.copy()
        t = 0.0
        times = [t]
        states = [self._state_to_array(state)]

        step = 0

        while t < duration and step < max_steps:
            # Compute propensities
            propensities = self._compute_propensities(state, growth_factors)
            total_propensity = sum(propensities.values())

            if total_propensity == 0:
                break

            # Sample time to next reaction
            dt = np.random.exponential(1 / total_propensity)
            t += dt

            if t > duration:
                break

            # Select reaction
            reaction = self._select_reaction(propensities, total_propensity)

            # Execute reaction
            state = self._execute_reaction(state, reaction)

            # Record state
            times.append(t)
            states.append(self._state_to_array(state))

            step += 1

        times = np.array(times)
        states = np.array(states)

        logger.info(f"Gillespie simulation: {step} steps, t={t:.2f} days")

        return times, states

    def _compute_propensities(
        self,
        state: Dict[str, int],
        growth_factors: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute reaction propensities."""
        P = state['P_molecules']
        D = state['D_molecules']
        N = state['cells']

        gf = growth_factors or {'fgf2': 1.0, 'retinoic_acid': 0.0}

        propensities = {
            'pluri_synthesis': self.rates['pluri_synthesis'] * N * (1 + gf['fgf2']),
            'pluri_degradation': self.rates['pluri_degradation'] * P,
            'diff_synthesis': self.rates['diff_synthesis'] * N * (1 + gf['retinoic_acid']),
            'diff_degradation': self.rates['diff_degradation'] * D,
            'cell_division': self.rates['cell_division'] * N,
            'cell_death': self.rates['cell_death'] * N
        }

        return propensities

    def _select_reaction(
        self,
        propensities: Dict[str, float],
        total_propensity: float
    ) -> str:
        """Select next reaction based on propensities."""
        r = np.random.uniform(0, total_propensity)
        cumsum = 0

        for reaction, prop in propensities.items():
            cumsum += prop
            if r < cumsum:
                return reaction

        return list(propensities.keys())[-1]

    def _execute_reaction(self, state: Dict[str, int], reaction: str) -> Dict[str, int]:
        """Execute the selected reaction."""
        new_state = state.copy()

        if reaction == 'pluri_synthesis':
            new_state['P_molecules'] += 1
        elif reaction == 'pluri_degradation':
            new_state['P_molecules'] = max(0, new_state['P_molecules'] - 1)
        elif reaction == 'diff_synthesis':
            new_state['D_molecules'] += 1
        elif reaction == 'diff_degradation':
            new_state['D_molecules'] = max(0, new_state['D_molecules'] - 1)
        elif reaction == 'cell_division':
            new_state['cells'] += 1
        elif reaction == 'cell_death':
            new_state['cells'] = max(1, new_state['cells'] - 1)

        return new_state

    def _state_to_array(self, state: Dict[str, int]) -> np.ndarray:
        """Convert state dict to array [P, D, N]."""
        # Normalize by cell number to get concentrations
        N = state['cells']
        P = state['P_molecules'] / (N * 1000)  # Normalize
        D = state['D_molecules'] / (N * 100)   # Normalize

        return np.array([P, D, N])
