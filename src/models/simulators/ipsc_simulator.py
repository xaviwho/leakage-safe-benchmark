"""
iPSC Differentiation Simulator using ODE-based models.

This module implements a mechanistic model of induced pluripotent stem cell (iPSC)
differentiation based on ordinary differential equations (ODEs).
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class iPSCDifferentiationSimulator:
    """
    Simulates iPSC differentiation dynamics using ODEs.

    The model includes:
    - Pluripotency marker dynamics (e.g., OCT4, NANOG, SOX2)
    - Differentiation marker expression
    - Cell population growth
    - Response to growth factors and culture conditions
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the simulator.

        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}

        # Default parameters
        self.params = {
            # Gene expression parameters
            'k_pluri_synth': 1.0,      # Pluripotency gene synthesis rate
            'k_pluri_deg': 0.5,        # Pluripotency gene degradation rate
            'k_diff_synth': 0.3,       # Differentiation gene synthesis rate
            'k_diff_deg': 0.3,         # Differentiation gene degradation rate

            # Cell dynamics
            'growth_rate': 0.8,        # Cell division rate (per day)
            'death_rate': 0.05,        # Cell death rate (per day)
            'carrying_capacity': 1e6,  # Maximum cell population

            # Differentiation dynamics
            'diff_threshold': 0.3,     # Pluripotency threshold for commitment
            'diff_rate': 0.15,         # Differentiation rate
            'k_basal': 0.05,           # Basal differentiation induction (protocol-driven)

            # Growth factor effects
            'fgf2_effect': 1.5,        # FGF2 enhances pluripotency
            'ra_effect': 2.0,          # Retinoic acid promotes differentiation

            # Hill function parameters
            'hill_coeff': 2.0,         # Cooperativity
            'km': 0.5,                 # Michaelis constant
        }

        # Update with config if provided
        if 'ode_model' in self.config:
            self.params.update(self.config['ode_model'])

        # State variables
        self.state = None
        self.time_points = None
        self.history = []

        logger.info("iPSC Differentiation Simulator initialized")

    def _ode_system(self, state: np.ndarray, t: float, growth_factors: Dict[str, float]) -> np.ndarray:
        """
        Define the ODE system for iPSC differentiation.

        State variables:
        [0] P: Pluripotency markers (OCT4, NANOG, SOX2)
        [1] D: Differentiation markers
        [2] N: Cell population number

        Args:
            state: Current state [P, D, N]
            t: Current time
            growth_factors: Dictionary of growth factor concentrations

        Returns:
            Derivatives [dP/dt, dD/dt, dN/dt]
        """
        P, D, N = state

        # Extract parameters
        k_ps = self.params['k_pluri_synth']
        k_pd = self.params['k_pluri_deg']
        k_ds = self.params['k_diff_synth']
        k_dd = self.params['k_diff_deg']
        r = self.params['growth_rate']
        d = self.params['death_rate']
        K = self.params['carrying_capacity']
        h = self.params['hill_coeff']
        km = self.params['km']

        # Growth factor effects
        fgf2 = growth_factors.get('fgf2', 0.0)
        ra = growth_factors.get('retinoic_acid', 0.0)

        # Hill function for gene regulation
        def hill_activation(x, k, n):
            """Hill activation function"""
            return (x**n) / (k**n + x**n)

        def hill_repression(x, k, n):
            """Hill repression function"""
            return k**n / (k**n + x**n)

        # Pluripotency dynamics
        # - Self-activation (positive feedback)
        # - Repression by differentiation markers
        # - Enhanced by FGF2
        # - Suppressed by retinoic acid
        pluri_activation = hill_activation(P, km, h)
        diff_repression = hill_repression(D, km, h)
        fgf_effect = 1.0 + self.params['fgf2_effect'] * fgf2
        ra_suppression = 1.0 / (1.0 + self.params['ra_effect'] * ra)

        dP_dt = (k_ps * pluri_activation * diff_repression * fgf_effect * ra_suppression
                 - k_pd * P)

        # Differentiation marker dynamics
        # - Basal induction (protocol-driven, allows D to increase from 0)
        # - Activated when pluripotency markers decline
        # - Enhanced by retinoic acid
        # - Self-activation (commitment)
        pluri_loss = hill_repression(P, self.params['diff_threshold'], h)
        ra_induction = 1.0 + self.params['ra_effect'] * ra
        diff_activation = hill_activation(D, km, h)

        dD_dt = (self.params['k_basal'] +  # NEW: basal term allows initiation from D=0
                 k_ds * pluri_loss * ra_induction * (0.5 + 0.5 * diff_activation)
                 - k_dd * D)

        # Cell population dynamics (logistic growth with death)
        # Growth rate decreases as cells differentiate
        effective_growth = r * (1.0 - 0.5 * D)  # Differentiated cells grow slower
        dN_dt = (effective_growth - d) * N * (1 - N / K)

        return np.array([dP_dt, dD_dt, dN_dt])

    def run_simulation(
        self,
        duration: float = 14.0,
        timesteps: int = 100,
        initial_state: Optional[np.ndarray] = None,
        growth_factors: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run differentiation simulation.

        Args:
            duration: Simulation duration in days
            timesteps: Number of time points
            initial_state: Initial conditions [P, D, N]. If None, uses defaults.
            growth_factors: Growth factor concentrations

        Returns:
            Tuple of (time_points, states)
            - time_points: Array of time values
            - states: Array of states at each time point [P, D, N]
        """
        # Set initial conditions
        if initial_state is None:
            P0 = self.config.get('simulation', {}).get('initial_pluripotency', 0.95)
            D0 = 0.05  # Small initial differentiation
            N0 = self.config.get('simulation', {}).get('initial_population', 10000)
            initial_state = np.array([P0, D0, N0])

        # Set growth factors
        if growth_factors is None:
            growth_factors = self.config.get('simulation', {}).get('growth_factors', {
                'fgf2': 1.0,
                'retinoic_acid': 0.0
            })

        # Time points
        self.time_points = np.linspace(0, duration, timesteps)

        # Solve ODE
        logger.info(f"Running simulation for {duration} days with {timesteps} timesteps")
        self.state = odeint(
            self._ode_system,
            initial_state,
            self.time_points,
            args=(growth_factors,)
        )

        # Store history
        self.history.append({
            'time': self.time_points,
            'state': self.state,
            'growth_factors': growth_factors
        })

        logger.info(f"Simulation complete. Final state: P={self.state[-1, 0]:.3f}, "
                   f"D={self.state[-1, 1]:.3f}, N={self.state[-1, 2]:.0f}")

        return self.time_points, self.state

    def _ode_system_pseudotime(self, state: np.ndarray, tau: float) -> np.ndarray:
        """
        Simplified ODE system in pseudotime for 2D state [P, D].

        This formulation treats differentiation as a latent process over
        pseudotime τ ∈ [0, 1], removing explicit cell count dynamics.

        State variables:
        [0] P: Pluripotency markers
        [1] D: Differentiation markers

        Args:
            state: Current state [P, D]
            tau: Current pseudotime

        Returns:
            Derivatives [dP/dτ, dD/dτ]
        """
        P, D = state

        # Extract parameters (now dimensionless, per pseudotime unit)
        k_ps = self.params['k_pluri_synth']
        k_pd = self.params['k_pluri_deg']
        k_ds = self.params['k_diff_synth']
        k_dd = self.params['k_diff_deg']
        diff_rate = self.params['diff_rate']
        k_basal = self.params['k_basal']

        # Simplified dynamics (no Hill functions, focus on core mechanisms)
        # Pluripotency: self-maintenance, degradation, suppression by differentiation
        dP_dtau = k_ps * (1 - P) - k_pd * P - diff_rate * P * D

        # Differentiation: basal induction + activation by P loss + self-activation
        dD_dtau = k_basal + k_ds * P * D + diff_rate * P * D - k_dd * D

        return np.array([dP_dtau, dD_dtau])

    def simulate_pseudotime(
        self,
        tau_span: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation in pseudotime coordinates (2D state space).

        This method aligns with the pseudotime-based trajectory representation
        used in ML models, treating differentiation as a latent continuous process.

        Args:
            tau_span: Pseudotime range (start, end), typically (0, 1)
            n_steps: Number of pseudotime steps
            initial_state: Initial [P, D] state. If None, uses [0.9, 0.1]

        Returns:
            Tuple of (pseudotimes, states)
            - pseudotimes: Array of τ values
            - states: Array of [P, D] states at each pseudotime
        """
        # Set initial conditions (2D state)
        if initial_state is None:
            initial_state = np.array([0.9, 0.1])  # High P, low D
        elif len(initial_state) == 3:
            # If 3D state provided, extract [P, D] only
            initial_state = initial_state[:2]

        # Pseudotime points
        tau_points = np.linspace(tau_span[0], tau_span[1], n_steps)

        # Solve ODE in pseudotime
        logger.info(f"Running pseudotime simulation: τ ∈ [{tau_span[0]}, {tau_span[1]}], {n_steps} steps")
        states = odeint(
            self._ode_system_pseudotime,
            initial_state,
            tau_points
        )

        logger.info(f"Pseudotime simulation complete. Final state: P={states[-1, 0]:.3f}, D={states[-1, 1]:.3f}")

        return tau_points, states

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current state as dictionary.

        Returns:
            Dictionary with current pluripotency, differentiation, and population
        """
        if self.state is None:
            raise ValueError("No simulation has been run yet")

        current = self.state[-1]
        return {
            'pluripotency': float(current[0]),
            'differentiation': float(current[1]),
            'population': float(current[2]),
            'time': float(self.time_points[-1])
        }

    def add_perturbation(
        self,
        growth_factors: Dict[str, float],
        duration: float = 2.0,
        timesteps: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add a perturbation (e.g., change growth factors) and continue simulation.

        Args:
            growth_factors: New growth factor concentrations
            duration: Duration of perturbation in days
            timesteps: Number of time points

        Returns:
            Tuple of (time_points, states) for the perturbation period
        """
        if self.state is None:
            raise ValueError("Must run initial simulation first")

        # Use last state as initial condition
        initial_state = self.state[-1]

        # Continue simulation
        logger.info(f"Applying perturbation: {growth_factors}")
        time_points, states = self.run_simulation(
            duration=duration,
            timesteps=timesteps,
            initial_state=initial_state,
            growth_factors=growth_factors
        )

        return time_points, states

    def reset(self):
        """Reset simulator to initial state."""
        self.state = None
        self.time_points = None
        self.history = []
        logger.info("Simulator reset")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create simulator
    simulator = iPSCDifferentiationSimulator()

    # Run simulation with high FGF2 (maintain pluripotency)
    print("Simulating iPSC maintenance (high FGF2)...")
    time, state = simulator.run_simulation(
        duration=14,
        timesteps=100,
        growth_factors={'fgf2': 1.0, 'retinoic_acid': 0.0}
    )

    print(f"Final state: {simulator.get_current_state()}")

    # Add differentiation stimulus
    print("\nApplying differentiation stimulus (retinoic acid)...")
    time2, state2 = simulator.add_perturbation(
        growth_factors={'fgf2': 0.0, 'retinoic_acid': 1.0},
        duration=7,
        timesteps=50
    )

    print(f"Final state after differentiation: {simulator.get_current_state()}")
