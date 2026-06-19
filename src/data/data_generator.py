"""
Data generator using ODE simulator for creating training data.

Generates diverse cell differentiation trajectories by varying:
- Initial conditions
- Growth factor protocols
- Model parameters
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic cell state trajectories using ODE simulator.

    Creates diverse training data by varying:
    - Initial conditions (P, D, N)
    - Growth factor concentrations
    - Temporal protocols (changing growth factors over time)
    - Model parameters (sampling from distributions)
    """

    def __init__(self, simulator, config: Optional[Dict] = None):
        """
        Initialize data generator.

        Args:
            simulator: iPSCDifferentiationSimulator instance
            config: Configuration dictionary
        """
        self.simulator = simulator
        self.config = config or {}

    def generate_random_protocol(self, duration: int = 14) -> List[Dict[str, float]]:
        """
        Generate a random growth factor protocol.

        Args:
            duration: Total duration in days

        Returns:
            List of growth factor dictionaries
        """
        n_phases = np.random.randint(1, 4)  # 1-3 phases
        phase_duration = duration // n_phases

        protocol = []
        for _ in range(n_phases):
            # Random growth factor concentrations
            gf = {
                'fgf2': np.random.uniform(0.0, 1.0),
                'retinoic_acid': np.random.uniform(0.0, 1.0)
            }
            protocol.append((phase_duration, gf))

        return protocol

    def generate_trajectory(
        self,
        protocol: Optional[List[Tuple[int, Dict[str, float]]]] = None,
        initial_state: Optional[np.ndarray] = None,
        timesteps: int = 100,
        add_noise: bool = True,
        noise_level: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single trajectory.

        Args:
            protocol: List of (duration, growth_factors) tuples
            initial_state: Initial conditions [P, D, N]
            timesteps: Number of time points
            add_noise: Whether to add measurement noise
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Tuple of (time_points, states)
        """
        # Default protocol: random
        if protocol is None:
            protocol = self.generate_random_protocol()

        # Random initial conditions if not provided
        if initial_state is None:
            P0 = np.random.uniform(0.85, 0.99)  # High pluripotency
            D0 = np.random.uniform(0.01, 0.15)  # Low differentiation
            N0 = np.random.uniform(5000, 15000)  # Variable population
            initial_state = np.array([P0, D0, N0])

        # Reset simulator
        self.simulator.reset()

        # Run sequential protocol
        all_times = []
        all_states = []
        current_time = 0

        for duration, growth_factors in protocol:
            time_points, states = self.simulator.run_simulation(
                duration=duration,
                timesteps=timesteps // len(protocol),
                initial_state=initial_state,
                growth_factors=growth_factors
            )

            # Shift time points
            time_points = time_points + current_time
            all_times.append(time_points)
            all_states.append(states)

            # Update for next phase
            initial_state = states[-1]
            current_time = time_points[-1]

        # Concatenate all phases
        time_points = np.concatenate(all_times)
        states = np.concatenate(all_states)

        # Add measurement noise
        if add_noise:
            noise = np.random.normal(0, noise_level, states.shape)
            states = states + noise
            # Clip to valid ranges
            states[:, 0] = np.clip(states[:, 0], 0, 1)  # P
            states[:, 1] = np.clip(states[:, 1], 0, 1)  # D
            states[:, 2] = np.clip(states[:, 2], 0, None)  # N (positive)

        return time_points, states

    def generate_dataset(
        self,
        n_trajectories: int = 1000,
        save_path: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a complete dataset of trajectories.

        Args:
            n_trajectories: Number of trajectories to generate
            save_path: Path to save dataset (if None, doesn't save)
            **kwargs: Additional arguments for generate_trajectory

        Returns:
            List of (time_points, states) tuples
        """
        logger.info(f"Generating {n_trajectories} trajectories...")

        dataset = []
        for i in tqdm(range(n_trajectories), desc="Generating trajectories"):
            time_points, states = self.generate_trajectory(**kwargs)
            dataset.append((time_points, states))

        logger.info(f"Generated {len(dataset)} trajectories")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

            logger.info(f"Dataset saved to {save_path}")

        return dataset

    def generate_protocol_specific_dataset(
        self,
        protocol_types: List[str],
        n_per_type: int = 100,
        save_path: Optional[str] = None
    ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate datasets for specific protocol types.

        Args:
            protocol_types: List of protocol types to generate
                Options: 'pluripotency', 'differentiation', 'sequential', 'random'
            n_per_type: Number of trajectories per protocol type
            save_path: Path to save dataset

        Returns:
            Dictionary mapping protocol type to list of trajectories
        """
        datasets = {}

        for ptype in protocol_types:
            logger.info(f"Generating {ptype} protocol trajectories...")
            trajectories = []

            for _ in tqdm(range(n_per_type), desc=f"{ptype} protocol"):
                if ptype == 'pluripotency':
                    # High FGF2, maintain pluripotency
                    protocol = [
                        (14, {'fgf2': np.random.uniform(0.8, 1.0), 'retinoic_acid': 0.0})
                    ]
                elif ptype == 'differentiation':
                    # High RA, induce differentiation
                    protocol = [
                        (14, {'fgf2': 0.0, 'retinoic_acid': np.random.uniform(0.8, 1.0)})
                    ]
                elif ptype == 'sequential':
                    # Two-phase: expansion then differentiation
                    protocol = [
                        (7, {'fgf2': np.random.uniform(0.8, 1.0), 'retinoic_acid': 0.0}),
                        (7, {'fgf2': 0.0, 'retinoic_acid': np.random.uniform(0.7, 1.0)})
                    ]
                elif ptype == 'random':
                    # Random protocol
                    protocol = self.generate_random_protocol()
                else:
                    raise ValueError(f"Unknown protocol type: {ptype}")

                time_points, states = self.generate_trajectory(protocol=protocol)
                trajectories.append((time_points, states))

            datasets[ptype] = trajectories

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(datasets, f)

            logger.info(f"Protocol-specific dataset saved to {save_path}")

        return datasets

    def generate_validation_set(
        self,
        n_trajectories: int = 100,
        save_path: Optional[str] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a validation set with known protocols.

        Args:
            n_trajectories: Number of trajectories
            save_path: Path to save dataset

        Returns:
            List of trajectories
        """
        # Use fixed protocols for validation
        protocols = [
            [(14, {'fgf2': 1.0, 'retinoic_acid': 0.0})],  # Pure pluripotency
            [(14, {'fgf2': 0.0, 'retinoic_acid': 1.0})],  # Pure differentiation
            [(7, {'fgf2': 1.0, 'retinoic_acid': 0.0}),
             (7, {'fgf2': 0.0, 'retinoic_acid': 1.0})],   # Sequential
        ]

        dataset = []
        for _ in tqdm(range(n_trajectories), desc="Generating validation set"):
            protocol = protocols[np.random.choice(len(protocols))]
            time_points, states = self.generate_trajectory(
                protocol=protocol,
                add_noise=True,
                noise_level=0.01  # Lower noise for validation
            )
            dataset.append((time_points, states))

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

            logger.info(f"Validation set saved to {save_path}")

        return dataset


def load_dataset(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a previously generated dataset.

    Args:
        path: Path to dataset pickle file

    Returns:
        List of (time_points, states) tuples
    """
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    logger.info(f"Loaded dataset from {path} with {len(dataset)} trajectories")
    return dataset
