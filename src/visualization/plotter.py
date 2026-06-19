"""
Visualization utilities for Digital Twin.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DigitalTwinPlotter:
    """Visualization utilities for digital twin results."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize plotter.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
            plt.style.use('default')

    def plot_trajectory(
        self,
        time_points: np.ndarray,
        states: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Cell State Trajectory",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot cell state trajectory over time.

        Args:
            time_points: Time points array
            states: State array [T, N] where T=time points, N=state variables
            labels: Labels for each state variable
            title: Plot title
            save_path: Path to save figure (if None, displays only)

        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = [f"State {i}" for i in range(states.shape[1])]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Pluripotency
        axes[0].plot(time_points, states[:, 0], 'b-', linewidth=2, label='Pluripotency')
        axes[0].set_xlabel('Time (days)')
        axes[0].set_ylabel('Pluripotency Markers')
        axes[0].set_title('Pluripotency (OCT4, NANOG, SOX2)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        # Differentiation
        axes[1].plot(time_points, states[:, 1], 'r-', linewidth=2, label='Differentiation')
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Differentiation Markers')
        axes[1].set_title('Differentiation Markers')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        # Population
        axes[2].plot(time_points, states[:, 2], 'g-', linewidth=2, label='Population')
        axes[2].set_xlabel('Time (days)')
        axes[2].set_ylabel('Cell Number')
        axes[2].set_title('Cell Population')
        axes[2].grid(True, alpha=0.3)
        axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_phase_space(
        self,
        states: np.ndarray,
        title: str = "Phase Space (Pluripotency vs Differentiation)",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot phase space trajectory.

        Args:
            states: State array [T, N]
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot trajectory
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7)

        # Mark start and end
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=12, label='End', zorder=5)

        # Add arrows to show direction
        n_arrows = 5
        arrow_indices = np.linspace(10, len(states)-10, n_arrows, dtype=int)
        for idx in arrow_indices:
            if idx < len(states) - 1:
                dx = states[idx+1, 0] - states[idx, 0]
                dy = states[idx+1, 1] - states[idx, 1]
                ax.arrow(states[idx, 0], states[idx, 1], dx, dy,
                        head_width=0.02, head_length=0.02, fc='black', ec='black', alpha=0.5)

        ax.set_xlabel('Pluripotency Markers', fontsize=12)
        ax.set_ylabel('Differentiation Markers', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_prediction(
        self,
        history_time: np.ndarray,
        history_states: np.ndarray,
        prediction_time: np.ndarray,
        prediction_states: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        title: str = "Digital Twin Prediction",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot historical data with predictions.

        Args:
            history_time: Historical time points
            history_states: Historical states
            prediction_time: Predicted time points
            prediction_states: Predicted states
            uncertainty: Uncertainty bounds (if available)
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Pluripotency
        axes[0].plot(history_time, history_states[:, 0], 'b-', linewidth=2,
                    label='Historical', marker='o', markersize=4)
        axes[0].plot(prediction_time, prediction_states[:, 0], 'b--', linewidth=2,
                    label='Predicted', marker='s', markersize=4)

        if uncertainty is not None:
            axes[0].fill_between(prediction_time,
                                prediction_states[:, 0] - uncertainty[:, 0],
                                prediction_states[:, 0] + uncertainty[:, 0],
                                alpha=0.2, color='blue', label='Uncertainty')

        axes[0].axvline(history_time[-1], color='gray', linestyle=':', linewidth=2,
                       label='Prediction Start')
        axes[0].set_ylabel('Pluripotency Markers', fontsize=11)
        axes[0].set_title('Pluripotency Prediction', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        # Differentiation
        axes[1].plot(history_time, history_states[:, 1], 'r-', linewidth=2,
                    label='Historical', marker='o', markersize=4)
        axes[1].plot(prediction_time, prediction_states[:, 1], 'r--', linewidth=2,
                    label='Predicted', marker='s', markersize=4)

        if uncertainty is not None:
            axes[1].fill_between(prediction_time,
                                prediction_states[:, 1] - uncertainty[:, 1],
                                prediction_states[:, 1] + uncertainty[:, 1],
                                alpha=0.2, color='red', label='Uncertainty')

        axes[1].axvline(history_time[-1], color='gray', linestyle=':', linewidth=2,
                       label='Prediction Start')
        axes[1].set_xlabel('Time (days)', fontsize=11)
        axes[1].set_ylabel('Differentiation Markers', fontsize=11)
        axes[1].set_title('Differentiation Prediction', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_comparison(
        self,
        time_points: np.ndarray,
        states_list: List[np.ndarray],
        labels: List[str],
        title: str = "Protocol Comparison",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Compare multiple simulation runs.

        Args:
            time_points: Time points (same for all runs)
            states_list: List of state arrays
            labels: Labels for each run
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(states_list)))

        # Pluripotency
        for states, label, color in zip(states_list, labels, colors):
            axes[0, 0].plot(time_points, states[:, 0], linewidth=2,
                          label=label, color=color)
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Pluripotency Markers')
        axes[0, 0].set_title('Pluripotency Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Differentiation
        for states, label, color in zip(states_list, labels, colors):
            axes[0, 1].plot(time_points, states[:, 1], linewidth=2,
                          label=label, color=color)
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Differentiation Markers')
        axes[0, 1].set_title('Differentiation Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Population
        for states, label, color in zip(states_list, labels, colors):
            axes[1, 0].plot(time_points, states[:, 2], linewidth=2,
                          label=label, color=color)
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Cell Number')
        axes[1, 0].set_title('Population Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # Phase space
        for states, label, color in zip(states_list, labels, colors):
            axes[1, 1].plot(states[:, 0], states[:, 1], linewidth=2,
                          label=label, color=color, alpha=0.7)
            axes[1, 1].plot(states[0, 0], states[0, 1], 'o', color=color, markersize=8)
            axes[1, 1].plot(states[-1, 0], states[-1, 1], 's', color=color, markersize=8)
        axes[1, 1].set_xlabel('Pluripotency')
        axes[1, 1].set_ylabel('Differentiation')
        axes[1, 1].set_title('Phase Space')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    t = np.linspace(0, 14, 100)
    P = 0.9 * np.exp(-0.2 * t)
    D = 1 - 0.95 * np.exp(-0.3 * t)
    N = 10000 * np.exp(0.5 * t) / (1 + 0.1 * np.exp(0.5 * t))
    states = np.column_stack([P, D, N])

    # Create plotter
    plotter = DigitalTwinPlotter()

    # Plot trajectory
    print("Creating trajectory plot...")
    plotter.plot_trajectory(t, states, save_path="trajectory_example.png")

    # Plot phase space
    print("Creating phase space plot...")
    plotter.plot_phase_space(states, save_path="phase_space_example.png")

    print("Plots created successfully!")
    plt.show()
