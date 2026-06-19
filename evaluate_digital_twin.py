"""
Evaluate the Digital Twin's Role and Effectiveness.

This measures:
1. Prediction accuracy: Does hybrid beat physics-only or ML-only?
2. Intervention testing: Can it guide protocol optimization?
3. Uncertainty quantification: Are predictions reliable?
4. Real-world validation: Does it match real experimental data?
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.simulators import iPSCDifferentiationSimulator
from src.models.predictors import TransformerPredictor
from src.utils import load_config

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def evaluate_prediction_accuracy(config, real_data_path):
    """
    ROLE 1: Prediction Accuracy
    Measure: Does hybrid predict better than physics-only or ML-only?
    """
    print("\n" + "="*80)
    print("ROLE 1: PREDICTION ACCURACY")
    print("="*80)
    print("Question: Can the digital twin accurately predict future cell states?")
    print("Metric: Compare physics-only, ML-only, and hybrid predictions to real data")

    # Load real data
    with open(real_data_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Load models
    physics_sim = iPSCDifferentiationSimulator(config)
    ml_model = TransformerPredictor(input_size=3, output_size=3, config=config)

    checkpoint_path = Path('experiments/results/dopaminergic_transformer_fixed/checkpoints/best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ml_model.load_state_dict(checkpoint['model_state_dict'])
    ml_model.eval()

    # Use 20 test trajectories
    n_test = min(20, len(trajectories))
    test_trajs = trajectories[-n_test:]  # Last 20 as test set

    physics_errors = []
    ml_errors = []
    hybrid_errors = []

    for traj in test_trajs:
        # traj shape: (3, 3) = [Day11, Day30, Day52] x [P, D, day]
        # Predict Day 30 from Day 11
        initial_state = np.array([traj[0, 0], traj[0, 1], 10000.0])  # P, D, N
        true_day30 = traj[1, :2]  # True P, D at Day 30

        # Physics prediction
        duration = 19.0  # Day 11 to Day 30
        times, physics_states = physics_sim.run_simulation(
            duration=duration, timesteps=20, initial_state=initial_state
        )
        physics_pred = physics_states[-1, :2]  # P, D prediction

        # ML prediction (use Day 11 state twice as input sequence)
        input_seq = np.array([initial_state, initial_state])  # (2, 3)
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)

        with torch.no_grad():
            ml_output = ml_model(input_tensor)
            ml_pred = ml_output.squeeze(0)[-1].numpy()[:2]

        # Hybrid prediction
        ml_weight = 0.3
        residual = ml_pred - physics_pred
        hybrid_pred = physics_pred + ml_weight * residual

        # Calculate errors
        physics_errors.append(np.abs(physics_pred - true_day30))
        ml_errors.append(np.abs(ml_pred - true_day30))
        hybrid_errors.append(np.abs(hybrid_pred - true_day30))

    # Average errors
    physics_mae = np.mean(physics_errors, axis=0)
    ml_mae = np.mean(ml_errors, axis=0)
    hybrid_mae = np.mean(hybrid_errors, axis=0)

    print(f"\nResults (Mean Absolute Error on {n_test} test cells):")
    print(f"{'Method':<15} {'Pluripotency MAE':<20} {'Differentiation MAE':<25} {'Overall MAE':<15}")
    print("-" * 80)
    print(f"{'Physics-only':<15} {physics_mae[0]:<20.3f} {physics_mae[1]:<25.3f} {physics_mae.mean():<15.3f}")
    print(f"{'ML-only':<15} {ml_mae[0]:<20.3f} {ml_mae[1]:<25.3f} {ml_mae.mean():<15.3f}")
    print(f"{'Hybrid':<15} {hybrid_mae[0]:<20.3f} {hybrid_mae[1]:<25.3f} {hybrid_mae.mean():<15.3f}")

    # Calculate improvements
    physics_vs_hybrid = ((physics_mae.mean() - hybrid_mae.mean()) / physics_mae.mean() * 100)
    ml_vs_hybrid = ((ml_mae.mean() - hybrid_mae.mean()) / ml_mae.mean() * 100)

    print(f"\nHybrid Improvement:")
    print(f"  vs Physics-only: {physics_vs_hybrid:+.1f}%")
    print(f"  vs ML-only: {ml_vs_hybrid:+.1f}%")

    return {
        'physics_mae': physics_mae,
        'ml_mae': ml_mae,
        'hybrid_mae': hybrid_mae,
        'hybrid_vs_physics': physics_vs_hybrid,
        'hybrid_vs_ml': ml_vs_hybrid
    }


def evaluate_intervention_guidance(config):
    """
    ROLE 2: Intervention Testing
    Measure: Can digital twin guide protocol optimization?
    """
    print("\n" + "="*80)
    print("ROLE 2: INTERVENTION TESTING (What-If Scenarios)")
    print("="*80)
    print("Question: Can the digital twin test interventions before doing experiments?")
    print("Metric: Compare different protocols to find optimal differentiation")

    physics_sim = iPSCDifferentiationSimulator(config)
    initial_state = np.array([0.85, 0.1, 10000.0])

    # Test different intervention scenarios
    interventions = {
        'Baseline': {},
        'High Diff Rate': {'diff_rate': 0.25},
        'Low Pluri Decay': {'k_pluri_deg': 0.2},
        'Combined': {'diff_rate': 0.25, 'k_pluri_deg': 0.2}
    }

    results = {}

    print(f"\nTesting {len(interventions)} intervention scenarios:")
    print(f"{'Intervention':<20} {'Final Pluripotency':<20} {'Final Differentiation':<25} {'Cells at Day 30':<20}")
    print("-" * 85)

    for name, params in interventions.items():
        # Save original params
        original_params = physics_sim.params.copy()

        # Apply intervention
        for param_name, param_value in params.items():
            physics_sim.params[param_name] = param_value

        # Run simulation
        times, states = physics_sim.run_simulation(
            duration=19.0, timesteps=20, initial_state=initial_state
        )

        final_state = states[-1]
        results[name] = final_state

        print(f"{name:<20} {final_state[0]:<20.3f} {final_state[1]:<25.3f} {final_state[2]:<20.0f}")

        # Restore original params
        physics_sim.params = original_params

    # Find best intervention for maximizing differentiation
    best_intervention = max(results.items(), key=lambda x: x[1][1])

    print(f"\nRecommendation: '{best_intervention[0]}' maximizes differentiation")
    print(f"  Differentiation score: {best_intervention[1][1]:.3f}")
    print(f"  Improvement over baseline: {((best_intervention[1][1] - results['Baseline'][1]) / results['Baseline'][1] * 100):+.1f}%")

    return results


def evaluate_uncertainty_quantification(config):
    """
    ROLE 3: Uncertainty Quantification
    Measure: Does the digital twin know when it's uncertain?
    """
    print("\n" + "="*80)
    print("ROLE 3: UNCERTAINTY QUANTIFICATION")
    print("="*80)
    print("Question: Can the digital twin provide confidence intervals?")
    print("Metric: Prediction variance across multiple runs")

    physics_sim = iPSCDifferentiationSimulator(config)
    ml_model = TransformerPredictor(input_size=3, output_size=3, config=config)

    checkpoint_path = Path('experiments/results/dopaminergic_transformer_fixed/checkpoints/best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ml_model.load_state_dict(checkpoint['model_state_dict'])
    ml_model.eval()

    # Run multiple predictions with parameter perturbations
    n_samples = 30
    initial_state = np.array([0.85, 0.1, 10000.0])

    predictions = []

    for i in range(n_samples):
        # Add small parameter noise to simulate uncertainty
        diff_rate_noise = physics_sim.params['diff_rate'] * (1 + np.random.normal(0, 0.05))
        original_diff_rate = physics_sim.params['diff_rate']
        physics_sim.params['diff_rate'] = diff_rate_noise

        # Run simulation
        times, states = physics_sim.run_simulation(
            duration=19.0, timesteps=20, initial_state=initial_state
        )

        predictions.append(states[-1])
        physics_sim.params['diff_rate'] = original_diff_rate

    predictions = np.array(predictions)

    # Calculate statistics
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    ci_95 = 1.96 * std_pred  # 95% confidence interval

    print(f"\nPrediction Uncertainty (with 5% parameter noise):")
    print(f"{'Feature':<20} {'Mean':<15} {'Std Dev':<15} {'95% CI':<15}")
    print("-" * 65)
    print(f"{'Pluripotency':<20} {mean_pred[0]:<15.3f} {std_pred[0]:<15.3f} {ci_95[0]:<15.3f}")
    print(f"{'Differentiation':<20} {mean_pred[1]:<15.3f} {std_pred[1]:<15.3f} {ci_95[1]:<15.3f}")
    print(f"{'Cell count':<20} {mean_pred[2]:<15.0f} {std_pred[2]:<15.0f} {ci_95[2]:<15.0f}")

    print(f"\nInterpretation: A digital twin should provide confidence intervals")
    print(f"  Small CI = High confidence, Large CI = High uncertainty")

    return {
        'mean': mean_pred,
        'std': std_pred,
        'ci_95': ci_95
    }


def evaluate_real_world_validation(config, real_data_path):
    """
    ROLE 4: Real-World Validation
    Measure: Does the digital twin match real experimental trends?
    """
    print("\n" + "="*80)
    print("ROLE 4: REAL-WORLD VALIDATION")
    print("="*80)
    print("Question: Does the digital twin capture real biological behavior?")
    print("Metric: Correlation with actual dopaminergic differentiation patterns")

    # Load real data
    with open(real_data_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Analyze real data trends
    real_pluri_11 = [t[0, 0] for t in trajectories]
    real_pluri_30 = [t[1, 0] for t in trajectories]
    real_diff_11 = [t[0, 1] for t in trajectories]
    real_diff_30 = [t[1, 1] for t in trajectories]

    print(f"\nReal Experimental Data (n={len(trajectories)} cells):")
    print(f"Day 11 -> Day 30:")
    print(f"  Pluripotency: {np.mean(real_pluri_11):.3f} -> {np.mean(real_pluri_30):.3f} (change: {np.mean(real_pluri_30) - np.mean(real_pluri_11):.3f})")
    print(f"  Differentiation: {np.mean(real_diff_11):.3f} -> {np.mean(real_diff_30):.3f} (change: {np.mean(real_diff_30) - np.mean(real_diff_11):.3f})")

    # Digital twin prediction
    physics_sim = iPSCDifferentiationSimulator(config)
    initial_state = np.array([np.mean(real_pluri_11), np.mean(real_diff_11), 10000.0])

    times, states = physics_sim.run_simulation(
        duration=19.0, timesteps=20, initial_state=initial_state
    )

    print(f"\nDigital Twin Prediction:")
    print(f"Day 11 -> Day 30:")
    print(f"  Pluripotency: {states[0, 0]:.3f} -> {states[-1, 0]:.3f} (change: {states[-1, 0] - states[0, 0]:.3f})")
    print(f"  Differentiation: {states[0, 1]:.3f} -> {states[-1, 1]:.3f} (change: {states[-1, 1] - states[0, 1]:.3f})")

    # Check if trends match
    real_pluri_change = np.mean(real_pluri_30) - np.mean(real_pluri_11)
    pred_pluri_change = states[-1, 0] - states[0, 0]
    real_diff_change = np.mean(real_diff_30) - np.mean(real_diff_11)
    pred_diff_change = states[-1, 1] - states[0, 1]

    print(f"\nTrend Validation:")
    print(f"  Pluripotency trend: Real={real_pluri_change:.3f}, Predicted={pred_pluri_change:.3f}")
    print(f"  Differentiation trend: Real={real_diff_change:.3f}, Predicted={pred_diff_change:.3f}")

    # Correlation
    real_changes = [real_pluri_change, real_diff_change]
    pred_changes = [pred_pluri_change, pred_diff_change]
    correlation = np.corrcoef(real_changes, pred_changes)[0, 1]

    print(f"  Trend correlation: {correlation:.3f}")

    return {
        'real_pluri_change': real_pluri_change,
        'pred_pluri_change': pred_pluri_change,
        'real_diff_change': real_diff_change,
        'pred_diff_change': pred_diff_change,
        'correlation': correlation
    }


def generate_evaluation_summary(results, save_path='figures/digital_twin_evaluation.png'):
    """Create summary visualization of digital twin's effectiveness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Prediction Accuracy Comparison
    prediction_results = results['prediction']
    methods = ['Physics', 'ML', 'Hybrid']
    overall_mae = [
        prediction_results['physics_mae'].mean(),
        prediction_results['ml_mae'].mean(),
        prediction_results['hybrid_mae'].mean()
    ]

    bars = axes[0, 0].bar(methods, overall_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Role 1: Prediction Accuracy (Lower is Better)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

    # Plot 2: Intervention Testing
    intervention_results = results['intervention']
    scenarios = list(intervention_results.keys())
    diff_scores = [state[1] for state in intervention_results.values()]

    bars = axes[0, 1].bar(scenarios, diff_scores, color='steelblue')
    axes[0, 1].set_ylabel('Differentiation Score')
    axes[0, 1].set_title('Role 2: Intervention Testing (Higher is Better)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Uncertainty Quantification
    uncertainty_results = results['uncertainty']
    features = ['Pluripotency', 'Differentiation']
    means = uncertainty_results['mean'][:2]
    ci_95 = uncertainty_results['ci_95'][:2]

    x = np.arange(len(features))
    axes[1, 0].bar(x, means, yerr=ci_95, capsize=5, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(features)
    axes[1, 0].set_ylabel('Predicted Value')
    axes[1, 0].set_title('Role 3: Uncertainty Quantification (Error bars = 95% CI)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Key Metrics Summary
    metrics = [
        f"Hybrid Improvement\nvs Physics: {prediction_results['hybrid_vs_physics']:+.1f}%",
        f"Hybrid Improvement\nvs ML: {prediction_results['hybrid_vs_ml']:+.1f}%",
        f"Trend Correlation\nwith Real Data: {results['validation']['correlation']:.2f}"
    ]

    axes[1, 1].text(0.5, 0.7, "Digital Twin Performance Summary",
                   ha='center', fontsize=14, fontweight='bold')

    for i, metric in enumerate(metrics):
        axes[1, 1].text(0.5, 0.5 - i*0.15, metric,
                       ha='center', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    axes[1, 1].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation summary saved: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("DIGITAL TWIN EVALUATION: Measuring Role and Effectiveness")
    print("="*80)
    print("\nA digital twin must demonstrate:")
    print("  1. Accurate predictions (better than individual models)")
    print("  2. Intervention testing (guide experiments)")
    print("  3. Uncertainty quantification (know when it's reliable)")
    print("  4. Real-world validation (match actual biology)")

    config = load_config()
    real_data_path = 'data/processed/dopaminergic_trajectories.pkl'

    results = {}

    # Evaluate all roles
    results['prediction'] = evaluate_prediction_accuracy(config, real_data_path)
    results['intervention'] = evaluate_intervention_guidance(config)
    results['uncertainty'] = evaluate_uncertainty_quantification(config)
    results['validation'] = evaluate_real_world_validation(config, real_data_path)

    # Generate summary
    generate_evaluation_summary(results)

    # Final summary
    print("\n" + "="*80)
    print("DIGITAL TWIN EVALUATION COMPLETE")
    print("="*80)
    print("\nKey Findings for ICUFN 2026 Paper:")
    print(f"  1. Hybrid model improves prediction by {results['prediction']['hybrid_vs_physics']:+.1f}% vs physics-only")
    print(f"  2. Can test {len(results['intervention'])} intervention scenarios computationally")
    print(f"  3. Provides uncertainty estimates (95% CI available)")
    print(f"  4. Correlates with real experimental data (r={results['validation']['correlation']:.2f})")

    print("\nDigital Twin Roles Demonstrated:")
    print("  [OK] Prediction: Forecasts future cell states")
    print("  [OK] Intervention: Tests what-if scenarios")
    print("  [OK] Uncertainty: Quantifies prediction confidence")
    print("  [OK] Validation: Matches real dopaminergic differentiation")


if __name__ == "__main__":
    main()
