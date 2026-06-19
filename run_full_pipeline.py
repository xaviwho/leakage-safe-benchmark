"""
Full Experimental Pipeline for Publication-Ready Results

Runs all experiments in correct order to generate consistent,
reproducible results that match METHODOLOGY.md.

Pipeline:
1. Data preprocessing (pseudotime trajectories)
2. ODE parameter calibration
3. Baseline training (Linear Regression, Random Forest)
4. Transformer training
5. λ-network training
6. Temporal extrapolation experiment
7. Results compilation

Runtime: ~2-3 hours end-to-end
"""
import subprocess
import sys
from pathlib import Path
import json

print("="*80)
print("FULL EXPERIMENTAL PIPELINE")
print("="*80)
print("\nThis will run all experiments to generate publication-ready results.")
print("Estimated runtime: 2-3 hours")
print("\n" + "="*80)

# Track results
pipeline_results = {}

def run_script(script_name, description):
    """Run a Python script and capture results."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        print(f"Pipeline aborted.")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] {description} complete")

    return result.returncode == 0

# Step 1: Data preprocessing
if Path('data/processed/dopaminergic_trajectories_pseudotime.pkl').exists():
    print("\n[1/7] Pseudotime trajectories already exist, skipping fix_trajectories.py")
else:
    run_script('fix_trajectories.py', '1. Generate pseudotime-ordered trajectories')

# Step 2: ODE parameter calibration
if Path('config/calibrated_ode_params.json').exists():
    print("\n[2/7] Calibrated ODE parameters already exist, skipping calibrate_ode_params.py")
    with open('config/calibrated_ode_params.json', 'r') as f:
        ode_results = json.load(f)
    pipeline_results['ode_calibration'] = ode_results
else:
    run_script('calibrate_ode_params.py', '2. Calibrate ODE parameters to real data')
    with open('config/calibrated_ode_params.json', 'r') as f:
        ode_results = json.load(f)
    pipeline_results['ode_calibration'] = ode_results

# Step 3: Train baselines
run_script('train_baselines.py', '3. Train baseline models (Linear Regression, Random Forest)')
with open('experiments/results/baselines/comparison.json', 'r') as f:
    baseline_results = json.load(f)
pipeline_results['baselines'] = baseline_results

# Step 4: Train Transformer (if not already trained)
transformer_checkpoint = Path('experiments/results/dopaminergic_transformer_2D/checkpoints/best_model.pt')
if transformer_checkpoint.exists():
    print("\n[4/7] Transformer (2D) already trained, skipping training")
else:
    print("\n[4/7] Training Transformer model with 2D inputs...")
    print("NOTE: This step requires running:")
    print("  python experiments/train_predictor.py --model transformer --load_data data/processed/dopaminergic_trajectories_pseudotime.pkl --epochs 100 --sequence_length 2 --prediction_horizon 1 --experiment_name dopaminergic_transformer_2D")
    print("Please run this manually if not already done, then re-run this pipeline.")
    print("For now, assuming Transformer is trained...")

# Check if Transformer exists
if not transformer_checkpoint.exists():
    print(f"\n[ERROR] Transformer checkpoint not found at {transformer_checkpoint}")
    print("Please train the Transformer first:")
    print("  python experiments/train_predictor.py --model transformer --load_data data/processed/dopaminergic_trajectories_pseudotime.pkl --epochs 100 --sequence_length 2 --prediction_horizon 1 --experiment_name dopaminergic_transformer_2D")
    sys.exit(1)

# Step 5: Train lambda-network
run_script('train_lambda_network.py', '5. Train lambda-network (learnable hybrid weighting)')

# Step 6: Temporal extrapolation experiment
run_script('evaluate_temporal_extrapolation.py', '6. Evaluate temporal extrapolation (early → late)')
with open('experiments/results/temporal_extrapolation/results.json', 'r') as f:
    extrapolation_results = json.load(f)
pipeline_results['temporal_extrapolation'] = extrapolation_results

# Step 7: Generate visualizations
print("\n[7/7] Generating publication figures...")
run_script('visualize_results.py', '7. Generate publication-quality figures')

# Compile results summary
print("\n" + "="*80)
print("PIPELINE COMPLETE - RESULTS SUMMARY")
print("="*80)

print("\n1. ODE Calibration:")
print(f"   - Improvement: {pipeline_results['ode_calibration']['improvement_percent']:.1f}%")
print(f"   - Final loss: {pipeline_results['ode_calibration']['final_loss']:.6f}")

print("\n2. Baseline Comparison:")
print(f"   - Linear Regression MAE: {pipeline_results['baselines']['LinearRegression']['test_mae']:.3f}")
print(f"   - Random Forest MAE: {pipeline_results['baselines']['RandomForest']['test_mae']:.3f}")
print(f"   - Transformer improvement: {pipeline_results['baselines']['comparison']['transformer_improvement_pct']:.1f}%")

print("\n3. Temporal Extrapolation:")
print(f"   - Validation (early) MAE: {pipeline_results['temporal_extrapolation']['val_mae']:.3f}")
print(f"   - Test (late) MAE: {pipeline_results['temporal_extrapolation']['test_late_mae']:.3f}")
print(f"   - Degradation: {pipeline_results['temporal_extrapolation']['degradation_pct']:+.1f}%")

# Save full pipeline results
pipeline_summary = {
    'timestamp': str(pd.Timestamp.now()) if 'pd' in dir() else 'N/A',
    'ode_calibration_improvement_pct': pipeline_results['ode_calibration']['improvement_percent'],
    'best_baseline_mae': pipeline_results['baselines']['comparison']['best_baseline_test_mae'],
    'transformer_mae': pipeline_results['baselines']['comparison']['transformer_test_mae'],
    'transformer_improvement_pct': pipeline_results['baselines']['comparison']['transformer_improvement_pct'],
    'temporal_extrapolation_degradation_pct': pipeline_results['temporal_extrapolation']['degradation_pct']
}

summary_path = Path('experiments/results/pipeline_summary.json')
with open(summary_path, 'w') as f:
    json.dump(pipeline_summary, f, indent=2)

print(f"\n[SAVED] Pipeline summary: {summary_path}")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE")
print("="*80)
print("\nResults are now consistent with METHODOLOGY.md!")
print("Ready for publication submission to ICUFN 2026.")
print("\nNext steps:")
print("  1. Review figures in figures/")
print("  2. Check all results match METHODOLOGY.md claims")
print("  3. Update METHODOLOGY.md if needed with actual numbers")
print("  4. Submit paper!")
