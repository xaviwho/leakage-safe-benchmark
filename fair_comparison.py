"""
Fair, leakage-free single-seed comparison of all models on an IDENTICAL split.

Motivation
----------
The original pipeline split the data inconsistently:
  - train_baselines.py            -> permuted indices (seed 42)         [train + test]
  - experiments/train_predictor.py-> permuted indices (seed 42)         [transformer TRAIN]
  - evaluate_transformer.py       -> CONTIGUOUS slice trajectories[170:] [transformer TEST]  <-- BUG
  - evaluate_ode_baseline.py      -> CONTIGUOUS slice trajectories[170:] [ODE TEST]

Because the transformer was TRAINED on 140 *permuted* indices but TESTED on the
contiguous last-30 block, ~70% of its "test" trajectories were actually in its
training set -> data leakage that inflates the transformer's reported MAE (0.081),
while the baselines (0.100) were scored leakage-free. The headline "+19.4%" is
therefore not an apples-to-apples comparison.

This script fixes that: ONE canonical split (by permuted indices), used identically
by Ridge, Random Forest, a freshly-retrained Transformer, and the calibrated ODE.
The Transformer is retrained on the canonical train set so its test set is genuinely
held out.

Run:
    python fair_comparison.py            # seed 42 (default)
    python fair_comparison.py --seed 7

This is the seed=42 case that Phase 1 will wrap in a multi-seed loop.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.models.predictors import TransformerPredictor
from src.models.predictors.trainer import PredictorTrainer, CellStateDataset
from src.models.simulators import iPSCDifferentiationSimulator
from src.utils import load_config


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_split(n_total: int, seed: int):
    """The ONE split every model uses. Permuted indices, 70/15/15."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )


def make_xy(trajs):
    """[D11, D30] -> D52 as flat vectors for sklearn baselines."""
    X = np.array([t[:2].flatten() for t in trajs])
    y = np.array([t[2] for t in trajs])
    return X, y


def per_feature_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mae_P = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_D = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    return float(mae), float(mae_P), float(mae_D)


def quantify_leakage(idx_train, n_total, seed):
    """Show how much the OLD buggy contiguous test set overlapped the train set."""
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    buggy_test = set(range(n_train + n_val, n_total))  # trajectories[170:]
    train_set = set(idx_train.tolist())
    overlap = buggy_test & train_set
    return len(overlap), len(buggy_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--data",
        type=str,
        default="data/processed/dopaminergic_trajectories_pseudotime.pkl",
        help="Frozen trajectory pkl (kept fixed so results are comparable to the claimed numbers).",
    )
    args = ap.parse_args()

    print("=" * 80)
    print(f"FAIR LEAKAGE-FREE COMPARISON  (seed={args.seed})")
    print("=" * 80)

    set_all_seeds(args.seed)

    # --- Load frozen trajectories (same artifact that produced the claimed numbers) ---
    with open(args.data, "rb") as f:
        trajectories = np.array(pickle.load(f))
    n_total = len(trajectories)
    print(f"\nLoaded {n_total} trajectories, shape {trajectories.shape}")

    # --- ONE canonical split for ALL models ---
    idx_tr, idx_va, idx_te = canonical_split(n_total, args.seed)
    tr, va, te = trajectories[idx_tr], trajectories[idx_va], trajectories[idx_te]
    print(f"Canonical split: {len(tr)} train / {len(va)} val / {len(te)} test (permuted, seed {args.seed})")

    # --- Quantify the original leakage for the report ---
    n_leak, n_buggy = quantify_leakage(idx_tr, n_total, args.seed)
    print(f"\n[leakage check] Old buggy transformer test = contiguous trajectories[170:] ({n_buggy} trajs).")
    print(f"                {n_leak}/{n_buggy} of those were in the transformer's TRAINING set -> leakage.")

    results = {}

    # --- Baselines: Ridge + Random Forest ---
    X_tr, y_tr = make_xy(tr)
    X_te, y_te = make_xy(te)

    ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    mae, mae_P, mae_D = per_feature_mae(y_te, ridge.predict(X_te))
    results["Ridge"] = {"test_mae": mae, "test_mae_P": mae_P, "test_mae_D": mae_D}
    print(f"\nRidge            test MAE={mae:.4f} (P={mae_P:.4f}, D={mae_D:.4f})")

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=args.seed, n_jobs=-1,
    ).fit(X_tr, y_tr)
    mae, mae_P, mae_D = per_feature_mae(y_te, rf.predict(X_te))
    results["RandomForest"] = {"test_mae": mae, "test_mae_P": mae_P, "test_mae_D": mae_D}
    print(f"RandomForest     test MAE={mae:.4f} (P={mae_P:.4f}, D={mae_D:.4f})")

    # --- Transformer: RETRAINED on canonical train, evaluated on canonical test ---
    config = load_config()
    train_ds = CellStateDataset(tr, sequence_length=2, prediction_horizon=1)
    val_ds = CellStateDataset(va, sequence_length=2, prediction_horizon=1)
    test_ds = CellStateDataset(te, sequence_length=2, prediction_horizon=1)

    model = TransformerPredictor(input_size=2, output_size=2, config=config)
    exp_dir = Path(f"experiments/results/fair_seed{args.seed}_transformer")
    trainer = PredictorTrainer(model=model, config=config, experiment_dir=exp_dir)
    trainer.epochs = args.epochs
    print(f"\nTraining Transformer ({model.count_parameters():,} params) on canonical train...")
    trainer.train(train_ds, val_ds, save_best=True)

    # reload best checkpoint, evaluate on held-out test
    best = torch.load(exp_dir / "checkpoints" / "best_model.pt", map_location=model.device)
    model.load_state_dict(best["model_state_dict"])
    m = trainer.evaluate(test_ds, return_predictions=True)
    preds = m["predictions"].reshape(-1, 2)
    targs = m["targets"].reshape(-1, 2)
    mae, mae_P, mae_D = per_feature_mae(targs, preds)
    results["Transformer"] = {"test_mae": mae, "test_mae_P": mae_P, "test_mae_D": mae_D,
                              "best_epoch": int(best["epoch"])}
    print(f"Transformer      test MAE={mae:.4f} (P={mae_P:.4f}, D={mae_D:.4f})")

    # --- Calibrated ODE on the SAME canonical test set ---
    params_path = Path("config/calibrated_ode_params.json")
    sim = iPSCDifferentiationSimulator(config)
    if params_path.exists():
        cal = json.loads(params_path.read_text())
        for k in ("k_pluri_deg", "k_diff_synth", "diff_rate", "k_basal"):
            sim.params[k] = cal[k]

    err_P, err_D = [], []
    for traj in te:
        for (src, tgt, dur, steps) in ((traj[0], traj[1], 19.0, 20), (traj[1], traj[2], 22.0, 23)):
            init = np.array([src[0], src[1], 50000.0])
            _, states = sim.run_simulation(duration=dur, timesteps=steps, initial_state=init)
            pred = np.clip(states[-1, :2], 0.0, 1.0)
            err_P.append(abs(pred[0] - tgt[0]))
            err_D.append(abs(pred[1] - tgt[1]))
    mae_P, mae_D = float(np.mean(err_P)), float(np.mean(err_D))
    mae = (mae_P + mae_D) / 2.0
    results["ODE"] = {"test_mae": mae, "test_mae_P": mae_P, "test_mae_D": mae_D}
    print(f"ODE (calibrated) test MAE={mae:.4f} (P={mae_P:.4f}, D={mae_D:.4f})")

    # --- Summary ---
    best_baseline = min(results["Ridge"]["test_mae"], results["RandomForest"]["test_mae"])
    tf = results["Transformer"]["test_mae"]
    improvement = (best_baseline - tf) / best_baseline * 100.0
    results["summary"] = {
        "seed": args.seed,
        "best_baseline_mae": best_baseline,
        "transformer_mae": tf,
        "transformer_improvement_pct": improvement,
    }

    print("\n" + "=" * 80)
    print("SUMMARY (all models on the IDENTICAL leakage-free test set)")
    print("=" * 80)
    for name in ("Ridge", "RandomForest", "Transformer", "ODE"):
        print(f"  {name:<16} {results[name]['test_mae']:.4f}")
    print(f"\n  Transformer vs best baseline: {improvement:+.1f}%")
    print(f"  (claimed in METHODOLOGY.md: +19.4% on the leaky split)")

    out = Path(f"experiments/results/fair_comparison_seed{args.seed}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
