"""
Phase 1 - Multi-seed statistical rigor (the decision-gate experiment).

For each seed we perturb BOTH sources of randomness (plan Sec 2.1):
  1. Pseudo-trajectory construction  -> which cells get sampled per timepoint
  2. Model training                  -> Transformer init/dropout/batch order, RF bootstrap

All models share ONE leakage-free split per seed. Marker normalization bounds are
computed from TRAINING cells only and applied (with clipping) to val/test, so there
is no train->test leakage through the [0,1] scaling either.

We log per-seed test MAE for Ridge, RandomForest, and the Transformer, then run paired
statistics (bootstrap CI + Wilcoxon + Cohen's d_z, Holm-corrected) comparing each
baseline to the Transformer. This decides whether the Transformer advantage is real.

    python run_multiseed.py --seeds 20
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Fair RF baseline: tune depth/leaf per seed by 3-fold CV on the training set.
# A fixed depth-10 RF overfits this small, near-linear task and is an unfair
# strawman; CV-tuning lets RF regularize to its data-appropriate complexity
# (it then ties Ridge), matching the fairness standard applied to the ODE.
RF_GRID = {"n_estimators": [300], "max_depth": [2, 3, 5, None],
           "min_samples_leaf": [1, 5, 10]}


def tuned_rf(seed):
    return GridSearchCV(RandomForestRegressor(random_state=seed, n_jobs=-1),
                        RF_GRID, scoring="neg_mean_absolute_error", cv=3)

from src.models.predictors import TransformerPredictor
from src.utils import load_config

PLURI_GENES = ["POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1"]
DIFF_GENES = ["TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2"]
TIMEPOINTS = ["D11", "D30", "D52"]
N_TRAJ = 200


# --------------------------------------------------------------------------- #
# Data: load once, then resample pseudo-trajectories per seed.
# --------------------------------------------------------------------------- #
def load_raw_marker_scores(h5_path):
    """Return per-cell RAW (un-normalized) P,D marker scores + per-timepoint index pools."""
    adata = sc.read_h5ad(h5_path)
    adata = adata[adata.obs["treatment"] == "NONE"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pluri = [g for g in PLURI_GENES if g in adata.var_names]
    diff = [g for g in DIFF_GENES if g in adata.var_names]

    def dense(x):
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    P_raw = dense(adata[:, pluri].X).mean(axis=1).ravel()
    D_raw = dense(adata[:, diff].X).mean(axis=1).ravel()

    tp = adata.obs["time_point"].to_numpy()
    pools = {t: np.where(tp == t)[0] for t in TIMEPOINTS}
    return P_raw, D_raw, pools


def sample_trajectories(P_raw, D_raw, pools, rng):
    """Sample N_TRAJ pseudo-trajectories of RAW [P,D] at (D11,D30,D52)."""
    traj = np.empty((N_TRAJ, 3, 2), dtype=np.float64)
    for k, t in enumerate(TIMEPOINTS):
        idx = rng.choice(pools[t], size=N_TRAJ, replace=True)
        traj[:, k, 0] = P_raw[idx]
        traj[:, k, 1] = D_raw[idx]
    return traj


def split_indices(n, rng):
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def normalize_with_train_bounds(traj, idx_tr):
    """Min-max each feature using TRAIN trajectories only; clip val/test to [0,1]."""
    train = traj[idx_tr]
    out = traj.copy()
    for f in (0, 1):
        lo, hi = train[:, :, f].min(), train[:, :, f].max()
        rng = hi - lo if hi > lo else 1.0
        out[:, :, f] = np.clip((traj[:, :, f] - lo) / rng, 0.0, 1.0)
    return out


def make_xy(trajs):
    X = np.array([t[:2].flatten() for t in trajs])   # [D11,D30] flattened
    y = np.array([t[2] for t in trajs])              # D52
    return X, y


# --------------------------------------------------------------------------- #
# Transformer: compact in-memory training (no per-seed checkpoint clutter).
# --------------------------------------------------------------------------- #
def train_transformer(tr, va, te, config, seed, epochs=100):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = TransformerPredictor(input_size=2, output_size=2, config=config).to(device)

    def to_xy(trajs):
        X = torch.tensor(trajs[:, :2, :], dtype=torch.float32)   # (N,2,2)
        y = torch.tensor(trajs[:, 2:3, :], dtype=torch.float32)  # (N,1,2)
        return X, y

    Xtr, ytr = to_xy(tr)
    Xva, yva = to_xy(va)
    Xte, yte = to_xy(te)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss()
    bs = 32
    g = torch.Generator().manual_seed(seed)

    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xtr[b])[:, -1:, :]
            loss = crit(pred, ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva)[:, -1:, :], yva).item()
        sched.step(vloss)
        if vloss < best_val:
            best_val, best_state = vloss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte)[:, -1:, :].squeeze(1).numpy()
    targ = yte.squeeze(1).numpy()
    return per_feature_mae(targ, pred)


def per_feature_mae(y_true, y_pred):
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
    )


# --------------------------------------------------------------------------- #
# Statistics (plan Sec 2.3).
# --------------------------------------------------------------------------- #
def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), (float(lo), float(hi))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def compare(per_seed, reference="Transformer"):
    ref = np.asarray(per_seed[reference])
    names, stats = [], []
    for name, vals in per_seed.items():
        if name == reference:
            continue
        diffs = np.asarray(vals) - ref            # >0 => transformer better
        mean, ci = paired_bootstrap_ci(diffs)
        try:
            _, p = wilcoxon(diffs)
        except ValueError:                        # all-zero diffs
            p = 1.0
        dz = mean / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0
        names.append(name)
        stats.append(dict(mean_diff=mean, ci=ci, wilcoxon_p=float(p), dz=float(dz)))
    adj = holm([s["wilcoxon_p"] for s in stats])
    for s, a in zip(stats, adj):
        s["wilcoxon_p_holm"] = float(a)
    return dict(zip(names, stats))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--h5", default="data/raw/dopaminergic_all_timepoints.h5")
    args = ap.parse_args()

    print("=" * 80)
    print(f"PHASE 1 - MULTI-SEED COMPARISON  (n_seeds={args.seeds})")
    print("=" * 80)
    print("\nLoading raw cells + marker scores (once)...")
    P_raw, D_raw, pools = load_raw_marker_scores(args.h5)
    print(f"  untreated cells: {len(P_raw):,}  | pools: " +
          ", ".join(f"{t}={len(pools[t]):,}" for t in TIMEPOINTS))

    config = load_config()
    models = ["Ridge", "RandomForest", "Transformer"]
    per_seed = {m: [] for m in models}
    per_seed_PD = {m: {"P": [], "D": []} for m in models}

    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        traj_raw = sample_trajectories(P_raw, D_raw, pools, rng)
        idx_tr, idx_va, idx_te = split_indices(N_TRAJ, rng)
        traj = normalize_with_train_bounds(traj_raw, idx_tr)
        tr, va, te = traj[idx_tr], traj[idx_va], traj[idx_te]

        Xtr, ytr = make_xy(tr)
        Xte, yte = make_xy(te)

        r = Ridge(alpha=1.0).fit(Xtr, ytr)
        mae, p, d = per_feature_mae(yte, r.predict(Xte))
        per_seed["Ridge"].append(mae); per_seed_PD["Ridge"]["P"].append(p); per_seed_PD["Ridge"]["D"].append(d)

        rf = tuned_rf(s).fit(Xtr, ytr)
        mae, p, d = per_feature_mae(yte, rf.predict(Xte))
        per_seed["RandomForest"].append(mae); per_seed_PD["RandomForest"]["P"].append(p); per_seed_PD["RandomForest"]["D"].append(d)

        mae, p, d = train_transformer(tr, va, te, config, seed=s, epochs=args.epochs)
        per_seed["Transformer"].append(mae); per_seed_PD["Transformer"]["P"].append(p); per_seed_PD["Transformer"]["D"].append(d)

        print(f"  seed {s:2d}: "
              f"Ridge={per_seed['Ridge'][-1]:.4f}  "
              f"RF={per_seed['RandomForest'][-1]:.4f}  "
              f"Transformer={per_seed['Transformer'][-1]:.4f}")

    # ---- aggregate ----
    print("\n" + "=" * 80)
    print("PER-MODEL TEST MAE  (mean +/- std over seeds, 95% CI)")
    print("=" * 80)
    agg = {}
    for m in models:
        v = np.asarray(per_seed[m])
        mean, ci = paired_bootstrap_ci(v, seed=12345)  # bootstrap the seed distribution
        agg[m] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), ci=ci,
                      P=float(np.mean(per_seed_PD[m]["P"])), D=float(np.mean(per_seed_PD[m]["D"])))
        print(f"  {m:<14} {v.mean():.4f} +/- {v.std(ddof=1):.4f}   "
              f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ---- paired tests vs Transformer ----
    print("\n" + "=" * 80)
    print("PAIRED: baseline - Transformer   (>0 => Transformer better)")
    print("=" * 80)
    cmp = compare(per_seed, reference="Transformer")
    for name, s in cmp.items():
        sig = "SIG" if s["wilcoxon_p_holm"] < 0.05 else "n.s."
        print(f"  {name:<14} mean_diff={s['mean_diff']:+.4f}  "
              f"CI[{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}]  "
              f"dz={s['dz']:+.2f}  p_holm={s['wilcoxon_p_holm']:.3f} [{sig}]")

    # ---- gate verdict ----
    print("\n" + "=" * 80)
    print("DECISION GATE")
    print("=" * 80)
    best_base = min(("Ridge", "RandomForest"), key=lambda m: agg[m]["mean"])
    bd = cmp[best_base]
    tf_better = bd["mean_diff"] > 0
    sig = bd["wilcoxon_p_holm"] < 0.05
    print(f"  Best baseline: {best_base} (MAE {agg[best_base]['mean']:.4f})")
    print(f"  Transformer  : MAE {agg['Transformer']['mean']:.4f}")
    if tf_better and sig:
        print("  -> Transformer edge SURVIVES with significance. Proceed as 'method wins'.")
    else:
        print("  -> Transformer edge does NOT survive. Pivot thesis:")
        print("     'temporal modeling provides no significant advantage on")
        print("      pseudo-trajectory data -- and here is the methodological reason.'")

    out = Path(f"experiments/results/multiseed_n{args.seeds}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(seeds=args.seeds, per_seed=per_seed, per_seed_PD=per_seed_PD,
             aggregate=agg, paired_vs_transformer=cmp), indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
