"""
Phase 3 - cross-lineage generalization: the IDENTICAL multi-seed protocol from
run_multiseed.py, applied to the GSE175634 iPSC->cardiomyocyte time-course.

Mirrors the dopaminergic task exactly: 3 timepoints, [t0, t1] -> t2, one shared
leakage-free split per seed, train-only normalization, paired stats. The only
changes are the data source and the (P, D) marker programs (cardiac, not neural).

We use 3 of the 7 available days spanning the trajectory (early / mid / late) so
the task structure matches the original. Default: day0 -> day7 -> day15.

    python run_multiseed_cardio.py --seeds 20
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Fair RF baseline: per-seed 3-fold-CV-tuned depth/leaf (see run_multiseed.py).
RF_GRID = {"n_estimators": [300], "max_depth": [2, 3, 5, None],
           "min_samples_leaf": [1, 5, 10]}


def tuned_rf(seed):
    return GridSearchCV(RandomForestRegressor(random_state=seed, n_jobs=-1),
                        RF_GRID, scoring="neg_mean_absolute_error", cv=3)

from src.models.predictors import TransformerPredictor
from src.utils import load_config

N_TRAJ = 200
SCORES = "data/processed/cardio_cell_scores.csv.gz"


def split_indices(n, rng):
    idx = rng.permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def sample_trajectories(pools_P, pools_D, stages, rng):
    traj = np.empty((N_TRAJ, 3, 2), dtype=np.float64)
    for k, st in enumerate(stages):
        idx = rng.choice(len(pools_P[st]), size=N_TRAJ, replace=True)
        traj[:, k, 0] = pools_P[st][idx]
        traj[:, k, 1] = pools_D[st][idx]
    return traj


def normalize_with_train_bounds(traj, idx_tr):
    train = traj[idx_tr]
    out = traj.copy()
    for f in (0, 1):
        lo, hi = train[:, :, f].min(), train[:, :, f].max()
        rng = hi - lo if hi > lo else 1.0
        out[:, :, f] = np.clip((traj[:, :, f] - lo) / rng, 0.0, 1.0)
    return out


def make_xy(trajs):
    return np.array([t[:2].flatten() for t in trajs]), np.array([t[2] for t in trajs])


def per_feature_mae(y_true, y_pred):
    return (float(mean_absolute_error(y_true, y_pred)),
            float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
            float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])))


def train_transformer(tr, va, te, config, seed, epochs=100):
    torch.manual_seed(seed)
    model = TransformerPredictor(input_size=2, output_size=2, config=config)

    def to_xy(t):
        return (torch.tensor(t[:, :2, :], dtype=torch.float32),
                torch.tensor(t[:, 2:3, :], dtype=torch.float32))

    Xtr, ytr = to_xy(tr); Xva, yva = to_xy(va); Xte, yte = to_xy(te)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(model(Xtr[b])[:, -1:, :], ytr[b])
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
    return per_feature_mae(yte.squeeze(1).numpy(), pred)


def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), (float(lo), float(hi))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(running, 1.0)
    return adj


def compare(per_seed, reference="Transformer"):
    ref = np.asarray(per_seed[reference]); names, stats = [], []
    for name, vals in per_seed.items():
        if name == reference:
            continue
        diffs = np.asarray(vals) - ref
        mean, ci = paired_bootstrap_ci(diffs)
        try:
            _, p = wilcoxon(diffs)
        except ValueError:
            p = 1.0
        dz = mean / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0
        names.append(name); stats.append(dict(mean_diff=mean, ci=ci, wilcoxon_p=float(p), dz=float(dz)))
    for s, a in zip(stats, holm([s["wilcoxon_p"] for s in stats])):
        s["wilcoxon_p_holm"] = float(a)
    return dict(zip(names, stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--stages", default="day0,day7,day15")
    args = ap.parse_args()
    stages = args.stages.split(",")

    print("=" * 80)
    print(f"PHASE 3 - CARDIOMYOCYTE MULTI-SEED  (n_seeds={args.seeds}, stages={stages})")
    print("=" * 80)
    df = pd.read_csv(SCORES)
    pools_P = {st: df.loc[df["diffday"] == st, "P_raw"].to_numpy() for st in stages}
    pools_D = {st: df.loc[df["diffday"] == st, "D_raw"].to_numpy() for st in stages}
    print("  cells/stage: " + ", ".join(f"{st}={len(pools_P[st]):,}" for st in stages))

    config = load_config()
    models = ["Ridge", "RandomForest", "Transformer"]
    per_seed = {m: [] for m in models}
    per_seed_PD = {m: {"P": [], "D": []} for m in models}

    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        traj_raw = sample_trajectories(pools_P, pools_D, stages, rng)
        idx_tr, idx_va, idx_te = split_indices(N_TRAJ, rng)
        traj = normalize_with_train_bounds(traj_raw, idx_tr)
        tr, va, te = traj[idx_tr], traj[idx_va], traj[idx_te]
        Xtr, ytr = make_xy(tr); Xte, yte = make_xy(te)

        for name, mdl in (("Ridge", Ridge(alpha=1.0)),
                          ("RandomForest", tuned_rf(s))):
            mdl.fit(Xtr, ytr)
            mae, p, d = per_feature_mae(yte, mdl.predict(Xte))
            per_seed[name].append(mae); per_seed_PD[name]["P"].append(p); per_seed_PD[name]["D"].append(d)

        mae, p, d = train_transformer(tr, va, te, config, seed=s, epochs=args.epochs)
        per_seed["Transformer"].append(mae); per_seed_PD["Transformer"]["P"].append(p); per_seed_PD["Transformer"]["D"].append(d)

        print(f"  seed {s:2d}: Ridge={per_seed['Ridge'][-1]:.4f}  "
              f"RF={per_seed['RandomForest'][-1]:.4f}  Transformer={per_seed['Transformer'][-1]:.4f}")

    print("\n" + "=" * 80)
    print("PER-MODEL TEST MAE  (mean +/- std, 95% CI)")
    print("=" * 80)
    agg = {}
    for m in models:
        v = np.asarray(per_seed[m]); mean, ci = paired_bootstrap_ci(v, seed=12345)
        agg[m] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), ci=ci)
        print(f"  {m:<14} {v.mean():.4f} +/- {v.std(ddof=1):.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    print("\n" + "=" * 80)
    print("PAIRED: baseline - Transformer  (>0 => Transformer better)")
    print("=" * 80)
    cmp = compare(per_seed)
    for name, s in cmp.items():
        sig = "SIG" if s["wilcoxon_p_holm"] < 0.05 else "n.s."
        print(f"  {name:<14} mean_diff={s['mean_diff']:+.4f}  "
              f"CI[{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}]  dz={s['dz']:+.2f}  "
              f"p_holm={s['wilcoxon_p_holm']:.3f} [{sig}]")

    best_base = min(("Ridge", "RandomForest"), key=lambda m: agg[m]["mean"])
    bd = cmp[best_base]
    print("\n" + "=" * 80)
    print("DECISION GATE (cardiomyocyte / cross-lineage)")
    print("=" * 80)
    print(f"  Best baseline: {best_base} (MAE {agg[best_base]['mean']:.4f})")
    print(f"  Transformer  : MAE {agg['Transformer']['mean']:.4f}")
    if bd["mean_diff"] > 0 and bd["wilcoxon_p_holm"] < 0.05:
        print("  -> Transformer edge SURVIVES on cardiomyocyte data.")
    else:
        print("  -> No significant Transformer edge on cardiomyocyte data either.")
        print("     The Phase-1 null GENERALIZES across lineage.")

    out = Path(f"experiments/results/multiseed_cardio_n{args.seeds}.json")
    out.write_text(json.dumps(dict(seeds=args.seeds, stages=stages, per_seed=per_seed,
                                   per_seed_PD=per_seed_PD, aggregate=agg,
                                   paired_vs_transformer=cmp), indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
