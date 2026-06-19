"""
Phase 4 - architectural ablations: does the Transformer use temporal structure?

For each dataset (dopaminergic neural + cardiomyocyte cardiac) we train three
Transformer variants on the SAME [t0, t1] -> t2 task across 20 seeds:

  ordered   : inputs in correct temporal order (the Phase 1/3 baseline)
  shuffle   : input timepoints permuted per-sample (train AND test) -> stage order destroyed
  no_pe     : positional encoding zeroed -> no positional signal at all

The critical control is `shuffle` (plan Sec 5, item 2): if destroying stage order
does NOT degrade MAE, the model is not using temporal structure and the
"forecasting" framing is hollow. `no_pe` is the corroborating control.

Paired (per-seed) tests compare each variant to `ordered`. A non-significant
difference is the POSITIVE result here: it proves temporal machinery is inert.

    python run_ablations.py --seeds 20
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error

from src.models.predictors import TransformerPredictor
from src.utils import load_config

N_TRAJ = 200
PLURI_DOPA = ["POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1"]
DIFF_DOPA = ["TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2"]
DOPA_STAGES = ["D11", "D30", "D52"]
CARDIO_STAGES = ["day0", "day5", "day11"]


# --------------------------------------------------------------------------- #
# Dataset loaders -> per-stage pools of (P_raw, D_raw)
# --------------------------------------------------------------------------- #
def load_dopaminergic():
    a = sc.read_h5ad("data/raw/dopaminergic_all_timepoints.h5")
    a = a[a.obs["treatment"] == "NONE"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    def dense(x):
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    pl = [g for g in PLURI_DOPA if g in a.var_names]
    di = [g for g in DIFF_DOPA if g in a.var_names]
    P = dense(a[:, pl].X).mean(axis=1).ravel()
    D = dense(a[:, di].X).mean(axis=1).ravel()
    tp = a.obs["time_point"].to_numpy()
    return ({s: P[tp == s] for s in DOPA_STAGES},
            {s: D[tp == s] for s in DOPA_STAGES}, DOPA_STAGES)


def load_cardio():
    df = pd.read_csv("data/processed/cardio_cell_scores.csv.gz")
    return ({s: df.loc[df["diffday"] == s, "P_raw"].to_numpy() for s in CARDIO_STAGES},
            {s: df.loc[df["diffday"] == s, "D_raw"].to_numpy() for s in CARDIO_STAGES},
            CARDIO_STAGES)


def sample_trajectories(pools_P, pools_D, stages, rng):
    traj = np.empty((N_TRAJ, 3, 2))
    for k, st in enumerate(stages):
        idx = rng.choice(len(pools_P[st]), size=N_TRAJ, replace=True)
        traj[:, k, 0] = pools_P[st][idx]
        traj[:, k, 1] = pools_D[st][idx]
    return traj


def split_indices(n, rng):
    idx = rng.permutation(n); a, b = int(0.7 * n), int(0.15 * n)
    return idx[:a], idx[a:a + b], idx[a + b:]


def normalize_with_train_bounds(traj, idx_tr):
    out = traj.copy()
    for f in (0, 1):
        lo, hi = traj[idx_tr][:, :, f].min(), traj[idx_tr][:, :, f].max()
        r = hi - lo if hi > lo else 1.0
        out[:, :, f] = np.clip((traj[:, :, f] - lo) / r, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Transformer with optional shuffle / no-PE
# --------------------------------------------------------------------------- #
def train_variant(tr, va, te, config, seed, mode, epochs=100):
    torch.manual_seed(seed)
    model = TransformerPredictor(input_size=2, output_size=2, config=config)
    if mode == "no_pe":
        model.pos_encoder.pe.zero_()          # kill positional signal

    def xin(t):
        return torch.tensor(t[:, :2, :], dtype=torch.float32)

    def yout(t):
        return torch.tensor(t[:, 2:3, :], dtype=torch.float32)

    Xtr, ytr = xin(tr), yout(tr)
    Xva, yva = xin(va), yout(va)
    Xte, yte = xin(te), yout(te)

    if mode == "shuffle":
        # per-sample fixed permutation of the 2 input timepoints (train + test)
        gsh = torch.Generator().manual_seed(1000 + seed)
        def shuffle(X):
            out = X.clone()
            for i in range(X.shape[0]):
                out[i] = X[i][torch.randperm(X.shape[1], generator=gsh)]
            return out
        Xtr, Xva, Xte = shuffle(Xtr), shuffle(Xva), shuffle(Xte)

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
    return float(mean_absolute_error(yte.squeeze(1).numpy(), pred))


def paired_bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), tuple(np.percentile(boots, [2.5, 97.5]).astype(float))


def run_dataset(name, loader, seeds, epochs):
    print("=" * 80)
    print(f"ABLATIONS - {name}")
    print("=" * 80)
    pools_P, pools_D, stages = loader()
    print("  stages: " + ", ".join(f"{s}={len(pools_P[s]):,}" for s in stages))
    variants = ["ordered", "shuffle", "no_pe"]
    res = {v: [] for v in variants}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        traj = sample_trajectories(pools_P, pools_D, stages, rng)
        itr, iva, ite = split_indices(N_TRAJ, rng)
        traj = normalize_with_train_bounds(traj, itr)
        tr, va, te = traj[itr], traj[iva], traj[ite]
        for v in variants:
            res[v].append(train_variant(tr, va, te, load_config(), seed=s, mode=v, epochs=epochs))
        print(f"  seed {s:2d}: ordered={res['ordered'][-1]:.4f}  "
              f"shuffle={res['shuffle'][-1]:.4f}  no_pe={res['no_pe'][-1]:.4f}")

    print(f"\n  {'variant':<10}{'mean MAE':>10}{'std':>9}")
    agg = {}
    for v in variants:
        a = np.asarray(res[v]); agg[v] = dict(mean=float(a.mean()), std=float(a.std(ddof=1)))
        print(f"  {v:<10}{a.mean():>10.4f}{a.std(ddof=1):>9.4f}")

    print(f"\n  paired vs ordered (>0 => variant WORSE than ordered; n.s. => order unused):")
    ordered = np.asarray(res["ordered"])
    tests = {}
    for v in ("shuffle", "no_pe"):
        diffs = np.asarray(res[v]) - ordered
        mean, ci = paired_bootstrap_ci(diffs)
        try:
            _, p = wilcoxon(diffs)
        except ValueError:
            p = 1.0
        tests[v] = dict(mean_diff=mean, ci=ci, wilcoxon_p=float(p))
        sig = "SIG (order matters)" if p < 0.05 else "n.s. (order unused)"
        print(f"    {v:<10} mean_diff={mean:+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.3f} [{sig}]")
    return dict(per_seed=res, aggregate=agg, tests=tests, stages=stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    out = {}
    out["dopaminergic"] = run_dataset("DOPAMINERGIC (neural)", load_dopaminergic, args.seeds, args.epochs)
    print()
    out["cardiomyocyte"] = run_dataset("CARDIOMYOCYTE (cardiac)", load_cardio, args.seeds, args.epochs)

    print("\n" + "=" * 80)
    print("PHASE 4 CONCLUSION")
    print("=" * 80)
    for name in ("dopaminergic", "cardiomyocyte"):
        t = out[name]["tests"]
        verdict = ("temporal order UNUSED" if t["shuffle"]["wilcoxon_p"] >= 0.05
                   else "temporal order matters")
        print(f"  {name:<14}: shuffle p={t['shuffle']['wilcoxon_p']:.3f}, "
              f"no_pe p={t['no_pe']['wilcoxon_p']:.3f}  -> {verdict}")

    p = Path(f"experiments/results/ablations_n{args.seeds}.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[SAVED] {p}")


if __name__ == "__main__":
    main()
