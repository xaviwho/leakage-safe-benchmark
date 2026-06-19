"""
Phase 5 - state-representation sensitivity.

Stress-tests the headline null (Transformer provides no advantage over Ridge)
against reasonable choices in how the 2D (P, D) state is built:

  baseline      : mean of log1p(CP10K) over the marker panel  (the choice used so far)
  LOO_<gene>    : drop one marker gene, re-aggregate            (marker-set robustness)
  agg_median    : median instead of mean over the panel        (aggregation robustness)
  agg_zscore    : per-gene z-scored, then mean                  (aggregation robustness)
  larger_panel  : expanded curated marker set (dopaminergic)    (panel-size robustness)

For each variant we run the multi-seed Ridge-vs-Transformer comparison on a SHARED
set of sampled cells (same cells per seed across variants -> clean paired design)
and report Transformer-minus-Ridge. The null is robust iff that gap stays ~0
everywhere. Default 10 seeds (enough to characterize the gap).

    python run_sensitivity.py --dataset dopaminergic --seeds 10
    python run_sensitivity.py --dataset cardio --seeds 10
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.models.predictors import TransformerPredictor
from src.utils import load_config

N_TRAJ = 200

DOPA_PLURI = ["POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1"]
DOPA_DIFF = ["TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2"]
DOPA_PLURI_EXTRA = ["LIN28A", "DPPA4", "ZFP42", "PRDM14", "POU5F1B"]
DOPA_DIFF_EXTRA = ["NR4A2", "PITX3", "EN1", "KCNJ6", "SLC18A2", "OTX2", "TUBB3", "MAP2"]
DOPA_STAGES = ["D11", "D30", "D52"]
CARDIO_STAGES = ["day0", "day5", "day11"]


# --------------------------------------------------------------------------- #
# Build gene-level pools: per stage, a (cells, genes) log1p matrix for P and D.
# --------------------------------------------------------------------------- #
def load_dopaminergic():
    a = sc.read_h5ad("data/raw/dopaminergic_all_timepoints.h5")
    a = a[a.obs["treatment"] == "NONE"].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    pluri = [g for g in DOPA_PLURI + DOPA_PLURI_EXTRA if g in a.var_names]
    diff = [g for g in DOPA_DIFF + DOPA_DIFF_EXTRA if g in a.var_names]
    Xp = dense(a[:, pluri].X); Xd = dense(a[:, diff].X)
    tp = a.obs["time_point"].to_numpy()
    poolsP = {s: Xp[tp == s] for s in DOPA_STAGES}
    poolsD = {s: Xd[tp == s] for s in DOPA_STAGES}
    base_p = [pluri.index(g) for g in DOPA_PLURI if g in pluri]
    base_d = [diff.index(g) for g in DOPA_DIFF if g in diff]
    extra_p = [pluri.index(g) for g in DOPA_PLURI_EXTRA if g in pluri]
    extra_d = [diff.index(g) for g in DOPA_DIFF_EXTRA if g in diff]
    return dict(poolsP=poolsP, poolsD=poolsD, genesP=pluri, genesD=diff,
                base_p=base_p, base_d=base_d, extra_p=extra_p, extra_d=extra_d,
                stages=DOPA_STAGES, has_extra=True)


def load_cardio():
    z = np.load("data/processed/cardio_marker_log.npz", allow_pickle=True)
    pl, cd, dd = z["pluri_log"], z["cardiac_log"], z["diffday"].astype(str)
    poolsP = {s: pl[dd == s] for s in CARDIO_STAGES}
    poolsD = {s: cd[dd == s] for s in CARDIO_STAGES}
    genesP = list(z["pluri_genes"].astype(str)); genesD = list(z["cardiac_genes"].astype(str))
    return dict(poolsP=poolsP, poolsD=poolsD, genesP=genesP, genesD=genesD,
                base_p=list(range(len(genesP))), base_d=list(range(len(genesD))),
                extra_p=[], extra_d=[], stages=CARDIO_STAGES, has_extra=False)


def aggregate(submat, mode, mu=None, sd=None):
    if mode == "mean":
        return submat.mean(axis=1)
    if mode == "median":
        return np.median(submat, axis=1)
    if mode == "zscore":
        return ((submat - mu) / np.where(sd > 0, sd, 1.0)).mean(axis=1)
    raise ValueError(mode)


def build_variants(data):
    """Return list of (name, p_cols, d_cols, agg_mode)."""
    base_p, base_d = data["base_p"], data["base_d"]
    v = [("baseline", base_p, base_d, "mean")]
    for k in base_p:
        v.append((f"LOO_P_{data['genesP'][k]}", [c for c in base_p if c != k], base_d, "mean"))
    for k in base_d:
        v.append((f"LOO_D_{data['genesD'][k]}", base_p, [c for c in base_d if c != k], "mean"))
    v.append(("agg_median", base_p, base_d, "median"))
    v.append(("agg_zscore", base_p, base_d, "zscore"))
    if data["has_extra"]:
        v.append(("larger_panel", base_p + data["extra_p"], base_d + data["extra_d"], "mean"))
    return v


# --------------------------------------------------------------------------- #
def split_indices(n, rng):
    idx = rng.permutation(n); a, b = int(0.7 * n), int(0.15 * n)
    return idx[:a], idx[a:a + b], idx[a + b:]


def normalize_with_train_bounds(traj, itr):
    out = traj.copy()
    for f in (0, 1):
        lo, hi = traj[itr][:, :, f].min(), traj[itr][:, :, f].max()
        r = hi - lo if hi > lo else 1.0
        out[:, :, f] = np.clip((traj[:, :, f] - lo) / r, 0.0, 1.0)
    return out


def train_transformer(tr, va, te, config, seed, epochs):
    torch.manual_seed(seed)
    model = TransformerPredictor(input_size=2, output_size=2, config=config)
    xin = lambda t: torch.tensor(t[:, :2, :], dtype=torch.float32)
    yout = lambda t: torch.tensor(t[:, 2:3, :], dtype=torch.float32)
    Xtr, ytr, Xva, yva, Xte, yte = xin(tr), yout(tr), xin(va), yout(va), xin(te), yout(te)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit = nn.MSELoss(); g = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]; opt.zero_grad()
            loss = crit(model(Xtr[b])[:, -1:, :], ytr[b]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva)[:, -1:, :], yva).item()
        sched.step(vl)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(Xte)[:, -1:, :].squeeze(1).numpy()
    return mean_absolute_error(yte.squeeze(1).numpy(), pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dopaminergic", "cardio"], required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    print("=" * 80); print(f"PHASE 5 - SENSITIVITY ({args.dataset})"); print("=" * 80)
    data = load_dopaminergic() if args.dataset == "dopaminergic" else load_cardio()
    stages = data["stages"]
    # global per-gene mean/std (over all stages) for zscore aggregation
    allP = np.concatenate([data["poolsP"][s] for s in stages], axis=0)
    allD = np.concatenate([data["poolsD"][s] for s in stages], axis=0)
    muP, sdP = allP.mean(0), allP.std(0)
    muD, sdD = allD.mean(0), allD.std(0)
    variants = build_variants(data)
    config = load_config()
    print(f"  {len(variants)} variants x {args.seeds} seeds")

    results = {}
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        # sample the SAME cells per stage for all variants this seed
        samp = {st: rng.choice(len(data["poolsP"][st]), size=N_TRAJ, replace=True) for st in stages}
        itr, iva, ite = split_indices(N_TRAJ, rng)
        for (name, pc, dc, mode) in variants:
            traj = np.empty((N_TRAJ, 3, 2))
            for k, st in enumerate(stages):
                idx = samp[st]
                traj[:, k, 0] = aggregate(data["poolsP"][st][idx][:, pc], mode, muP[pc], sdP[pc])
                traj[:, k, 1] = aggregate(data["poolsD"][st][idx][:, dc], mode, muD[dc], sdD[dc])
            traj = normalize_with_train_bounds(traj, itr)
            tr, va, te = traj[itr], traj[iva], traj[ite]
            Xtr = tr[:, :2, :].reshape(len(tr), -1); ytr = tr[:, 2, :]
            Xte = te[:, :2, :].reshape(len(te), -1); yte = te[:, 2, :]
            r = mean_absolute_error(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte))
            t = train_transformer(tr, va, te, config, seed=s, epochs=args.epochs)
            results.setdefault(name, {"ridge": [], "tf": []})
            results[name]["ridge"].append(r); results[name]["tf"].append(t)
        print(f"  seed {s} done")

    print(f"\n  {'variant':<22}{'Ridge':>8}{'Transf':>9}{'gap(T-R)':>10}{'dz':>7}{'gap CI':>22}")
    summary = {}
    for name, d in results.items():
        r = np.asarray(d["ridge"]); t = np.asarray(d["tf"]); gap = t - r
        rng = np.random.default_rng(7)
        boots = rng.choice(gap, size=(5000, len(gap)), replace=True).mean(1)
        ci = tuple(np.percentile(boots, [2.5, 97.5]).astype(float))
        dz = float(gap.mean() / gap.std(ddof=1)) if gap.std(ddof=1) > 0 else 0.0
        summary[name] = dict(ridge=float(r.mean()), tf=float(t.mean()),
                             gap=float(gap.mean()), dz=dz, gap_ci=ci)
        flag = "" if ci[0] <= 0 <= ci[1] else "  <-- gap excludes 0"
        print(f"  {name:<22}{r.mean():>8.4f}{t.mean():>9.4f}{gap.mean():>+10.4f}{dz:>+7.2f}"
              f"   [{ci[0]:+.4f},{ci[1]:+.4f}]{flag}")

    gaps = np.array([v["gap"] for v in summary.values()])
    print(f"\n  gap(T-R) across {len(summary)} variants: "
          f"mean={gaps.mean():+.4f}, range=[{gaps.min():+.4f}, {gaps.max():+.4f}]")
    n_excl = sum(1 for v in summary.values() if not (v["gap_ci"][0] <= 0 <= v["gap_ci"][1]))
    print(f"  variants where Transformer significantly beats Ridge: {n_excl}/{len(summary)}")
    print("  -> null is ROBUST to representation choices" if n_excl == 0
          else "  -> some representations show a Transformer edge")

    out = Path(f"experiments/results/sensitivity_{args.dataset}_n{args.seeds}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
