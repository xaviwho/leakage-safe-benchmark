"""
Constant (marginal-median) baseline -- the analytical null of the benchmark.

Because each pseudo-trajectory samples one cell independently per stage, the target
s_t2 is independent of the inputs (s_t0, s_t1), and the Bayes-optimal predictor under
MAE is the constant component-wise median of s_t2 (paper, Proposition 1). This script
scores that constant on exactly the per-seed trajectories and splits used by
run_multiseed.py / run_multiseed_cardio.py, then pairs it against the stored per-seed
MAEs of Ridge, RandomForest and the Transformer.

Reproduction check: Ridge is refit on the regenerated data and must match the stored
per-seed Ridge MAE, which confirms the trajectories and splits are identical.

    python run_constant_baseline.py --lineage dopaminergic
    python run_constant_baseline.py --lineage cardio
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_multiseed as neural
import run_multiseed_cardio as cardio
from run_multiseed import compare, paired_bootstrap_ci, per_feature_mae

EQ_MARGIN = 0.01
STORED = {"dopaminergic": "experiments/results/multiseed_n20.json",
          "cardio": "experiments/results/multiseed_cardio_n20.json"}


def seed_data(lineage, s, pools, stages):
    """Regenerate seed s's normalized train/test arrays exactly as the multiseed scripts do."""
    rng = np.random.default_rng(s)
    if lineage == "dopaminergic":
        P_raw, D_raw, idx_pools = pools
        traj_raw = neural.sample_trajectories(P_raw, D_raw, idx_pools, rng)
    else:
        pools_P, pools_D = pools
        traj_raw = cardio.sample_trajectories(pools_P, pools_D, stages, rng)
    idx_tr, _, idx_te = neural.split_indices(neural.N_TRAJ, rng)
    traj = neural.normalize_with_train_bounds(traj_raw, idx_tr)
    Xtr, ytr = neural.make_xy(traj[idx_tr])
    Xte, yte = neural.make_xy(traj[idx_te])
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=["dopaminergic", "cardio"], required=True)
    ap.add_argument("--h5", default="data/raw/dopaminergic_all_timepoints.h5")
    args = ap.parse_args()

    stored = json.loads(Path(STORED[args.lineage]).read_text())
    n_seeds = stored["seeds"]
    stages = stored.get("stages")

    print("=" * 80)
    print(f"CONSTANT (MARGINAL-MEDIAN) BASELINE  lineage={args.lineage}  n_seeds={n_seeds}")
    print("=" * 80)
    if args.lineage == "dopaminergic":
        pools = neural.load_raw_marker_scores(args.h5)
    else:
        df = pd.read_csv(cardio.SCORES)
        pools = ({st: df.loc[df["diffday"] == st, "P_raw"].to_numpy() for st in stages},
                 {st: df.loc[df["diffday"] == st, "D_raw"].to_numpy() for st in stages})

    const, const_PD, ridge_check = [], {"P": [], "D": []}, []
    const_mean, zero_frac = [], []
    for s in range(n_seeds):
        Xtr, ytr, Xte, yte = seed_data(args.lineage, s, pools, stages)

        m = np.median(ytr, axis=0)                      # training-set marginal median of s_t2
        mae, p, d = per_feature_mae(yte, np.tile(m, (len(yte), 1)))
        const.append(mae); const_PD["P"].append(p); const_PD["D"].append(d)

        # Training-mean constant: what MSE-trained models converge to when x carries no information
        const_mean.append(per_feature_mae(yte, np.tile(ytr.mean(axis=0), (len(yte), 1)))[0])
        zero_frac.append((ytr == 0).mean(axis=0))       # mass clipped to the lower bound

        r_mae = per_feature_mae(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte))[0]
        ridge_check.append(r_mae)
        print(f"  seed {s:2d}: Constant={mae:.4f}  Ridge={r_mae:.4f} "
              f"(stored {stored['per_seed']['Ridge'][s]:.4f})")

    max_dev = float(np.max(np.abs(np.asarray(ridge_check) - stored["per_seed"]["Ridge"])))
    print(f"\n  Reproduction check: max |Ridge - stored Ridge| = {max_dev:.2e}")
    if max_dev > 1e-8:
        raise SystemExit("Regenerated data do not match the stored run; paired comparison invalid.")

    per_seed = {"Constant": const, **{k: stored["per_seed"][k]
                                      for k in ("Ridge", "RandomForest", "Transformer")}}

    print("\n" + "=" * 80)
    print("PER-MODEL TEST MAE  (mean +/- std over seeds)")
    print("=" * 80)
    agg = {}
    for name, vals in per_seed.items():
        v = np.asarray(vals)
        _, ci = paired_bootstrap_ci(v, seed=12345)
        agg[name] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), ci=ci)
        print(f"  {name:<14} {v.mean():.4f} +/- {v.std(ddof=1):.4f}")

    print("\n" + "=" * 80)
    print(f"PAIRED: model - Constant   (>0 => Constant better; equivalence margin {EQ_MARGIN})")
    print("=" * 80)
    cmp = compare(per_seed, reference="Constant")
    for name, st in cmp.items():
        st["equivalent"] = bool(-EQ_MARGIN < st["ci"][0] and st["ci"][1] < EQ_MARGIN)
        print(f"  {name:<14} mean_diff={st['mean_diff']:+.4f}  "
              f"CI[{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}]  dz={st['dz']:+.2f}  "
              f"p_holm={st['wilcoxon_p_holm']:.3f}  equivalent={st['equivalent']}")

    zf = np.mean(zero_frac, axis=0)
    print(f"\n  Mean-constant MAE: {np.mean(const_mean):.4f}   "
          f"fraction of s_t2 at lower bound: P={zf[0]:.3f}, D={zf[1]:.3f}")

    out = Path(f"experiments/results/constant_baseline_{args.lineage}_n{n_seeds}.json")
    out.write_text(json.dumps(
        dict(lineage=args.lineage, seeds=n_seeds, stages=stages, eq_margin=EQ_MARGIN,
             per_seed=per_seed, constant_PD=const_PD, ridge_reproduction_max_dev=max_dev,
             aggregate=agg, paired_vs_constant=cmp,
             mean_constant=dict(per_seed=const_mean, mean=float(np.mean(const_mean))),
             target_zero_fraction=dict(P=float(zf[0]), D=float(zf[1]))), indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
