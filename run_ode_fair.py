"""
Phase 2 - fair mechanistic ODE benchmark.

The conference ODE was calibrated to POPULATION MEANS and then scored at the
SAMPLE level (MAE ~0.43) -- a strawman. Here we evaluate three calibration
regimes on the IDENTICAL leakage-free multi-seed test sets (both datasets),
all in normalized [0,1] space so the comparison to Ridge is fair:

  pop      : fit 4 params once to TRAIN population means at the 3 stages
             (the existing population-trend baseline).
  dist     : fit params so the simulated ENSEMBLE (propagating the train t0
             distribution) matches train mean AND std at stages t1, t2.
  pertraj  : for EACH test trajectory, fit params to its own t0->t1 step,
             then integrate on to forecast t2 (fairest sample-level use).

Two-state ODE (normalized P, D), fixed k_ps=1.0, k_dd=0.3 (per methodology):
  dP/dt = k_ps*(1-P) - k_pd*P - diff_rate*P*D
  dD/dt = k_basal + k_ds*P*D + diff_rate*P*D - k_dd*D
free params = [k_pd, k_ds, diff_rate, k_basal]

We report each regime's test MAE vs Ridge (paired over seeds). Expectation under
the paper's thesis: fair calibration improves over the strawman but still does
not recover sample-level predictive power, because pseudo-trajectories carry no
within-trajectory dynamics for a mechanistic model to exploit.

    python run_ode_fair.py --seeds 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

N_TRAJ = 200
K_PS, K_DD = 1.0, 0.3
BOUNDS = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 1.0)]  # k_pd, k_ds, diff_rate, k_basal

PLURI_DOPA = ["POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1"]
DIFF_DOPA = ["TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2"]
DOPA = dict(stages=["D11", "D30", "D52"], times=np.array([0.0, 19.0, 41.0]))
CARDIO = dict(stages=["day0", "day5", "day11"], times=np.array([0.0, 5.0, 11.0]))


def deriv(t, y, p):
    P, D = y
    k_pd, k_ds, diff_rate, k_basal = p
    dP = K_PS * (1 - P) - k_pd * P - diff_rate * P * D
    dD = k_basal + k_ds * P * D + diff_rate * P * D - K_DD * D
    return [dP, dD]


def integrate(y0, t_eval, p):
    """Integrate from y0 over the absolute times in t_eval; return states at each."""
    sol = solve_ivp(deriv, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    args=(p,), method="RK45", rtol=1e-4, atol=1e-6)
    return np.clip(sol.y.T, 0.0, 1.0)  # (len(t_eval), 2)


# --------------------------------------------------------------------------- #
# data (mirrors run_multiseed exactly)
# --------------------------------------------------------------------------- #
def load_dopaminergic():
    a = sc.read_h5ad("data/raw/dopaminergic_all_timepoints.h5")
    a = a[a.obs["treatment"] == "NONE"].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    P = dense(a[:, [g for g in PLURI_DOPA if g in a.var_names]].X).mean(1).ravel()
    D = dense(a[:, [g for g in DIFF_DOPA if g in a.var_names]].X).mean(1).ravel()
    tp = a.obs["time_point"].to_numpy()
    return {s: P[tp == s] for s in DOPA["stages"]}, {s: D[tp == s] for s in DOPA["stages"]}


def load_cardio():
    df = pd.read_csv("data/processed/cardio_cell_scores.csv.gz")
    return ({s: df.loc[df.diffday == s, "P_raw"].to_numpy() for s in CARDIO["stages"]},
            {s: df.loc[df.diffday == s, "D_raw"].to_numpy() for s in CARDIO["stages"]})


def sample_trajectories(pP, pD, stages, rng):
    traj = np.empty((N_TRAJ, 3, 2))
    for k, st in enumerate(stages):
        idx = rng.choice(len(pP[st]), size=N_TRAJ, replace=True)
        traj[:, k, 0] = pP[st][idx]; traj[:, k, 1] = pD[st][idx]
    return traj


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


# --------------------------------------------------------------------------- #
# calibration regimes
# --------------------------------------------------------------------------- #
def calibrate_pop(tr, times):
    """Fit params so integrating from mean(t0) matches mean(t1), mean(t2)."""
    m = tr.mean(axis=0)  # (3,2) stage means
    def obj(p):
        pred = integrate(m[0], times, p)
        return np.sum((pred[1:] - m[1:]) ** 2)
    res = differential_evolution(obj, BOUNDS, seed=0, maxiter=60, popsize=12, tol=1e-7)
    return res.x


def calibrate_dist(tr, times):
    """Match train mean AND std at t1,t2 by propagating the train t0 ensemble."""
    y0s = tr[:, 0, :]                       # train t0 states
    target_m = tr.mean(axis=0)[1:]          # (2,2)
    target_s = tr.std(axis=0)[1:]
    # subsample ICs for speed
    sel = y0s[np.linspace(0, len(y0s) - 1, min(40, len(y0s))).astype(int)]
    def obj(p):
        sims = np.stack([integrate(y0, times, p) for y0 in sel])  # (n,3,2)
        sm, ss = sims[:, 1:, :].mean(0), sims[:, 1:, :].std(0)
        return np.concatenate([(sm - target_m).ravel(), (ss - target_s).ravel()])
    res = least_squares(obj, x0=[0.3, 0.3, 0.2, 0.05],
                        bounds=([b[0] for b in BOUNDS], [b[1] for b in BOUNDS]), max_nfev=200)
    return res.x


def predict_pop_or_dist(te, times, p):
    """Integrate each test traj from its own t0 to t2; MAE on t2."""
    preds = np.array([integrate(traj[0], times, p)[2] for traj in te])
    return mean_absolute_error(te[:, 2, :], preds)


def predict_pertraj(te, times, p_init):
    """Per traj: fit params to t0->t1, then forecast t2."""
    lo = [b[0] for b in BOUNDS]; hi = [b[1] for b in BOUNDS]
    preds = []
    for traj in te:
        y0, y1 = traj[0], traj[1]
        def obj(p):
            return integrate(y0, times[:2], p)[1] - y1
        try:
            r = least_squares(obj, x0=p_init, bounds=(lo, hi), max_nfev=60)
            p = r.x
        except Exception:
            p = p_init
        preds.append(integrate(y0, times, p)[2])
    return mean_absolute_error(te[:, 2, :], np.array(preds))


# --------------------------------------------------------------------------- #
def run_dataset(name, loader, cfg, seeds):
    print("=" * 80); print(f"FAIR ODE - {name}"); print("=" * 80)
    pP, pD = loader()
    stages, times = cfg["stages"], cfg["times"]
    res = {k: [] for k in ("Ridge", "ODE_pop", "ODE_dist", "ODE_pertraj")}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        traj = sample_trajectories(pP, pD, stages, rng)
        itr, iva, ite = split_indices(N_TRAJ, rng)
        traj = normalize_with_train_bounds(traj, itr)
        tr, te = traj[itr], traj[ite]

        Xtr = tr[:, :2, :].reshape(len(tr), -1); ytr = tr[:, 2, :]
        Xte = te[:, :2, :].reshape(len(te), -1); yte = te[:, 2, :]
        res["Ridge"].append(mean_absolute_error(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)))

        p_pop = calibrate_pop(tr, times)
        res["ODE_pop"].append(predict_pop_or_dist(te, times, p_pop))
        p_dist = calibrate_dist(tr, times)
        res["ODE_dist"].append(predict_pop_or_dist(te, times, p_dist))
        res["ODE_pertraj"].append(predict_pertraj(te, times, p_pop))
        print(f"  seed {s:2d}: Ridge={res['Ridge'][-1]:.4f}  "
              f"ODE_pop={res['ODE_pop'][-1]:.4f}  ODE_dist={res['ODE_dist'][-1]:.4f}  "
              f"ODE_pertraj={res['ODE_pertraj'][-1]:.4f}")

    print(f"\n  {'model':<14}{'mean MAE':>10}{'std':>9}  vs Ridge")
    agg = {}
    ridge = np.asarray(res["Ridge"])
    for k in ("Ridge", "ODE_pop", "ODE_dist", "ODE_pertraj"):
        v = np.asarray(res[k]); agg[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
        delta = "" if k == "Ridge" else f"  {((v.mean()-ridge.mean())/ridge.mean()*100):+.0f}%"
        print(f"  {k:<14}{v.mean():>10.4f}{v.std(ddof=1):>9.4f}{delta}")
    return dict(per_seed=res, aggregate=agg, stages=stages, times=times.tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    out = {}
    out["dopaminergic"] = run_dataset("DOPAMINERGIC (neural)", load_dopaminergic, DOPA, args.seeds)
    print()
    out["cardiomyocyte"] = run_dataset("CARDIOMYOCYTE (cardiac)", load_cardio, CARDIO, args.seeds)

    print("\n" + "=" * 80); print("PHASE 2 CONCLUSION"); print("=" * 80)
    for name in ("dopaminergic", "cardiomyocyte"):
        a = out[name]["aggregate"]
        best_ode = min(("ODE_pop", "ODE_dist", "ODE_pertraj"), key=lambda k: a[k]["mean"])
        verb = "competitive with" if a[best_ode]["mean"] <= a["Ridge"]["mean"] * 1.1 else "still worse than"
        print(f"  {name:<14}: best fair ODE = {best_ode} ({a[best_ode]['mean']:.4f}) "
              f"-> {verb} Ridge ({a['Ridge']['mean']:.4f})")

    p = Path(f"experiments/results/ode_fair_n{args.seeds}.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[SAVED] {p}")


if __name__ == "__main__":
    main()
