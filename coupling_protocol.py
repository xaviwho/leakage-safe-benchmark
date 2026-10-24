"""
Shared machinery for the positive-control experiments (run_positive_control_sim.py,
run_ot_coupling.py).

Under independent pairing the late stage carries no information from the early stages,
so the benchmark is a null by construction (paper, Proposition 1). The positive controls
ask whether the SAME protocol detects a model advantage when pseudo-trajectories do carry
temporal dependence. This module provides:

  ot_chain_trajectories : pseudo-trajectories from chained entropic-OT couplings
                          (Waddington-OT style: t0 -> t1 -> t2)
  run_protocol          : one seed of the leakage-safe protocol -- shared split,
                          training-bounds normalization, and the full model panel
                          (constant median/mean, Ridge, tuned RF, Transformer, and the
                          Transformer under the stage-order shuffle control)
  summarize             : multi-seed aggregation with paired statistics vs the constant
"""
import copy

import numpy as np
import ot
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from run_multiseed import (compare, make_xy, normalize_with_train_bounds,
                           paired_bootstrap_ci, per_feature_mae, split_indices, tuned_rf)
from src.models.predictors import TransformerPredictor

EQ_MARGIN = 0.01
MODELS = ["Constant", "ConstantMean", "Ridge", "RandomForest", "Transformer",
          "Transformer_shuffle"]


# --------------------------------------------------------------------------- #
# Pseudo-trajectory construction
# --------------------------------------------------------------------------- #
def entropic_coupling(Xa, Xb, reg, normalize_cost=False):
    """Balanced entropic OT plan between two uniformly weighted point clouds."""
    M = ot.dist(Xa, Xb)                            # squared Euclidean cost
    if normalize_cost:                             # Waddington-OT convention
        M = M / np.median(M)
    a = np.full(len(Xa), 1.0 / len(Xa)); b = np.full(len(Xb), 1.0 / len(Xb))
    return ot.sinkhorn(a, b, M, reg, method="sinkhorn_log", numItermax=5000)


def ot_chain_trajectories(feats, states, n, reg, rng, normalize_cost=False):
    """
    feats[k]  : (n_k, f) coupling features of the cells at stage k
    states[k] : (n_k, 2) raw (P, D) states of the same cells
    Draw t0 cells uniformly, then t1 | t0 and t2 | t1 from the row-normalized OT plans.
    """
    plans = [entropic_coupling(feats[k], feats[k + 1], reg, normalize_cost) for k in (0, 1)]
    i0 = rng.integers(len(feats[0]), size=n)
    chain = [i0]
    for k, plan in enumerate(plans):
        rows = plan[chain[-1]]
        rows = rows / rows.sum(axis=1, keepdims=True)
        cdf = np.cumsum(rows, axis=1)
        u = rng.random(n)[:, None]
        chain.append(np.minimum((cdf < u).sum(axis=1), plan.shape[1] - 1))
    return np.stack([states[k][chain[k]] for k in range(3)], axis=1)   # (n, 3, 2)


def independent_trajectories(states, n, rng):
    """Baseline construction: one independently drawn cell per stage."""
    return np.stack([states[k][rng.integers(len(states[k]), size=n)] for k in range(3)], axis=1)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def shuffle_inputs(X, seed):
    """Stage-order shuffle control, as in run_ablations.py: per-sample random
    permutation of the two input tokens (each swapped with probability 1/2)."""
    g = torch.Generator().manual_seed(1000 + seed)
    out = X.clone()
    for i in range(X.shape[0]):
        out[i] = X[i][torch.randperm(X.shape[1], generator=g)]
    return out


def fit_transformer(tr, va, config, seed, epochs=100, shuffle=False):
    """Train exactly as run_multiseed.train_transformer; return a predict function."""
    torch.manual_seed(seed)
    model = TransformerPredictor(input_size=2, output_size=2, config=config)
    xin = lambda t: torch.tensor(t[:, :2, :], dtype=torch.float32)
    yout = lambda t: torch.tensor(t[:, 2:3, :], dtype=torch.float32)
    Xtr, ytr, Xva, yva = xin(tr), yout(tr), xin(va), yout(va)
    if shuffle:
        Xtr, Xva = shuffle_inputs(Xtr, seed), shuffle_inputs(Xva, seed + 500)

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

    def predict(te):
        X = xin(te)
        if shuffle:
            X = shuffle_inputs(X, seed + 900)
        with torch.no_grad():
            return model(X)[:, -1:, :].squeeze(1).numpy()
    return predict


def run_protocol(traj_raw, seed, config, extra_tests=None, epochs=100):
    """
    One seed of the leakage-safe protocol on raw (n, 3, 2) trajectories.
    extra_tests: {name: raw (m, 3, 2)} additional test sets (e.g. ground-truth
    trajectories), normalized with the SAME training bounds.
    Returns {test_set: {model: MAE}}.
    """
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = split_indices(len(traj_raw), rng)
    tests = {"own": traj_raw[idx_te], **(extra_tests or {})}
    stacked = np.concatenate([traj_raw] + list(tests.values()))
    norm = normalize_with_train_bounds(stacked, idx_tr)
    traj = norm[:len(traj_raw)]
    offs = np.cumsum([len(traj_raw)] + [len(v) for v in tests.values()])
    tests = {k: norm[offs[i]:offs[i + 1]] for i, k in enumerate(tests)}
    tests["own"] = traj[idx_te]
    tr, va = traj[idx_tr], traj[idx_va]
    Xtr, ytr = make_xy(tr)

    preds = {
        "Constant": lambda t: np.tile(np.median(ytr, axis=0), (len(t), 1)),
        "ConstantMean": lambda t: np.tile(ytr.mean(axis=0), (len(t), 1)),
    }
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    preds["Ridge"] = lambda t: ridge.predict(make_xy(t)[0])
    rf = tuned_rf(seed).fit(Xtr, ytr)
    preds["RandomForest"] = lambda t: rf.predict(make_xy(t)[0])
    preds["Transformer"] = fit_transformer(tr, va, config, seed, epochs)
    preds["Transformer_shuffle"] = fit_transformer(tr, va, config, seed, epochs, shuffle=True)

    return {name: {m: per_feature_mae(make_xy(t)[1], preds[m](t))[0] for m in MODELS}
            for name, t in tests.items()}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize(per_seed):
    """per_seed: {model: [MAE per seed]} -> aggregate + paired stats vs Constant and
    the shuffle control vs the ordered Transformer."""
    agg = {}
    for m, vals in per_seed.items():
        v = np.asarray(vals)
        agg[m] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
    vs_const = compare({m: per_seed[m] for m in ("Constant", "Ridge", "RandomForest",
                                                 "Transformer")}, reference="Constant")
    for st in vs_const.values():
        st["equivalent"] = bool(-EQ_MARGIN < st["ci"][0] and st["ci"][1] < EQ_MARGIN)
    # Loss-matched reference: the trained models minimize squared error, whose
    # input-free optimum is the training mean -- beating it shows the inputs are used.
    vs_mean = compare({m: per_seed[m] for m in ("ConstantMean", "Ridge", "RandomForest",
                                                "Transformer")}, reference="ConstantMean")
    for st in vs_mean.values():
        st["equivalent"] = bool(-EQ_MARGIN < st["ci"][0] and st["ci"][1] < EQ_MARGIN)
    shuf = compare({"Transformer": per_seed["Transformer"],
                    "Transformer_shuffle": per_seed["Transformer_shuffle"]},
                   reference="Transformer")["Transformer_shuffle"]
    shuf["equivalent"] = bool(-EQ_MARGIN < shuf["ci"][0] and shuf["ci"][1] < EQ_MARGIN)
    tf_vs_rf = compare({"RandomForest": per_seed["RandomForest"],
                        "Transformer": per_seed["Transformer"]},
                       reference="RandomForest")["Transformer"]   # >0 => RF better
    return dict(aggregate=agg, paired_vs_constant=vs_const, paired_vs_constant_mean=vs_mean,
                shuffle_vs_ordered=shuf, transformer_vs_rf=tf_vs_rf)


def print_summary(title, s):
    print(f"\n  {title}")
    for m, a in s["aggregate"].items():
        print(f"    {m:<20} {a['mean']:.4f} +/- {a['std']:.4f}")
    for m, st in s["paired_vs_constant"].items():
        print(f"    {m:<12} - Constant = {st['mean_diff']:+.4f} "
              f"CI[{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}] p_holm={st['wilcoxon_p_holm']:.3g}")
    for m, st in s["paired_vs_constant_mean"].items():
        print(f"    {m:<12} - ConstMean = {st['mean_diff']:+.4f} "
              f"CI[{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}] p_holm={st['wilcoxon_p_holm']:.3g}")
    sh = s["shuffle_vs_ordered"]
    print(f"    shuffle - ordered     = {sh['mean_diff']:+.4f} "
          f"CI[{sh['ci'][0]:+.4f},{sh['ci'][1]:+.4f}] p={sh['wilcoxon_p']:.3g}")


def resummarize(path):
    """Recompute the summary block of a saved positive-control JSON from its per-seed MAEs."""
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    ps = d["per_seed"]
    if "Constant" in next(iter(ps.values())):          # real data: {construction: {model: [...]}}
        d["summary"] = {c: summarize(v) for c, v in ps.items()}
    else:                                              # simulation: {construction: {test: {...}}}
        d["summary"] = {c: {t: summarize(v) for t, v in tests.items()} for c, tests in ps.items()}
    Path(path).write_text(json.dumps(d, indent=2))
    print(f"[RESUMMARIZED] {path}")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        resummarize(p)
