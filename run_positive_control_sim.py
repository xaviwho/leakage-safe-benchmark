"""
Positive control (simulation): does the protocol detect forecasting skill when temporal
dependence is genuinely present?

Cells follow a 2D gradient SDE  dX = -grad V(X) dt + sigma dW  with
    V(P, D) = A/2 * P^2 + B * (D - D_LO)^2 (D - D_HI)^2
P decays (Ornstein-Uhlenbeck), and D starts near the barrier of a double well, so each
cell commits to a low- or high-differentiation fate depending on its early position.
The process is Markov, so s_t2 depends on (s_t0, s_t1) through s_t1, and the true
coupling between time points is known.

Snapshots are cross-sectional: each stage is an independent population. Per seed, three
pseudo-trajectory constructions are built and passed through the identical protocol:
  true        : tracked particles (oracle; not available in real scRNA-seq)
  independent : one independent cell per stage (the null of the main analysis)
  ot          : chained entropic-OT couplings of the snapshots, eps = 2 sigma^2 dt
                (heat-kernel choice for constant-diffusion gradient SDEs; Lavenant et al.)
Every model is scored on its own test split AND on held-out true trajectories.

    python run_positive_control_sim.py --n-traj 200
    python run_positive_control_sim.py --n-traj 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from coupling_protocol import (MODELS, independent_trajectories, ot_chain_trajectories,
                               print_summary, run_protocol, summarize)
from src.utils import load_config

A, B, D_LO, D_HI, SIGMA = 1.0, 8.0, 0.2, 1.0, 0.15
TIMES = [0.0, 1.0, 2.0]
DT = 0.01
POOL = 2000          # snapshot cells per stage
N_TRUE_TEST = 1000   # held-out tracked trajectories for ground-truth scoring
OT_REG = 2 * SIGMA ** 2 * (TIMES[1] - TIMES[0])


def grad_V(x):
    p, d = x[:, 0], x[:, 1]
    gd = 2 * B * (d - D_LO) * (d - D_HI) * ((d - D_LO) + (d - D_HI))
    return np.stack([A * p, gd], axis=1)


def simulate(n, rng):
    """Euler-Maruyama; returns (n, 3, 2) states of n tracked particles at TIMES."""
    x = np.stack([rng.normal(1.0, 0.1, n), rng.normal(0.6, 0.1, n)], axis=1)
    out, t = [x.copy()], 0.0
    for t_next in TIMES[1:]:
        for _ in range(int(round((t_next - t) / DT))):
            x = x - grad_V(x) * DT + SIGMA * np.sqrt(DT) * rng.normal(size=x.shape)
        out.append(x.copy()); t = t_next
    return np.stack(out, axis=1)


def snapshots(rng):
    """Cross-sectional data: a fresh, independent population observed at each stage."""
    return [simulate(POOL, rng)[:, k, :] for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    config = load_config()

    print("=" * 80)
    print(f"POSITIVE CONTROL (SIMULATED SDE)  n_traj={args.n_traj}  seeds={args.seeds}  "
          f"eps={OT_REG:.3f}")
    print("=" * 80)
    constructions = ["true", "independent", "ot"]
    per_seed = {c: {t: {m: [] for m in MODELS} for t in ("own", "true_test")}
                for c in constructions}
    for s in range(args.seeds):
        rng = np.random.default_rng(10_000 + s)
        pools = snapshots(rng)
        true_test = simulate(N_TRUE_TEST, rng)
        trajs = {"true": simulate(args.n_traj, rng),
                 "independent": independent_trajectories(pools, args.n_traj, rng),
                 "ot": ot_chain_trajectories(pools, pools, args.n_traj, OT_REG, rng)}
        line = []
        for c in constructions:
            res = run_protocol(trajs[c], s, config, extra_tests={"true_test": true_test},
                               epochs=args.epochs)
            for t in ("own", "true_test"):
                for m in MODELS:
                    per_seed[c][t][m].append(res[t][m])
            line.append(f"{c}: C={res['own']['Constant']:.3f} R={res['own']['Ridge']:.3f} "
                        f"T={res['own']['Transformer']:.3f}")
        print(f"  seed {s:2d}  " + " | ".join(line))

    summary = {}
    for c in constructions:
        summary[c] = {}
        for t in ("own", "true_test"):
            summary[c][t] = summarize(per_seed[c][t])
            print_summary(f"[{c}] scored on {t}", summary[c][t])

    out = Path(f"experiments/results/positive_control_sim_n{args.n_traj}.json")
    out.write_text(json.dumps(dict(
        seeds=args.seeds, n_traj=args.n_traj,
        sde=dict(A=A, B=B, D_LO=D_LO, D_HI=D_HI, SIGMA=SIGMA, TIMES=TIMES, DT=DT,
                 POOL=POOL, N_TRUE_TEST=N_TRUE_TEST, OT_REG=OT_REG),
        per_seed=per_seed, summary=summary), indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
