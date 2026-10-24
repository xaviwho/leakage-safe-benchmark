"""
Positive control (real data): rebuild pseudo-trajectories with optimal-transport couplings
instead of independent pairing, and rerun the identical protocol.

Couplings are computed Waddington-OT style in a 30-component PCA of the whole transcriptome
(extract_ot_features.py), with squared-Euclidean cost normalized by its median and entropic
regularization eps = 0.05. Per seed, 2,000 cells per stage are drawn from the stored
4,000-cell subsample, the t0->t1 and t1->t2 plans are computed, and trajectories are
sampled along the chain. The same cells also yield independently paired trajectories, so
the two constructions differ only in the coupling.

There is no ground-truth lineage here: a model advantage under OT coupling shows that the
coupling induces dependence the protocol can detect, not that the dependence is biological.

    python run_ot_coupling.py --lineage cardio --n-traj 200
    python run_ot_coupling.py --lineage dopaminergic --n-traj 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from coupling_protocol import (MODELS, independent_trajectories, ot_chain_trajectories,
                               print_summary, run_protocol, summarize)
from src.utils import load_config

OT_REG = 0.05        # Waddington-OT default, on median-normalized cost
CELLS_PER_STAGE = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=["dopaminergic", "cardio"], required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    config = load_config()

    z = np.load(f"data/processed/{args.lineage}_ot_features.npz")
    stage, pca, stages = z["stage"], z["pca"], list(z["stages"])
    state = np.stack([z["P_raw"], z["D_raw"]], axis=1)
    by_stage = [np.where(stage == s)[0] for s in stages]

    print("=" * 80)
    print(f"POSITIVE CONTROL (OT COUPLING, REAL DATA)  lineage={args.lineage}  "
          f"n_traj={args.n_traj}  seeds={args.seeds}")
    print("=" * 80)
    constructions = ["independent", "ot"]
    per_seed = {c: {m: [] for m in MODELS} for c in constructions}
    for s in range(args.seeds):
        rng = np.random.default_rng(20_000 + s)
        cells = [rng.choice(ix, CELLS_PER_STAGE, replace=False) for ix in by_stage]
        feats = [pca[c] for c in cells]
        states = [state[c] for c in cells]
        trajs = {"independent": independent_trajectories(states, args.n_traj, rng),
                 "ot": ot_chain_trajectories(feats, states, args.n_traj, OT_REG, rng,
                                             normalize_cost=True)}
        line = []
        for c in constructions:
            res = run_protocol(trajs[c], s, config, epochs=args.epochs)["own"]
            for m in MODELS:
                per_seed[c][m].append(res[m])
            line.append(f"{c}: C={res['Constant']:.3f} R={res['Ridge']:.3f} "
                        f"T={res['Transformer']:.3f} Tsh={res['Transformer_shuffle']:.3f}")
        print(f"  seed {s:2d}  " + " | ".join(line))

    summary = {c: summarize(per_seed[c]) for c in constructions}
    for c in constructions:
        print_summary(f"[{c}]", summary[c])

    out = Path(f"experiments/results/ot_coupling_{args.lineage}_n{args.n_traj}.json")
    out.write_text(json.dumps(dict(
        lineage=args.lineage, stages=stages, seeds=args.seeds, n_traj=args.n_traj,
        ot=dict(reg=OT_REG, cost="sqeuclidean/median", space="PCA30 of HVG2000 log1p(CP10K)",
                cells_per_stage=CELLS_PER_STAGE),
        per_seed=per_seed, summary=summary), indent=2))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
