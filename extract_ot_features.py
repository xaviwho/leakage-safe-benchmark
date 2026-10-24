"""
Positive control (real data), step 1: transcriptome-wide features for OT couplings.

Optimal-transport trajectory inference (Waddington-OT; Schiebinger et al. 2019) couples
cells between consecutive time points in a low-dimensional, transcriptome-wide
expression embedding (here: PCA of 2,000 highly variable genes). Coupling in the 2D (P, D) marker space instead would make any induced
temporal dependence circular, since the targets are built from those markers.

For each lineage we draw a fixed subsample of cells per stage (seed 0) and store:
  P_raw, D_raw : marker-program scores, computed exactly as in run_multiseed*.py
  pca          : 30-component PCA of log1p(CP10K) over 2,000 highly variable genes

Output: data/processed/{lineage}_ot_features.npz

    python extract_ot_features.py --lineage dopaminergic
    python extract_ot_features.py --lineage cardio
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import run_multiseed as neural

K_PER_STAGE = 4000
N_HVG, N_PCS = 2000, 30
CARDIO_DIR = Path("data/raw/cardio_gse175634")
CARDIO_STAGES = ["day0", "day5", "day11"]


def subsample(labels, stages, rng):
    return np.concatenate([rng.choice(np.where(labels == s)[0], K_PER_STAGE, replace=False)
                           for s in stages])


def embed(adata):
    """log1p(CP10K) -> HVG -> PCA, on the subsample only."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    sc.pp.pca(adata, n_comps=N_PCS, mask_var="highly_variable")
    return adata.obsm["X_pca"].astype(np.float32)


def dopaminergic(rng):
    a = sc.read_h5ad("data/raw/dopaminergic_all_timepoints.h5")
    a = a[a.obs["treatment"] == "NONE"]
    idx = subsample(a.obs["time_point"].to_numpy(), neural.TIMEPOINTS, rng)
    a = a[idx].copy()
    stage = a.obs["time_point"].to_numpy().astype(str)
    pca = embed(a)                                 # normalizes a in place
    dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    pluri = [g for g in neural.PLURI_GENES if g in a.var_names]
    diff = [g for g in neural.DIFF_GENES if g in a.var_names]
    P_raw = dense(a[:, pluri].X).mean(axis=1).ravel()
    D_raw = dense(a[:, diff].X).mean(axis=1).ravel()
    return stage, P_raw, D_raw, pca, neural.TIMEPOINTS


def cardio(rng):
    scores = pd.read_csv("data/processed/cardio_cell_scores.csv.gz")   # metadata cell order
    idx = subsample(scores["diffday"].to_numpy(), CARDIO_STAGES, rng)
    keep = np.zeros(len(scores), dtype=bool); keep[idx] = True
    col_of = -np.ones(len(scores), dtype=np.int64); col_of[idx] = np.arange(len(idx))

    genes = pd.read_csv(CARDIO_DIR / "GSE175634_gene_indices_counts.tsv.gz", sep="\t")
    n_genes = len(genes)
    rows, cols, vals = [], [], []
    print("streaming counts matrix (subsampled cells only)...")
    reader = pd.read_csv(CARDIO_DIR / "GSE175634_cell_counts.mtx.gz", sep=" ", comment="%",
                         header=None, skiprows=1, chunksize=20_000_000,
                         names=["i", "j", "v"], dtype={"i": np.int64, "j": np.int64, "v": np.float32})
    gene_is_row = None
    for n, ch in enumerate(reader):
        if gene_is_row is None:     # first data line after the comments is the size header
            gene_is_row = int(ch["i"].iloc[0]) == n_genes
            ch = ch.iloc[1:]
        g, c = (ch["i"], ch["j"]) if gene_is_row else (ch["j"], ch["i"])
        m = keep[c.to_numpy() - 1]
        rows.append(col_of[c.to_numpy()[m] - 1]); cols.append(g.to_numpy()[m] - 1)
        vals.append(ch["v"].to_numpy()[m])
        print(f"  chunk {n + 1}")
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(idx), n_genes))
    a = sc.AnnData(X)
    a.var_names = genes["gene_name"].astype(str).to_numpy()
    a.var_names_make_unique()
    pca = embed(a)
    sub = scores.iloc[idx]
    return (sub["diffday"].to_numpy().astype(str), sub["P_raw"].to_numpy(),
            sub["D_raw"].to_numpy(), pca, CARDIO_STAGES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=["dopaminergic", "cardio"], required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    stage, P_raw, D_raw, pca, stages = (dopaminergic if args.lineage == "dopaminergic"
                                        else cardio)(rng)
    out = Path(f"data/processed/{args.lineage}_ot_features.npz")
    np.savez_compressed(out, stage=stage, P_raw=P_raw, D_raw=D_raw, pca=pca,
                        stages=np.array(stages))
    print(f"[SAVED] {out}  cells={len(stage)}  pca={pca.shape}")
    for s in stages:
        m = stage == s
        print(f"  {s:<6} n={m.sum():5d}  median P_raw={np.median(P_raw[m]):.3f}  "
              f"median D_raw={np.median(D_raw[m]):.3f}")


if __name__ == "__main__":
    main()
