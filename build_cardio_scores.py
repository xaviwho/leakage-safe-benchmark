"""
Phase 3 (cross-lineage): build per-cell (P, D) marker scores for the
GSE175634 iPSC -> cardiomyocyte differentiation time-course.

Mirrors the dopaminergic protocol EXACTLY so the comparison is apples-to-apples:
  P = mean log1p(CP10K-normalized expression) over pluripotency markers
  D = mean log1p(CP10K-normalized expression) over cardiomyocyte markers

The 1 GB counts matrix is streamed in MatrixMarket coordinate form: we accumulate
per-cell total UMIs (for CP10K normalization) and the raw counts of only the ~20
marker genes, so we never hold the full 230k x 39k matrix in memory.

Output: data/processed/cardio_cell_scores.parquet
        columns: cell, diffday, individual, type, P_raw, D_raw
"""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

D = Path("data/raw/cardio_gse175634")
PLURI = ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4", "UTF1", "TDGF1"]
CARDIAC = ["TNNT2", "MYH6", "MYH7", "ACTC1", "NPPA", "MYL7", "TNNI1", "TNNI3",
           "RYR2", "NKX2-5", "GATA4", "TBX5", "MYL2"]
N_GENES, N_CELLS = 38943, 230786


def main():
    genes = pd.read_csv(D / "GSE175634_gene_indices_counts.tsv.gz", sep="\t")
    name_to_idx = dict(zip(genes["gene_name"].astype(str), genes["gene_index"].astype(int)))
    pluri_idx = {name_to_idx[g] for g in PLURI if g in name_to_idx}
    cardiac_idx = {name_to_idx[g] for g in CARDIAC if g in name_to_idx}
    marker_idx = pluri_idx | cardiac_idx
    # 1-based gene_index -> column position in the per-cell marker arrays
    pluri_cols = {gi: k for k, gi in enumerate(sorted(pluri_idx))}
    cardiac_cols = {gi: k for k, gi in enumerate(sorted(cardiac_idx))}
    print(f"markers: {len(pluri_idx)} pluripotency, {len(cardiac_idx)} cardiac")

    meta = pd.read_csv(D / "GSE175634_cell_metadata.tsv.gz", sep="\t",
                       usecols=["cell", "diffday", "individual", "type"])
    ci = pd.read_csv(D / "GSE175634_cell_indices.tsv.gz", sep="\t")
    assert (ci["cell_name"].values == meta["cell"].values).all(), "cell order mismatch"

    cell_total = np.zeros(N_CELLS, dtype=np.float64)
    pluri_raw = np.zeros((N_CELLS, len(pluri_idx)), dtype=np.float64)
    cardiac_raw = np.zeros((N_CELLS, len(cardiac_idx)), dtype=np.float64)

    print("streaming counts matrix...")
    with gzip.open(D / "GSE175634_cell_counts.mtx.gz", "rt") as f:
        # skip comment header
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        nrows, ncols, nnz = (int(x) for x in line.split())
        gene_is_row = (nrows == N_GENES)
        print(f"  header: {nrows} x {ncols}, nnz={nnz}, gene_is_row={gene_is_row}")

        for n, line in enumerate(f):
            i, j, v = line.split()
            i = int(i); j = int(j); v = float(v)
            g, c = (i, j) if gene_is_row else (j, i)
            cell_total[c - 1] += v
            if g in marker_idx:
                if g in pluri_cols:
                    pluri_raw[c - 1, pluri_cols[g]] = v
                else:
                    cardiac_raw[c - 1, cardiac_cols[g]] = v
            if (n + 1) % 50_000_000 == 0:
                print(f"  {(n + 1) // 1_000_000}M entries...")

    # CP10K normalize + log1p, then mean over each marker program
    tot = np.where(cell_total > 0, cell_total, 1.0)
    scale = 1e4 / tot
    pluri_log = np.log1p(pluri_raw * scale[:, None])     # (N, n_pluri)
    cardiac_log = np.log1p(cardiac_raw * scale[:, None])  # (N, n_cardiac)
    P_raw = pluri_log.mean(axis=1)
    D_raw = cardiac_log.mean(axis=1)

    out = meta.copy()
    out["P_raw"] = P_raw
    out["D_raw"] = D_raw
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/processed/cardio_cell_scores.csv.gz", index=False)
    print("[SAVED] data/processed/cardio_cell_scores.csv.gz", out.shape)

    # per-gene log1p(CP10K) matrices for Phase 5 representation-sensitivity
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    np.savez_compressed(
        "data/processed/cardio_marker_log.npz",
        pluri_log=pluri_log.astype(np.float32),
        cardiac_log=cardiac_log.astype(np.float32),
        pluri_genes=np.array([idx_to_name[gi] for gi in sorted(pluri_idx)]),    # column order
        cardiac_genes=np.array([idx_to_name[gi] for gi in sorted(cardiac_idx)]),
        diffday=meta["diffday"].to_numpy().astype(str),
    )
    print("[SAVED] data/processed/cardio_marker_log.npz")

    # biological-trend validation (population medians per timepoint)
    order = ["day0", "day1", "day3", "day5", "day7", "day11", "day15"]
    print("\nPopulation medians (raw mean-log1p score):")
    print(f"  {'day':<7}{'n':>8}{'P_med':>10}{'D_med':>10}")
    for d in order:
        s = out[out["diffday"] == d]
        print(f"  {d:<7}{len(s):>8}{s['P_raw'].median():>10.3f}{s['D_raw'].median():>10.3f}")


if __name__ == "__main__":
    main()
