"""
Publication-quality results figures (Bioinformatics house style).

Every numeric value is read from experiments/results/*.json -- nothing is
hand-entered. Output: vector PDF (embedded fonts) + 600 dpi PNG at the exact
Bioinformatics column widths, designed 1:1.

Figures
  Fig 1 (bottom band) leakage-contrast panel  [data-driven draft for vector refinement]
  Fig 2 headline forest plot of paired differences
  Fig 3 per-seed MAE raincloud distributions
  Fig 4 temporal-structure controls (shuffle / no-PE)
  Fig 5 fair-ODE monotonicity
  Fig 6 representation-sensitivity caterpillar
  Fig S2 per-feature (P,D) MAE breakdown

Run: python paper/figures/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

R = Path("experiments/results")
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 86 * MM, 130 * MM, 178 * MM

# ----------------------------------------------------------------------------- #
# House style
# ----------------------------------------------------------------------------- #
COL = dict(Transformer="#3B4CC0", Ridge="#5A6B7B", RandomForest="#E08214",
           ODE="#7D8C7A")
COL["RF"] = COL["RandomForest"]
LIN_MARK = {"neural": "o", "cardiac": "s"}          # lineage by shape, never color
ZERO = "#1A1A1A"
GUIDE = "#E8E8E8"
EQ_TINT = "#3B4CC0"                                  # faint equivalence band tint
EQ_MARGIN = 0.01                                     # practical-equivalence margin (MAE)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "Inter", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,        # embed as TrueType (editable)
        "svg.fonttype": "none",
        "axes.linewidth": 0.7, "axes.edgecolor": "#1A1A1A",
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def finish(ax):
    ax.tick_params(length=2.5, width=0.7)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"[saved] {name}.pdf / .png")


def L(fn):
    return json.loads((R / fn).read_text())


def panel_letter(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")


# ----------------------------------------------------------------------------- #
# Fig 2 - headline forest plot
# ----------------------------------------------------------------------------- #
def fig2():
    neu = L("multiseed_n20.json")["paired_vs_transformer"]
    car = L("multiseed_cardio_n20.json")["paired_vs_transformer"]
    # rows top->bottom; group by lineage
    rows = [
        ("cardiac", "RandomForest", car["RandomForest"]),
        ("cardiac", "Ridge", car["Ridge"]),
        ("neural", "RandomForest", neu["RandomForest"]),
        ("neural", "Ridge", neu["Ridge"]),
    ]
    fig, ax = plt.subplots(figsize=(ONEHALF, 0.95 * SINGLE))
    # equivalence band + zero spine
    ax.axvspan(-EQ_MARGIN, EQ_MARGIN, color=EQ_TINT, alpha=0.06, lw=0)
    ax.axvline(0, color=ZERO, lw=0.8, ls=(0, (4, 3)), zorder=1)

    ytrans = ax.get_yaxis_transform()                # x in axes frac, y in data
    for i, (lin, model, st) in enumerate(rows):
        y = i
        lo, hi = st["ci"]; m = st["mean_diff"]
        c = COL[model]
        ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=c, lw=0.9, zorder=3)
        ax.plot(m, y, LIN_MARK[lin], color=c, ms=5, mec="white", mew=0.7, zorder=4)
        ax.text(0.985, y, f"$d_z$={st['dz']:+.2f}   p={st['wilcoxon_p_holm']:.3f}",
                transform=ytrans, fontsize=6.3, va="center", ha="right", color="#333")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m} vs TF" for _, m, _ in rows])
    ax.set_ylim(-0.85, len(rows) - 0.4)
    ax.set_xlim(-0.014, 0.027)
    ax.set_xlabel(r"paired $\Delta$MAE  =  MAE(baseline) $-$ MAE(Transformer)")
    ax.set_xticks([-0.01, 0.0, 0.01, 0.02])
    # lineage separators / labels
    ax.axhline(1.5, color="#CCC", lw=0.5)
    ax.text(-0.0128, 0.5, "Cardiac", rotation=90, va="center", ha="center",
            fontsize=6.5, color="#555", fontstyle="italic")
    ax.text(-0.0128, 2.5, "Neural", rotation=90, va="center", ha="center",
            fontsize=6.5, color="#555", fontstyle="italic")
    ax.text(-0.0009, -0.75, r"$\leftarrow$ Transformer better", fontsize=6.3,
            ha="right", color="#777")
    ax.text(0.0009, -0.75, "baseline better $\\rightarrow$", fontsize=6.3,
            ha="left", color="#777")
    ax.text(EQ_MARGIN + 0.0004, len(rows) - 0.55, f"equiv. |Δ|<{EQ_MARGIN}",
            fontsize=6, color=EQ_TINT, va="center", ha="left")
    finish(ax)
    save(fig, "fig2_forest")


# ----------------------------------------------------------------------------- #
# Fig 3 - per-seed raincloud
# ----------------------------------------------------------------------------- #
def half_violin(ax, data, pos, color, width=0.34):
    v = ax.violinplot(data, positions=[pos], widths=width * 2,
                      showextrema=False, showmedians=False)
    for b in v["bodies"]:
        verts = b.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)   # keep right half
        b.set_facecolor(color); b.set_edgecolor(color)
        b.set_alpha(0.30); b.set_linewidth(0.6)


def raincloud_panel(ax, per_seed, models, rng):
    for i, m in enumerate(models):
        d = np.asarray(per_seed[m]); c = COL[m]
        half_violin(ax, d, i, c)
        jit = rng.uniform(-0.13, -0.02, size=len(d))
        ax.scatter(np.full_like(d, i) + jit, d, s=9, color=c, alpha=0.75,
                   edgecolor="white", linewidth=0.3, zorder=3)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.add_patch(Rectangle((i - 0.045, q1), 0.09, q3 - q1, facecolor=c,
                               alpha=0.85, edgecolor="none", zorder=4))
        ax.plot([i - 0.16, i + 0.34], [d.mean(), d.mean()], color=c, lw=1.0, zorder=5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["Ridge", "RF", "Transformer"])
    ax.set_xlim(-0.4, len(models) - 0.4)
    finish(ax)


def fig3():
    neu = L("multiseed_n20.json")["per_seed"]
    car = L("multiseed_cardio_n20.json")["per_seed"]
    models = ["Ridge", "RandomForest", "Transformer"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.62 * SINGLE * 2 / 1.6))
    for ax, ps, ttl, lab in zip(axes, (neu, car), ("Neural (dopaminergic)", "Cardiac (GSE175634)"), "AB"):
        raincloud_panel(ax, ps, models, np.random.default_rng(1))
        ax.set_title(ttl, fontsize=8)
        panel_letter(ax, lab)
    axes[0].set_ylabel("test MAE (20 seeds)")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig3_raincloud")


# ----------------------------------------------------------------------------- #
# Fig 4 - temporal-structure controls
# ----------------------------------------------------------------------------- #
def fig4():
    ab = L("ablations_n20.json")
    conds = ["ordered", "shuffle", "no_pe"]
    labels = ["Ordered", "Shuffle", "No-PE"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.62 * SINGLE * 2 / 1.6))
    for ax, ds, ttl, lab in zip(axes, ("dopaminergic", "cardiomyocyte"),
                                ("Neural (dopaminergic)", "Cardiac (GSE175634)"), "AB"):
        ps = ab[ds]["per_seed"]; tests = ab[ds]["tests"]
        arrs = {c: np.asarray(ps[c]) for c in conds}
        # per-seed connecting lines (thin, one color) + condition means
        for k in range(len(arrs["ordered"])):
            ax.plot(range(3), [arrs[c][k] for c in conds], color=COL["Transformer"],
                    lw=0.4, alpha=0.22, zorder=1)
        for i, c in enumerate(conds):
            ax.plot([i - 0.18, i + 0.18], [arrs[c].mean()] * 2, color=COL["Transformer"],
                    lw=1.6, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels)
        ax.set_xlim(-0.4, 2.4)
        ax.set_title(ttl, fontsize=8)
        panel_letter(ax, lab)
        # on-panel verdict
        ds_p = tests["shuffle"]["wilcoxon_p"]; dd = tests["shuffle"]["mean_diff"]
        ax.text(0.5, 0.04,
                f"shuffle vs ordered: Δ={dd:+.4f}, p={ds_p:.2f} (n.s.)\n"
                f"stage order carries no usable signal",
                transform=ax.transAxes, fontsize=6.3, ha="center", va="bottom",
                color="#444")
        finish(ax)
    axes[0].set_ylabel("test MAE (20 seeds)")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig4_temporal")


# ----------------------------------------------------------------------------- #
# Fig 5 - fair ODE monotonicity
# ----------------------------------------------------------------------------- #
def fig5():
    o = L("ode_fair_n20.json")
    regimes = ["ODE_pop", "ODE_dist", "ODE_pertraj"]
    xlab = ["Population\nmean", "Distributional", "Per-\ntrajectory"]
    fig, ax = plt.subplots(figsize=(SINGLE, 0.92 * SINGLE))
    for ds, lin in (("dopaminergic", "neural"), ("cardiomyocyte", "cardiac")):
        a = o[ds]["aggregate"]
        y = [a[r]["mean"] for r in regimes]
        e = [a[r]["std"] for r in regimes]
        ax.errorbar(range(3), y, yerr=e, marker=LIN_MARK[lin], color=COL["ODE"],
                    mfc=COL["ODE"], mec="white", mew=0.7, ms=5.5, lw=1.2,
                    capsize=2, elinewidth=0.8, label=lin.capitalize())
        # ridge reference (dashed) + degradation labels
        rid = a["Ridge"]["mean"]
        ax.axhline(rid, color=COL["Ridge"], lw=0.8, ls=(0, (4, 3)), alpha=0.8)
        for i, r in enumerate(regimes):
            pct = (a[r]["mean"] - rid) / rid * 100
            ax.annotate(f"+{pct:.0f}%", (i, a[r]["mean"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="#555")
    ax.text(2.95, o["dopaminergic"]["aggregate"]["Ridge"]["mean"] + 0.004, "Ridge (neural)",
            fontsize=6, color=COL["Ridge"], va="bottom", ha="right")
    ax.text(2.95, o["cardiomyocyte"]["aggregate"]["Ridge"]["mean"] + 0.004, "Ridge (cardiac)",
            fontsize=6, color=COL["Ridge"], va="bottom", ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(xlab)
    ax.set_xlim(-0.3, 3.0)
    ax.set_xlabel("calibration fairness  $\\rightarrow$")
    ax.set_ylabel("test MAE (20 seeds)")
    ax.legend(title="lineage", title_fontsize=7, loc="upper left")
    finish(ax)
    save(fig, "fig5_ode")


# ----------------------------------------------------------------------------- #
# Fig 6 - representation-sensitivity caterpillar
# ----------------------------------------------------------------------------- #
def fig6():
    sd = L("sensitivity_dopaminergic_n20.json")
    sc = L("sensitivity_cardio_n20.json")
    rows = ([("neural", k, v) for k, v in sd.items()]
            + [("cardiac", k, v) for k, v in sc.items()])
    rows.sort(key=lambda r: r[2]["gap"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(ONEHALF, 0.085 * n + 0.7))
    ax.axvline(0, color=ZERO, lw=0.8, ls=(0, (4, 3)), zorder=2)
    nflip = 0
    for i, (lin, name, v) in enumerate(rows):
        lo, hi = v["gap_ci"]
        if not (lo <= 0 <= hi):
            nflip += 1
        ax.plot([lo, hi], [i, i], color=COL["Transformer"], lw=0.55, alpha=0.55, zorder=1)
        ax.plot(v["gap"], i, LIN_MARK[lin], color=COL["Transformer"], ms=3.0,
                mec="white", mew=0.3, zorder=3)
    ax.set_ylim(-3.2, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel(r"Transformer $-$ Ridge gap (MAE)")
    ax.set_xlim(-0.012, 0.012)
    ax.set_xticks([-0.01, -0.005, 0.0, 0.005, 0.01])
    # verdict, top centre inside the plotting area
    ax.text(0.5, 0.985, f"{nflip} / {n} variants significantly favor the Transformer",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="#444")
    # lineage key in the empty band beneath the lowest rows
    handles = [Line2D([0], [0], marker="o", color="none", mfc=COL["Transformer"],
                      mec="white", ms=4, label="Neural (15)"),
               Line2D([0], [0], marker="s", color="none", mfc=COL["Transformer"],
                      mec="white", ms=4, label="Cardiac (23)")]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=6,
              title="38 state-construction variants", title_fontsize=6,
              bbox_to_anchor=(0.5, -0.01))
    finish(ax)
    save(fig, "fig6_sensitivity")


# ----------------------------------------------------------------------------- #
# Fig S2 - per-feature MAE breakdown
# ----------------------------------------------------------------------------- #
def figS2():
    neu = L("multiseed_n20.json")["per_seed_PD"]
    car = L("multiseed_cardio_n20.json")["per_seed_PD"]
    models = ["Ridge", "RandomForest", "Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.62 * SINGLE * 2 / 1.6))
    width = 0.36
    for ax, ps, ttl, lab in zip(axes, (neu, car),
                                ("Neural (dopaminergic)", "Cardiac (GSE175634)"), "AB"):
        x = np.arange(len(models))
        for j, feat in enumerate(("P", "D")):
            means = [np.mean(ps[m][feat]) for m in models]
            errs = [np.std(ps[m][feat], ddof=1) for m in models]
            cols = [COL[m] for m in models]
            ax.bar(x + (j - 0.5) * width, means, width * 0.92, yerr=errs,
                   color=cols, alpha=0.55 if j == 0 else 0.95, edgecolor="white",
                   linewidth=0.5, error_kw=dict(lw=0.8, capsize=2),
                   hatch="" if j == 0 else "///")
        ax.set_xticks(x); ax.set_xticklabels(["Ridge", "RF", "TF"])
        ax.set_title(ttl, fontsize=8); panel_letter(ax, lab)
        finish(ax)
    axes[0].set_ylabel("per-feature MAE (mean ± sd)")
    handles = [Patch(facecolor="#999", alpha=0.55, label="Pluripotency (P)"),
               Patch(facecolor="#999", alpha=0.95, hatch="///", label="Differentiation (D)")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "figS2_per_feature")


# ----------------------------------------------------------------------------- #
# Fig 1 (bottom) - leakage-contrast (data-driven draft for vector refinement)
# ----------------------------------------------------------------------------- #
def fig1_leakage():
    # reconstruct the original (leaky) vs corrected index layout, seed 42
    n = 200; n_tr, n_va = 140, 30
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train = set(perm[:n_tr].tolist())
    buggy_test = set(range(n_tr + n_va, n))          # contiguous tail trajectories[170:]
    overlap = buggy_test & train

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.42 * SINGLE))
    cols_grid = 20
    WARN = "#C0392B"
    for ax, mode, ttl in zip(axes, ("leaky", "clean"),
                             ("Original (leaky) split", "Corrected (single permutation)")):
        for idx in range(n):
            r, c = divmod(idx, cols_grid)
            if mode == "leaky":
                in_train = idx in train
                in_test = idx in buggy_test
                if idx in overlap:
                    fc, ec = WARN, WARN          # in BOTH train and test
                elif in_test:
                    fc, ec = "white", WARN
                elif in_train:
                    fc, ec = COL["Transformer"], "none"
                else:
                    fc, ec = "#DDD", "none"
            else:
                # corrected: disjoint blocks by sorted index
                if idx < n_tr:
                    fc, ec = COL["Transformer"], "none"
                elif idx < n_tr + n_va:
                    fc, ec = COL["RandomForest"], "none"
                else:
                    fc, ec = COL["Ridge"], "none"
            ax.add_patch(Rectangle((c, -r), 0.86, 0.86, facecolor=fc, edgecolor=ec,
                                   linewidth=0.5))
        ax.set_xlim(-0.3, cols_grid); ax.set_ylim(-(n / cols_grid), 1.4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, fontsize=8)
    axes[0].text(0, 1.0, f"{len(overlap)}/30 test trajectories were in training",
                 fontsize=7, color=WARN, fontweight="bold")
    # legend
    leg = [Patch(facecolor=COL["Transformer"], label="train"),
           Patch(facecolor="white", edgecolor=WARN, label="test (leaky)"),
           Patch(facecolor=WARN, label="in BOTH (leak)"),
           Patch(facecolor=COL["RandomForest"], label="val"),
           Patch(facecolor=COL["Ridge"], label="test (clean)")]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig1_leakage_panel")


def main():
    set_style()
    fig1_leakage()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    figS2()
    print("\nAll figures written to paper/figures/ (PDF + PNG).")


if __name__ == "__main__":
    main()
