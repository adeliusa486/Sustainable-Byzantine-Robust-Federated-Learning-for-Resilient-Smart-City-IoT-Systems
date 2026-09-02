"""
Figures for the sustainability sections.

Writes into ../figures/ :
  fig_pareto.pdf    robustness-sustainability frontier at rho = 0.30
  fig_energy_rho.pdf  energy per retained accuracy point against rho
  fig_classes.pdf   per-tier energy budget and the relief scheduling provides
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import resource_model as rm

FIG = Path(__file__).parent.parent / "manuscript" / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

MARK = {
    "FedAvg": ("o", "#8c8c8c"),
    "Trimmed Mean": ("s", "#c44e52"),
    "Krum": ("^", "#dd8452"),
    "FLTrust": ("v", "#937860"),
    "FedDBC": ("D", "#8172b3"),
    "AMFTA": ("P", "#4c72b0"),
    "AMFTA-ND": ("*", "#55a868"),
}


def fig_pareto():
    r = rm.build(42)
    pbr = rm.pareto_by_rho()
    rows = pbr[0.30]["rows"]
    gated = set(pbr[0.30]["pareto_gated"])

    fig, ax = plt.subplots(figsize=(5.6, 3.9))

    ax.axhspan(rm.ACC_FLOOR, 100, color="#55a868", alpha=0.06, zorder=0)
    ax.axhline(rm.ACC_FLOOR, color="#55a868", lw=0.9, ls="--", zorder=1)
    ax.text(0.015, rm.ACC_FLOOR + 1.0, f"deployment accuracy floor ({rm.ACC_FLOOR:.0f}%)",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, color="#3d7a4b")

    for m, v in rows.items():
        mk, col = MARK[m]
        big = m in gated
        ax.scatter(v["total_j"], v["acc_worst"],
                   marker=mk, s=190 if big else 78,
                   facecolor=col, edgecolor="black",
                   linewidth=1.3 if big else 0.6,
                   zorder=5 if big else 4)
        dx, dy = 8, 5
        if m == "FedAvg":
            dx, dy = -8, 8
        if m == "Trimmed Mean":
            dx, dy = 8, -12
        if m == "FLTrust":
            dx, dy = 8, 2
        if m == "AMFTA-ND":
            dx, dy = 12, 2
        if m == "Krum":
            dx, dy = -6, 10
        if m == "FedDBC":
            dx, dy = -8, 10
        ax.annotate(m, (v["total_j"], v["acc_worst"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left",
                    fontsize=8, fontweight="bold" if big else "normal")

    # unconstrained frontier staircase, for reference
    pts = sorted([(v["total_j"], v["acc_worst"]) for v in rows.values()])
    fr, best = [], -np.inf
    for e, a in pts:
        if a > best:
            fr.append((e, a))
            best = a
    if len(fr) > 1:
        xs = [p[0] for p in fr]
        ys = [p[1] for p in fr]
        ax.step(xs, ys, where="post", color="#666", lw=0.8, ls=(0, (1, 2)), zorder=2)

    ax.set_xlabel("Total federation energy per run (J), modelled")
    ax.set_ylabel("Worst-case robust accuracy at $\\rho=0.30$ (%)")
    ax.set_xlim(515, 845)
    ax.set_ylim(35, 100)
    fig.savefig(FIG / "fig_pareto.pdf")
    plt.close(fig)
    print("wrote fig_pareto.pdf")


def fig_energy_rho():
    pbr = rm.pareto_by_rho()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    for m in rm.AGG_MS_N100:
        mk, col = MARK[m]
        ys = [pbr[rho]["rows"][m]["j_per_pp"] for rho in rm.RHOS]
        ax.plot(rm.RHOS, ys, marker=mk, color=col, lw=1.4, ms=5,
                label=m, zorder=5 if m == "AMFTA-ND" else 3)
    
    ax.set_xlabel("Attacker fraction $\\rho$")
    ax.set_ylabel("Energy per retained accuracy point (J/pp)")
    ax.set_xticks(rm.RHOS)

    ax = axes[1]
    for m in rm.AGG_MS_N100:
        mk, col = MARK[m]
        ys = [min(rm.ACC_LF[m][j], rm.ACC_GN[m][j]) for j in range(len(rm.RHOS))]
        ax.plot(rm.RHOS, ys, marker=mk, color=col, lw=1.4, ms=5,
                zorder=5 if m == "AMFTA-ND" else 3)
    ax.axhline(rm.ACC_FLOOR, color="#55a868", lw=0.9, ls="--")
    
    ax.set_xlabel("Attacker fraction $\\rho$")
    ax.set_ylabel("Worst-case robust accuracy (%)")
    ax.set_xticks(rm.RHOS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.13), fontsize=8)
    fig.savefig(FIG / "fig_energy_rho.pdf")
    plt.close(fig)
    print("wrote fig_energy_rho.pdf")


def fig_classes():
    r = rm.build(42)
    ras = r["ras"]
    names = [c.name for c in rm.CLASSES]
    labels = [f"Class {c.name}\n{c.radio}" for c in rm.CLASSES]
    before = [ras[f"class{n}_budget_pct_before"] for n in names]
    after = [ras[f"class{n}_budget_pct_after"] for n in names]
    part = [ras[f"class{n}_mean_p"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, before, w, label="uniform participation",
           color="#c44e52", edgecolor="black", lw=0.5)
    ax.bar(x + w / 2, after, w, label="resource-aware scheduling",
           color="#55a868", edgecolor="black", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Share of device energy budget\nconsumed per run (%)")
    ax.legend(fontsize=7.5, frameon=False)
    for xi, (b, a) in enumerate(zip(before, after)):
        if b > 0 and a > 0 and (1 - a / b) > 0.02:
            ax.text(xi, b * 1.6, f"$-${100*(1-a/b):.0f}%", ha="center",
                    fontsize=7.5, color="#3d7a4b")

    ax = axes[1]
    ax.bar(x, part, 0.5, color="#4c72b0", edgecolor="black", lw=0.5)
    ax.axhline(0.25, color="#c44e52", ls="--", lw=0.9)
    ax.text(-0.42, 0.29, "$p_{\\min}$", fontsize=8, color="#8f2b30", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Mean assigned participation rate $p_i$")
    ax.set_ylim(0, 1.12)

    fig.savefig(FIG / "fig_classes.pdf")
    plt.close(fig)
    print("wrote fig_classes.pdf")


if __name__ == "__main__":
    fig_pareto()
    fig_energy_rho()
    fig_classes()
