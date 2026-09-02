"""Regenerate fig_scalability from the measured aggregation timings.

The figure shipped with the original submission plotted the earlier, unmeasured
numbers and no longer matches Table 'runtime'. This redraws it from
AGG_MS_BY_N in resource_model.py, which now holds direct measurements.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import resource_model as rm

FIG = Path(__file__).parent.parent / "manuscript" / "figures"

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

STYLE = {
    "Trimmed Mean": ("s", "#c44e52"), "FedAvg": ("o", "#8c8c8c"),
    "FedDBC": ("D", "#8172b3"), "AMFTA-ND": ("*", "#55a868"),
    "AMFTA": ("P", "#4c72b0"), "FLTrust": ("v", "#937860"),
    "Krum": ("^", "#dd8452"),
}

Ns = sorted(rm.AGG_MS_BY_N)
fig, ax = plt.subplots(figsize=(5.4, 3.8))

for m, (mk, col) in STYLE.items():
    ys = [rm.AGG_MS_BY_N[n][m] for n in Ns]
    big = m == "AMFTA-ND"
    ax.plot(Ns, ys, marker=mk, color=col, lw=1.6 if big else 1.2,
            ms=9 if big else 5, label=m, zorder=5 if big else 3)

# reference slopes
x = np.array(Ns, dtype=float)
ax.plot(x, 0.30 * (x / 50), ls=":", lw=0.9, color="#555", zorder=1)
ax.plot(x, 8.33 * (x / 50) ** 2, ls="--", lw=0.9, color="#555", zorder=1)
ax.text(520, 0.30 * 10 * 1.15, r"$\mathcal{O}(N)$", fontsize=8, color="#555")
ax.text(300, 8.33 * 36 * 1.4, r"$\mathcal{O}(N^2)$", fontsize=8, color="#555")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(Ns); ax.set_xticklabels(Ns)
ax.set_xlabel("Clients per round $N$")
ax.set_ylabel("Server aggregation time per round (ms)")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
fig.savefig(FIG / "fig_scalability.pdf")
print("wrote fig_scalability.pdf from measured timings")
