"""
Academic Figure Generation Script for Federated Learning Experiments
=====================================================================
Generates publication-quality figures using standard Matplotlib settings
indistinguishable from human-created IEEE/Elsevier journal plots.
"""

import os
import glob
import json
import re
from collections import defaultdict
from statistics import mean, pstdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use classic academic publication serif typography and styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
METHOD_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed\nMean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
    "amfta_noq": "AMFTA-ND",
}
METHOD_LEGEND_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
    "amfta_noq": "AMFTA-ND",
}

# Classic Elsevier / IEEE publication palette with distinct markers and line styles
METHOD_STYLE = {
    "fedavg":       {"color": "#7F7F7F", "marker": "o", "ls": "--", "lw": 1.6},
    "trimmed_mean": {"color": "#1F77B4", "marker": "s", "ls": "-.", "lw": 1.6},
    "krum":         {"color": "#2CA02C", "marker": "^", "ls": ":",  "lw": 1.8},
    "fltrust":      {"color": "#D62728", "marker": "v", "ls": ":",  "lw": 1.8},
    "feddbc":       {"color": "#FF7F0E", "marker": "D", "ls": "--", "lw": 1.6},
    "amfta":        {"color": "#800080", "marker": "P", "ls": "-",  "lw": 2.2},
    "amfta_noq":    {"color": "#8C564B", "marker": "*", "ls": "-",  "lw": 1.8},
}

PATTERN = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)

def collect_data():
    latest = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        name = os.path.basename(path)
        m = PATTERN.match(name)
        if not m:
            continue
        key = (m["method"], float(m["byz"]), m["attack"], int(m["seed"]))
        ts = m["ts"]
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, path)

    grouped = defaultdict(dict)
    for (method, byz, attack, seed), (ts, path) in latest.items():
        try:
            rounds = json.load(open(path))
            if not isinstance(rounds, list) or not rounds:
                continue
            tail = rounds[-5:]
            avg_acc = sum(r["accuracy"] for r in tail if "accuracy" in r) / len(tail)
            grouped[(method, byz, attack)][seed] = avg_acc * 100.0
        except Exception as e:
            print(f"WARN: could not read {path}: {e}")
    return grouped

def compute_stats(grouped):
    stats = {}
    for (method, byz, attack), seeds_dict in grouped.items():
        vals = list(seeds_dict.values())
        if not vals:
            continue
        m_val = mean(vals)
        s_val = pstdev(vals) if len(vals) > 1 else 0.0
        stats[(method, byz, attack)] = (m_val, s_val, len(vals))
    return stats

def apply_journal_axes_style(ax):
    """Applies classic Elsevier/IEEE journal spine and tick styling."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#A0A0A0")
    ax.tick_params(direction="in", length=5, width=1.0, top=True, right=True, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

def plot_line_chart(stats, attack_type, filename):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if attack_type == "gaussian_noise":
        byz_rates = [0.10, 0.20, 0.30]
    else:
        byz_rates = [0.10, 0.20, 0.30, 0.40]
    
    for method in METHOD_ORDER:
        if method not in METHOD_LEGEND_LABEL:
            continue
        style = METHOD_STYLE.get(method, {"color": "black", "marker": "o", "ls": "-", "lw": 1.5})
        
        x_vals, y_vals, y_errs = [], [], []
        for byz in byz_rates:
            if (method, byz, attack_type) in stats:
                m_val, s_val, _ = stats[(method, byz, attack_type)]
                x_vals.append(byz * 100)
                y_vals.append(m_val)
                y_errs.append(s_val)
        
        if x_vals:
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                label=METHOD_LEGEND_LABEL[method],
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=style["lw"],
                capsize=3, markersize=6, markeredgecolor="black", markeredgewidth=0.6
            )
            
    apply_journal_axes_style(ax)
    ax.set_xlabel("Byzantine Client Fraction (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks([b * 100 for b in byz_rates])
    ax.set_xticklabels([f"{int(b * 100)}%" for b in byz_rates])
    ax.set_ylim(35, 102)
    
    # Legend inside plot box, classic Elsevier journal style
    ax.legend(
        loc="lower left", ncol=2, frameon=True,
        edgecolor="black", facecolor="white", framealpha=0.95,
        columnspacing=1.2, handlelength=2.0
    )
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, filename.replace(".pdf", ".png")), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

def plot_bar_chart(stats, filename):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    byz = 0.30
    
    methods = [m for m in METHOD_ORDER if any((m, byz, att) in stats for att in ["label_flipping", "gaussian_noise"])]
    n_methods = len(methods)
    
    x = np.arange(n_methods)
    width = 0.35
    
    flip_means, flip_errs = [], []
    gauss_means, gauss_errs = [], []
    
    for m in methods:
        if (m, byz, "label_flipping") in stats:
            fm, fe, _ = stats[(m, byz, "label_flipping")]
            flip_means.append(fm)
            flip_errs.append(fe)
        else:
            flip_means.append(0.0)
            flip_errs.append(0.0)
            
        if (m, byz, "gaussian_noise") in stats:
            gm, ge, _ = stats[(m, byz, "gaussian_noise")]
            gauss_means.append(gm)
            gauss_errs.append(ge)
        else:
            gauss_means.append(0.0)
            gauss_errs.append(0.0)
            
    apply_journal_axes_style(ax)
    
    # Classic Elsevier palette: Earthy Burnt Orange and Forest Green with solid black bar borders
    ax.bar(x - width/2, flip_means, width, label="Label Flipping (30%)",
           color="#D35400", edgecolor="black", linewidth=1.0, alpha=0.90)
    ax.bar(x + width/2, gauss_means, width, label="Gaussian Noise (30%)",
           color="#27AE60", edgecolor="black", linewidth=1.0, alpha=0.90)
    
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=0, ha="center")
    ax.set_ylim(0, 108)
    
    # Legend placed cleanly inside plot area
    ax.legend(
        loc="upper right", frameon=True,
        edgecolor="black", facecolor="white", framealpha=0.95
    )
    
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf")
    fig.savefig(os.path.join(FIGURES_DIR, filename.replace(".pdf", ".png")), format="png")
    plt.close(fig)
    print(f"Generated {filename}")

def print_summary_table(stats):
    print("\n" + "="*80)
    print("SUMMARY TABLE OF ALL EXPERIMENT RESULTS FOUND IN Phase 1 (results/)")
    print("="*80)
    print(f"{'Method':<15} | {'Attack Type':<16} | {'Byz Fraction':<12} | {'Accuracy (Mean ± Std)':<22} | {'Seeds':<5}")
    print("-"*80)
    
    for method in METHOD_ORDER:
        for attack in ["label_flipping", "gaussian_noise"]:
            for byz in [0.10, 0.20, 0.30, 0.40]:
                key = (method, byz, attack)
                if key in stats:
                    m_val, s_val, n_seeds = stats[key]
                    label = METHOD_LEGEND_LABEL.get(method, method)
                    print(f"{label:<15} | {attack:<16} | {byz*100:5.1f}%       | {m_val:6.2f}% ± {s_val:5.2f}%        | {n_seeds:<5}")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("Collecting real data from results/...")
    grouped = collect_data()
    stats = compute_stats(grouped)
    
    print_summary_table(stats)
    
    print("Generating figures...")
    plot_line_chart(stats, "label_flipping", "fig_lineflip.pdf")
    plot_line_chart(stats, "gaussian_noise", "fig_linegauss.pdf")
    plot_bar_chart(stats, "fig_bar30.pdf")
    print("All figures successfully generated in figures/ directory.")
