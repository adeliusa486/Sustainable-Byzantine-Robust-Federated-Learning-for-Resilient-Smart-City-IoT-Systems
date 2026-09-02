"""
Evaluation Visualisation
=========================

Generates all paper figures from saved results JSON files.

Figures produced:
  - Figure 3: Accuracy per round (convergence curves)
  - Figure 4: Bar chart — F1 vs Byzantine fraction per method
  - Figure 5: Trust score distribution across rounds
  - Figure 6: Ablation comparison bar chart
  - Figure 7: Confusion matrix heatmap

Usage:
    python evaluation/visualize.py --results_dir results --output_dir figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (for CI/server environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants (paper-matching aesthetics)
# ---------------------------------------------------------------------------

COLORS = {
    "amfta":        "#2563EB",   # Blue
    "fltrust":      "#16A34A",   # Green
    "trimmed_mean": "#DC2626",   # Red
    "krum":         "#D97706",   # Amber
    "fedavg":       "#7C3AED",   # Purple
}

METHOD_LABELS = {
    "amfta":        "AMFTA (ours)",
    "fltrust":      "FLTrust",
    "trimmed_mean": "Trimmed Mean",
    "krum":         "Krum",
    "fedavg":       "FedAvg",
}

LINE_STYLES = {
    "amfta":        "-",
    "fltrust":      "--",
    "trimmed_mean": "-.",
    "krum":         ":",
    "fedavg":       (0, (3, 1, 1, 1)),
}

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Figure 3: Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(
    history_by_method: Dict[str, List[Dict[str, Any]]],
    metric: str = "f1",
    title: str = "Convergence Curves",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot per-round metric curves for each method.

    Parameters
    ----------
    history_by_method : dict {method_name: list of round metrics}
    metric            : str  'accuracy', 'f1', 'precision', 'recall'
    title             : str  Figure title.
    output_path       : Path  If provided, save to this file.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for method, history in history_by_method.items():
        rounds = [h["round"] for h in history]
        values = [h.get(metric, 0.0) for h in history]
        label = METHOD_LABELS.get(method, method.upper())
        color = COLORS.get(method, "gray")
        ls    = LINE_STYLES.get(method, "-")

        ax.plot(rounds, values, label=label, color=color, linestyle=ls,
                linewidth=2.0, alpha=0.9)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Convergence plot saved: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 4: Bar chart — Final F1 vs Byzantine fraction
# ---------------------------------------------------------------------------

def plot_byz_comparison(
    results: Dict[str, Dict],
    metric: str = "f1",
    attack: str = "label_flipping",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart comparing methods at each Byzantine fraction.

    Parameters
    ----------
    results : {method: {byz_rate: {metric: (mean, std)}}}
    """
    methods = list(results.keys())
    # Extract Byzantine rates from first method
    byz_rates = sorted(results[methods[0]].keys()) if methods else [0.1, 0.2, 0.3, 0.4]

    n_rates = len(byz_rates)
    n_methods = len(methods)
    x = np.arange(n_rates)
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, method in enumerate(methods):
        means, stds = [], []
        for rate in byz_rates:
            rate_key = str(rate) if isinstance(list(results[method].keys())[0], str) else rate
            entry = results[method].get(rate_key, {}).get(metric, (0.0, 0.0))
            means.append(entry[0] if isinstance(entry, (list, tuple)) else entry)
            stds.append(entry[1] if isinstance(entry, (list, tuple)) else 0.0)

        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=METHOD_LABELS.get(method, method.upper()),
            color=COLORS.get(method, "gray"),
            yerr=stds, capsize=3, alpha=0.85,
        )

    ax.set_xlabel("Byzantine Fraction")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Final {metric.upper()} vs Byzantine Fraction ({attack.replace('_', ' ').title()})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in byz_rates])
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Byzantine comparison plot saved: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 5: Trust score distribution
# ---------------------------------------------------------------------------

def plot_trust_distribution(
    trust_history: Dict[int, List[float]],
    byzantine_ids: set,
    rounds_to_plot: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plots showing trust score distribution at selected rounds.

    Parameters
    ----------
    trust_history  : {client_id: [trust_score per round]}
    byzantine_ids  : set of Byzantine client IDs
    rounds_to_plot : list of round indices to display (1-indexed)
    """
    all_rounds = len(next(iter(trust_history.values()), []))
    if rounds_to_plot is None:
        rounds_to_plot = [
            max(1, all_rounds // 5),
            max(1, all_rounds // 2),
            all_rounds,
        ]

    fig, axes = plt.subplots(1, len(rounds_to_plot), figsize=(4 * len(rounds_to_plot), 4))
    if len(rounds_to_plot) == 1:
        axes = [axes]

    for ax, round_idx in zip(axes, rounds_to_plot):
        r = min(round_idx - 1, all_rounds - 1)
        benign_scores = [trust_history[cid][r] for cid in trust_history if cid not in byzantine_ids]
        byz_scores    = [trust_history[cid][r] for cid in trust_history if cid in byzantine_ids]

        ax.boxplot(
            [benign_scores, byz_scores],
            labels=["Benign", "Byzantine"],
            notch=False, patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="navy", linewidth=2),
        )
        ax.set_title(f"Round {round_idx}")
        ax.set_ylabel("Trust Score")
        ax.set_ylim(0, 1.05)

    plt.suptitle("Trust Score Distribution: Benign vs Byzantine", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Trust distribution plot saved: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 6: Ablation comparison
# ---------------------------------------------------------------------------

def plot_ablation(
    ablation_results: Dict[str, Dict],
    metric: str = "f1",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart for AMFTA ablation study results."""
    variants = list(ablation_results.keys())
    means = [ablation_results[v].get(metric, (0.0, 0.0))[0] for v in variants]
    stds  = [ablation_results[v].get(metric, (0.0, 0.0))[1] for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))

    palette = ["#CBD5E1", "#93C5FD", "#60A5FA", "#2563EB"]
    bars = ax.bar(variants, means, yerr=stds, capsize=4,
                  color=palette[:len(variants)], alpha=0.9, edgecolor="white", linewidth=0.5)

    # Annotate bars with values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study — {metric.upper()} Comparison")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("AMFTA Variant")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Ablation plot saved: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 7: Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualise a 2×2 confusion matrix as a heatmap."""
    class_names = class_names or ["Normal", "Attack"]

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Confusion matrix saved: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate AMFTA experiment figures.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--metric", default="f1", choices=["accuracy", "f1", "precision", "recall"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    # --- Main results ---
    main_path = results_dir / "main_results.json"
    if main_path.exists():
        with open(main_path) as f:
            main_results = json.load(f)
        logger.info("Loaded main_results.json")
        # Restructure for plot_byz_comparison
        structured = {}
        for key, v in main_results.items():
            parts = key.split("_")
            method = parts[0]
            if method not in structured:
                structured[method] = {}
            try:
                byz_rate = float(parts[1])
                structured[method][byz_rate] = v
            except (ValueError, IndexError):
                pass

        if structured:
            plot_byz_comparison(
                structured,
                metric=args.metric,
                output_path=output_dir / f"fig4_byz_comparison_{args.metric}.png",
            )

    # --- Ablation results ---
    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_results = json.load(f)
        plot_ablation(
            ablation_results,
            metric=args.metric,
            output_path=output_dir / f"fig6_ablation_{args.metric}.png",
        )
        logger.info("Ablation figure generated.")

    plt.close("all")
    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
