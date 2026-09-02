"""
Main Experiment Runner — Reproduces Paper Table II
====================================================

Runs all methods (FedAvg, Trimmed Mean, Krum, FLTrust, AMFTA) across
5 random seeds under label flipping and Gaussian noise attacks at
Byzantine fraction ∈ {0.10, 0.20, 0.30, 0.40}.

Expected runtime: ~2-4 hours on GPU for full reproduction (100 clients,
100 rounds, 5 seeds, 5 methods, 4+ Byzantine rates).

For quick validation, use --num_rounds 20 --num_clients 20.

Usage:
    python experiments/run_main.py --method amfta --byzantine_fraction 0.30
    python experiments/run_main.py --all --use_synthetic  # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from amfta.utils.logging_utils import setup_logging
from amfta.utils.metrics import aggregate_metrics
from amfta.utils.reproducibility import PAPER_SEEDS
from training.federated_runner import FederatedRunner, RunConfig

logger = logging.getLogger(__name__)

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
BYZ_RATES = [0.10, 0.20, 0.30, 0.40]
ATTACK_TYPES = ["label_flipping", "gaussian_noise", "sign_flipping", "mimicry"]


def run_single(
    method: str,
    byzantine_fraction: float,
    attack_type: str,
    seeds: list,
    args: argparse.Namespace,
) -> dict:
    """Run one (method, byz_rate, attack, seeds) combination."""
    seed_results = []

    for seed in seeds:
        logger.info(
            "Running: method=%s byz=%.0f%% attack=%s seed=%d",
            method, 100 * byzantine_fraction, attack_type, seed,
        )

        config = RunConfig(
            method=method,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            byzantine_fraction=byzantine_fraction,
            attack_type=attack_type if byzantine_fraction > 0 else "none",
            seed=seed,
            use_synthetic=args.use_synthetic,
            results_dir=args.results_dir,
            log_interval=args.log_interval,
        )

        runner = FederatedRunner(config)
        history = runner.run()

        # Report final-round metrics (last round)
        final = history[-1] if history else {}
        seed_results.append({k: v for k, v in final.items() if isinstance(v, float)})

    # Aggregate across seeds
    summary = aggregate_metrics(seed_results)
    print_summary(method, byzantine_fraction, attack_type, summary)
    return summary


def print_summary(method, byz_rate, attack, summary):
    """Pretty-print aggregated results."""
    print(f"\n{'=' * 60}")
    print(f"  {method.upper()} | byz={byz_rate:.0%} | attack={attack}")
    print(f"{'=' * 60}")
    for metric, (mean, std) in summary.items():
        print(f"  {metric:12s}: {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="AMFTA main experiment runner"
    )
    parser.add_argument("--method", choices=METHODS + ["all"], default="amfta")
    parser.add_argument("--byzantine_fraction", type=float, default=0.30)
    parser.add_argument("--attack", choices=ATTACK_TYPES, default="label_flipping")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--all", action="store_true", help="Run all methods × all Byzantine rates")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data (no TON_IoT download required)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    setup_logging(args.log_level, Path(args.results_dir) / "run_main.log")

    all_results = {}

    if args.all:
        methods_to_run = METHODS
        byz_rates = BYZ_RATES
        attacks_to_run = ATTACK_TYPES
    else:
        methods_to_run = [args.method]
        byz_rates = [args.byzantine_fraction]
        attacks_to_run = [args.attack]

    for method in methods_to_run:
        for byz_rate in byz_rates:
            for attack in attacks_to_run:
                if byz_rate == 0.0:
                    attack = "none"
                key = f"{method}_{byz_rate:.2f}_{attack}"
                try:
                    result = run_single(method, byz_rate, attack, args.seeds, args)
                    all_results[key] = result
                except Exception as e:
                    import traceback
                    logger.error("FAILED: %s - %s\n%s", key, e, traceback.format_exc())
                    all_results[key] = {"error": str(e)}

    # Save consolidated results
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = Path(args.results_dir) / "main_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("All results saved to %s", out_path)


if __name__ == "__main__":
    main()
