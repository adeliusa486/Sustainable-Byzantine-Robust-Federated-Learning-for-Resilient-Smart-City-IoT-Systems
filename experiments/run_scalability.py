"""
Scalability Experiment — Byzantine Rate Sweep
==============================================

Reproduces the results in paper Figure 4:
Accuracy vs. Byzantine fraction ∈ {0.10, 0.20, 0.30, 0.40}
across all methods under label flipping attack.

Usage:
    python experiments/run_scalability.py --use_synthetic
    python experiments/run_scalability.py  # Full (requires TON_IoT)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amfta.utils.logging_utils import setup_logging
from amfta.utils.metrics import aggregate_metrics
from amfta.utils.reproducibility import PAPER_SEEDS
from training.federated_runner import FederatedRunner, RunConfig

logger = logging.getLogger(__name__)

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "amfta"]
BYZ_RATES = [0.0, 0.10, 0.20, 0.30, 0.40]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="label_flipping")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    scalability_results = {}

    for method in METHODS:
        method_results = {}
        for byz_rate in BYZ_RATES:
            seed_results = []
            for seed in args.seeds:
                attack = args.attack if byz_rate > 0 else "none"
                config = RunConfig(
                    method=method,
                    num_clients=args.num_clients,
                    num_rounds=args.num_rounds,
                    byzantine_fraction=byz_rate,
                    attack_type=attack,
                    seed=seed,
                    use_synthetic=args.use_synthetic,
                    results_dir=args.results_dir,
                )
                runner = FederatedRunner(config)
                history = runner.run()
                final = history[-1] if history else {}
                seed_results.append({k: v for k, v in final.items() if isinstance(v, float)})

            summary = aggregate_metrics(seed_results)
            method_results[byz_rate] = {k: v for k, v in summary.items()}
            print(
                f"  {method:14s} byz={byz_rate:.0%}: "
                f"acc={summary['accuracy'][0]:.4f}±{summary['accuracy'][1]:.4f}"
            )

        scalability_results[method] = method_results

    out_path = Path(args.results_dir) / "scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(scalability_results, f, indent=2, default=str)
    print(f"\nScalability results saved to {out_path}")


if __name__ == "__main__":
    main()
