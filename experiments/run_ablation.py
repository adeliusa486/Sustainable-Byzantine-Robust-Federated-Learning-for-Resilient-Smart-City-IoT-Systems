"""
Ablation Study Runner — Reproduces Paper Table III
====================================================

Tests AMFTA with individual trust factors disabled to verify the contribution
of each component.

Ablation variants:
  - AMFTA-S  : Similarity only (disable H and Q)
  - AMFTA-SH : Similarity + History (disable Q)
  - AMFTA-SQ : Similarity + Quality (disable H)
  - AMFTA    : Full model (all three factors)

Usage:
    python experiments/run_ablation.py --byzantine_fraction 0.30
    python experiments/run_ablation.py --use_synthetic --num_rounds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amfta.utils.logging_utils import setup_logging
from amfta.utils.metrics import aggregate_metrics
from amfta.utils.reproducibility import PAPER_SEEDS
from training.federated_runner import FederatedRunner, RunConfig

logger = logging.getLogger(__name__)

ABLATION_VARIANTS = {
    "AMFTA-S":    {"disable_factor_h": True,  "disable_factor_q": True},
    "AMFTA-SH":   {"disable_factor_h": False, "disable_factor_q": True},
    "AMFTA-SQ":   {"disable_factor_h": True,  "disable_factor_q": False},
    "AMFTA-Full": {"disable_factor_h": False, "disable_factor_q": False},
}


def run_ablation(variant_name, variant_flags, args):
    seed_results = []
    for seed in args.seeds:
        logger.info("Ablation: %s | seed=%d", variant_name, seed)

        config = RunConfig(
            method="amfta",
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            byzantine_fraction=args.byzantine_fraction,
            attack_type=args.attack,
            seed=seed,
            use_synthetic=args.use_synthetic,
            results_dir=args.results_dir,
            **variant_flags,
        )

        runner = FederatedRunner(config)
        history = runner.run()
        final = history[-1] if history else {}
        seed_results.append({k: v for k, v in final.items() if isinstance(v, float)})

    summary = aggregate_metrics(seed_results)
    print(f"\n  {variant_name:12s}: acc={summary['accuracy'][0]:.4f}±{summary['accuracy'][1]:.4f} "
          f"f1={summary['f1'][0]:.4f}±{summary['f1'][1]:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--byzantine_fraction", type=float, default=0.30)
    parser.add_argument("--attack", default="label_flipping")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    print("\n" + "=" * 60)
    print("  AMFTA Ablation Study")
    print(f"  Byzantine fraction: {args.byzantine_fraction:.0%} | Attack: {args.attack}")
    print("=" * 60)

    results = {}
    for name, flags in ABLATION_VARIANTS.items():
        results[name] = run_ablation(name, flags, args)

    # Save
    out_path = Path(args.results_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    main()
