"""
Focused Study Orchestrator (final paper scope, ablation-first)
==============================================================

Final scoped experiment plan after capping the paper at two attack families
and a 30% Byzantine ceiling, with the no-trusted-data ablation prioritised
because it answers the key FedDBC review objection ("why does AMFTA need
trusted validation data?").

  Threat model:
    1. label_flipping  (data-poisoning)   @ rates 0.1/0.2/0.3
    2. gaussian_noise  (model-poisoning)  @ rates 0.1/0.2/0.3

  Standard methods: fedavg, trimmed_mean, krum, fltrust, feddbc, amfta
  Ablation method : amfta_noq  (AMFTA with Factor III / val-buffer disabled)
  3 seeds (42,123,456); 25 rounds.

Phases (run in order; resumable — completed configs are skipped):
  PHASE 1  amfta_noq ablation on BOTH attacks  (18 runs, ~5h)  <-- decisive
  PHASE 2  gaussian_noise on 6 standard methods (54 runs, ~14h)
  (label_flipping on standard methods is already complete -> skipped)
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
SWEEP = os.path.join(HERE, "run_sweep_resumable.py")

SEEDS = ["42", "123", "456"]
ROUNDS = "25"
SINCE = "20260626_000000"
RATES = ["0.1", "0.2", "0.3"]  # capped at 30%

STANDARD = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta"]

# (label, methods, attacks)
PHASES = [
    # Decisive ablation first: no-trusted-data AMFTA on both attack families.
    ("ablation_noq", ["amfta_noq"], ["label_flipping", "gaussian_noise"]),
    # Then complete the gaussian phase for the standard methods.
    ("gaussian_main", STANDARD, ["gaussian_noise"]),
    # label_flipping on standard methods already done -> included so it is
    # re-verified/skipped, costs nothing.
    ("labelflip_verify", STANDARD, ["label_flipping"]),
]


def main():
    for label, methods, attacks in PHASES:
        print(f"\n########## PHASE: {label} methods={methods} attacks={attacks} ##########", flush=True)
        cmd = [
            sys.executable, SWEEP,
            "--methods", *methods,
            "--attacks", *attacks,
            "--rates", *RATES,
            "--seeds", *SEEDS,
            "--num_rounds", ROUNDS,
            "--since", SINCE,
        ]
        subprocess.run(cmd, cwd=os.path.abspath(os.path.join(HERE, "..")))
    print("\n########## FOCUSED STUDY COMPLETE ##########", flush=True)


if __name__ == "__main__":
    main()
