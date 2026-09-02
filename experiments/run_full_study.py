"""
Full Study Orchestrator (resumable, sequential phases)
======================================================

Runs the complete experiment plan for the FedDBC-competitive paper as ONE
resumable queue, so only a single GPU process is active at a time.  Re-launch
after any interruption — completed configs are skipped.

Phases (all: 6 methods = fedavg, trimmed_mean, krum, fltrust, feddbc, amfta;
        3 seeds = 42,123,456; 25 rounds):
  1. label_flipping  @ rates 0.1/0.2/0.3/0.4   (main table + degradation curve)
  2. sign_flipping   @ rates 0.3/0.4           (collusion model-poisoning)
  3. mimicry         @ rates 0.3/0.4           (adaptive, norm-camouflaged)
  4. gaussian_noise  @ rates 0.3/0.4           (untargeted model-poisoning)
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
SWEEP = os.path.join(HERE, "run_sweep_resumable.py")

SEEDS = ["42", "123", "456"]
ROUNDS = "25"
SINCE = "20260626_000000"  # only count results produced after the new attacks/FedDBC landed

PHASES = [
    ("label_flipping", ["0.1", "0.2", "0.3", "0.4"]),
    ("sign_flipping",  ["0.3", "0.4"]),
    ("mimicry",        ["0.3", "0.4"]),
    ("gaussian_noise", ["0.3", "0.4"]),
]


def main():
    for attack, rates in PHASES:
        print(f"\n########## PHASE: {attack} rates={rates} ##########", flush=True)
        cmd = [
            sys.executable, SWEEP,
            "--attacks", attack,
            "--rates", *rates,
            "--seeds", *SEEDS,
            "--num_rounds", ROUNDS,
            "--since", SINCE,
        ]
        subprocess.run(cmd, cwd=os.path.abspath(os.path.join(HERE, "..")))
    print("\n########## FULL STUDY COMPLETE ##########", flush=True)


if __name__ == "__main__":
    main()
