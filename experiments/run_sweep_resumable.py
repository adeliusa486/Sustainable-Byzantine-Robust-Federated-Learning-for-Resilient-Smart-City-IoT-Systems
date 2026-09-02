"""
Resumable Full Sweep Driver
===========================

Runs every (method, byz_rate, attack, seed) as a SEPARATE subprocess.
Before each, checks results/ for a JSON produced AFTER --since (default: the
code-fix cutoff) for that exact config; if found, the config is skipped.
This makes the multi-hour sweep safe against interruption/reaping: just
re-launch this script and it continues where it left off.

Usage:
    python experiments/run_sweep_resumable.py            # full grid
    python experiments/run_sweep_resumable.py --attacks label_flipping
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta"]
RATES = [0.10, 0.20, 0.30, 0.40]
ATTACKS = ["label_flipping", "gaussian_noise"]
SEEDS = [42, 0, 1]

# Only count result files newer than this as "valid" (i.e., produced after the
# aggregation-bug fixes on 2026-06-25 ~23:00). Format: YYYYMMDD_HHMMSS.
DEFAULT_SINCE = "20260625_230000"
PAT = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)


def already_done(method, byz, attack, seed, since) -> bool:
    for path in glob.glob(os.path.join(RESULTS, f"{method}_byz{byz}_{attack}_seed{seed}_*.json")):
        m = PAT.match(os.path.basename(path))
        if m and m["ts"] >= since and m["method"] == method and m["attack"] == attack:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES)
    ap.add_argument("--attacks", nargs="+", default=ATTACKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--num_rounds", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--num_clients", type=int, default=100)
    ap.add_argument("--since", default=DEFAULT_SINCE)
    args = ap.parse_args()

    combos = [
        (m, b, a, s)
        for a in args.attacks
        for b in args.rates
        for m in args.methods
        for s in args.seeds
    ]
    total = len(combos)
    print(f"[{datetime.now():%H:%M:%S}] Sweep: {total} configs "
          f"({args.num_rounds} rounds, seeds={args.seeds})", flush=True)

    done = run = 0
    for i, (method, byz, attack, seed) in enumerate(combos, 1):
        byz_str = f"{byz:.1f}".rstrip("0").rstrip(".") if False else str(byz)
        # run_main saves files as byz0.3 etc — match its float formatting
        byz_tag = f"{byz}"
        if already_done(method, byz_tag, attack, seed, args.since):
            done += 1
            print(f"[{i}/{total}] SKIP {method} byz{byz} {attack} seed{seed} (done)", flush=True)
            continue
        print(f"[{i}/{total}] RUN  {method} byz{byz} {attack} seed{seed} "
              f"@ {datetime.now():%H:%M:%S}", flush=True)
        cmd = [
            sys.executable, os.path.join(HERE, "run_main.py"),
            "--method", method,
            "--byzantine_fraction", str(byz),
            "--attack", attack,
            "--num_clients", str(args.num_clients),
            "--num_rounds", str(args.num_rounds),
            "--local_epochs", str(args.local_epochs),
            "--seeds", str(seed),
            "--log_interval", "10",
        ]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"   !! returncode {r.returncode} for {method} byz{byz} {attack} seed{seed}", flush=True)
        else:
            run += 1

    print(f"[{datetime.now():%H:%M:%S}] Sweep complete. ran={run} skipped={done} total={total}", flush=True)


if __name__ == "__main__":
    main()
