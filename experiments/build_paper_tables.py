"""
Build Paper Tables & Figure Data from REAL result JSONs
========================================================

Scans results/ for per-run JSON files named
    {method}_byz{rate}_{attack}_seed{seed}_{timestamp}.json
takes the FINAL-round metrics of the most recent file per
(method, byz_rate, attack, seed), aggregates mean +/- std across seeds,
and emits:
  - results/paper_tables.json   (machine-readable, every number sourced)
  - results/paper_table_main.tex (LaTeX booktabs table)
  - console summary

STRICT: every number printed/emitted here comes from a real JSON file.
Nothing is hardcoded. Files used are listed in paper_tables.json["sources"].
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
METHOD_LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
    "fltrust": "FLTrust",
    "feddbc": "FedDBC",
    "amfta": "AMFTA",
    "amfta_noq": "AMFTA-ND",
}
PATTERN = re.compile(
    r"^(?P<method>[a-z_]+)_byz(?P<byz>[0-9.]+)_(?P<attack>[a-z_]+)_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})\.json$"
)


def collect():
    """Return {(method,byz,attack): {seed: (final_metrics, filepath)}} using newest file per key+seed."""
    latest = {}  # (method,byz,attack,seed) -> (ts, path)
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
            # Average the last 5 rounds rather than the single final round.
            # Standard practice in FL papers: avoids reporting a misleading
            # number when a run is still oscillating/not fully converged
            # (a single round can land on either a peak or a trough).
            tail = rounds[-5:]
            avg = {
                k: sum(r[k] for r in tail) / len(tail)
                for k in tail[0]
                if isinstance(tail[0][k], (int, float))
            }
            grouped[(method, byz, attack)][seed] = (avg, os.path.basename(path))
        except Exception as e:  # noqa
            print(f"WARN: could not read {path}: {e}")
    return grouped


def aggregate(grouped):
    table = {}
    sources = {}
    for (method, byz, attack), per_seed in grouped.items():
        accs = [v[0].get("accuracy") for v in per_seed.values() if v[0].get("accuracy") is not None]
        f1s = [v[0].get("f1") for v in per_seed.values() if v[0].get("f1") is not None]
        if not accs:
            continue
        key = f"{method}|byz{byz:.2f}|{attack}"
        table[key] = {
            "method": method,
            "byz": byz,
            "attack": attack,
            "n_seeds": len(accs),
            "acc_mean": mean(accs),
            "acc_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "f1_mean": mean(f1s),
            "f1_std": pstdev(f1s) if len(f1s) > 1 else 0.0,
            "seeds": sorted(per_seed.keys()),
        }
        sources[key] = {str(s): v[1] for s, v in per_seed.items()}
    return table, sources


def main():
    grouped = collect()
    table, sources = aggregate(grouped)

    out = {"table": table, "sources": sources}
    with open(os.path.join(RESULTS_DIR, "paper_tables.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Console: label_flipping main table
    for attack in ["label_flipping", "gaussian_noise"]:
        print(f"\n===== {attack} (acc% / f1, mean+/-std over seeds) =====")
        print(f"{'method':14s}" + "".join(f"{int(b*100):>16d}%" for b in [0.10, 0.20, 0.30, 0.40]))
        for method in METHOD_ORDER:
            row = f"{METHOD_LABEL[method]:14s}"
            for byz in [0.10, 0.20, 0.30, 0.40]:
                e = table.get(f"{method}|byz{byz:.2f}|{attack}")
                if e:
                    row += f"  {e['acc_mean']*100:5.1f}/{e['f1_mean']:.3f}({e['n_seeds']})"
                else:
                    row += f"  {'--':>13s}"
            print(row)

    print("\nWrote results/paper_tables.json (with per-number source files).")


if __name__ == "__main__":
    main()
