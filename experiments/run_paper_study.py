"""
Run the complete study reported in the manuscript.
==================================================

Every table in the paper corresponds to one block below. Blocks are
independent and resumable: each writes one tidy CSV under ``--out``, and a
block whose CSV already contains a given (method, attack, rho, alpha, seed)
row is skipped unless ``--force`` is given. That matters because the full
study is long; you can run it in pieces and stop whenever.

    python experiments/run_paper_study.py --block clean
    python experiments/run_paper_study.py --block labelflip --seeds 42 123 456
    python experiments/run_paper_study.py --block all --seeds 42 123 456 789 1024 \
        2024 7 13 99 2718

Blocks and the manuscript tables they produce
---------------------------------------------
  clean        Clean accuracy at rho = 0
  labelflip    Accuracy under label flipping, four attacker fractions
  gaussian     Accuracy under Gaussian-noise model poisoning
  adaptive     AGR-tailored adaptive attack at rho = 0.30
  extras       Coordinate-wise Median, Multi-Krum, FoolsGold at rho = 0.30
  normclip     NormClip-Only against AMFTA-ND, isolating norm rescaling
  alpha01      Severe heterogeneity, Dirichlet alpha = 0.1
  scalability  Server aggregation wall-clock against client population

Model class
-----------
Defaults to ``--model logistic``, which is the configuration the manuscript
describes. Pass ``--model mlp`` to reproduce the released experiment logs
instead. The two are not comparable and should never be mixed in one table.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from amfta.utils.logging_utils import setup_logging
from training.federated_runner import FederatedRunner, RunConfig

logger = logging.getLogger(__name__)

CORE = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
RHOS = [0.10, 0.20, 0.30, 0.40]
FIELDS = ["block", "method", "attack", "rho", "alpha", "seed", "round",
          "accuracy", "f1", "precision", "recall"]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def out_path(out_dir: Path, block: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{block}.csv"


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            done.add((r["method"], r["attack"], r["rho"], r["alpha"], r["seed"]))
    return done


def append_rows(path: Path, rows: list) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def run_one(block, method, attack, rho, alpha, seed, args) -> list:
    cfg = RunConfig(
        method=method,
        model_class=args.model,
        model_input_dim=args.input_dim,
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.lr,
        local_batch_size=args.batch_size,
        byzantine_fraction=rho,
        attack_type=attack,
        alpha_dirichlet=alpha,
        repartition=(abs(alpha - 0.5) > 1e-9),
        trim_fraction=max(rho, 0.01),   # Trimmed Mean receives the true rho
        seed=seed,
        results_dir=str(Path(args.out) / "_runs"),
        log_interval=args.rounds + 1,
    )
    t0 = time.time()
    history = FederatedRunner(cfg).run()
    logger.info("  %s / %s / rho=%.2f / a=%.2f / seed %d  -> %.1fs",
                method, attack, rho, alpha, seed, time.time() - t0)

    rows = []
    for h in history:
        rows.append({
            "block": block, "method": method, "attack": attack,
            "rho": f"{rho:.2f}", "alpha": f"{alpha:.2f}", "seed": seed,
            "round": h.get("round"),
            "accuracy": h.get("accuracy"), "f1": h.get("f1"),
            "precision": h.get("precision"), "recall": h.get("recall"),
        })
    return rows


def sweep(block, combos, args):
    """combos: iterable of (method, attack, rho, alpha)."""
    path = out_path(Path(args.out), block)
    done = set() if args.force else load_done(path)
    total = len(combos) * len(args.seeds)
    n = 0
    for method, attack, rho, alpha in combos:
        for seed in args.seeds:
            n += 1
            key = (method, attack, f"{rho:.2f}", f"{alpha:.2f}", str(seed))
            if key in done:
                logger.info("[%d/%d] skip (done): %s", n, total, key)
                continue
            logger.info("[%d/%d] %s", n, total, key)
            try:
                append_rows(path, run_one(block, method, attack, rho, alpha, seed, args))
            except Exception as e:
                import traceback
                logger.error("FAILED %s: %s\n%s", key, e, traceback.format_exc())
    logger.info("block '%s' -> %s", block, path)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def block_clean(args):
    sweep("clean", [(m, "none", 0.0, 0.5) for m in CORE], args)


def block_labelflip(args):
    sweep("labelflip",
          [(m, "label_flipping", r, 0.5) for m in CORE for r in RHOS], args)


def block_gaussian(args):
    sweep("gaussian",
          [(m, "gaussian_noise", r, 0.5) for m in CORE for r in RHOS], args)


def block_adaptive(args):
    sweep("adaptive", [(m, "adaptive", 0.30, 0.5) for m in CORE], args)


def block_extras(args):
    extra = ["median", "multikrum", "foolsgold", "amfta_noq"]
    sweep("extras",
          [(m, a, 0.30, 0.5) for m in extra
           for a in ("label_flipping", "gaussian_noise")], args)


def block_normclip(args):
    sweep("normclip",
          [(m, a, r, 0.5) for m in ("normclip", "amfta_noq")
           for a in ("label_flipping", "gaussian_noise") for r in RHOS], args)


def block_alpha01(args):
    sweep("alpha01",
          [(m, "label_flipping", 0.30, 0.1) for m in CORE], args)


def block_scalability(args):
    """Measure server aggregation time alone, excluding local training."""
    from amfta.aggregation.baselines import build_aggregator
    import amfta.aggregation.extra_baselines  # noqa: F401
    from amfta.models.logistic import build_global_model

    path = out_path(Path(args.out), "scalability")
    rows = []
    d_model = build_global_model(args.model, input_dim=args.input_dim)
    template = {k: v.clone() for k, v in d_model.state_dict().items()}

    for N in (50, 100, 200, 500):
        for method in CORE:
            for seed in args.seeds[:3]:
                torch.manual_seed(seed)
                updates = {
                    i: {k: torch.randn_like(v.float()) * 0.01
                        for k, v in template.items()}
                    for i in range(N)
                }
                name = {"amfta": "amfta", "amfta_noq": "amfta"}.get(method, method)
                if name == "amfta":
                    from amfta.aggregation.amfta import AMFTAAggregator
                    agg = AMFTAAggregator(
                        num_clients=N,
                        use_quality_eval=(method == "amfta"),
                        **({} if method == "amfta"
                           else {"alpha_s": 0.57, "alpha_h": 0.43, "alpha_q": 0.0}),
                    )
                elif name == "krum":
                    agg = build_aggregator("krum", {"num_byzantine": int(0.3 * N)})
                elif name == "fltrust":
                    agg = build_aggregator("fltrust")
                else:
                    agg = build_aggregator(name)

                kw = {}
                if name == "fltrust":
                    kw["root_update"] = {k: torch.randn_like(v.float()) * 0.01
                                         for k, v in template.items()}

                t0 = time.perf_counter()
                for _ in range(args.timing_reps):
                    try:
                        agg.aggregate(d_model, updates, **kw)
                    except Exception as e:
                        logger.error("timing failed %s N=%d: %s", method, N, e)
                        break
                ms = 1000 * (time.perf_counter() - t0) / args.timing_reps
                rows.append({
                    "block": "scalability", "method": method, "attack": "none",
                    "rho": "0.00", "alpha": "0.50", "seed": seed, "round": N,
                    "accuracy": ms, "f1": "", "precision": "", "recall": "",
                })
                logger.info("N=%4d %-12s %8.2f ms", N, method, ms)
    append_rows(path, rows)
    logger.info("block 'scalability' -> %s  (accuracy column holds ms/round)", path)


BLOCKS = {
    "clean": block_clean,
    "labelflip": block_labelflip,
    "gaussian": block_gaussian,
    "adaptive": block_adaptive,
    "extras": block_extras,
    "normclip": block_normclip,
    "alpha01": block_alpha01,
    "scalability": block_scalability,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", default="all", choices=list(BLOCKS) + ["all"])
    ap.add_argument("--model", default="logistic", choices=["logistic", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 456, 789, 1024, 2024, 7, 13, 99, 2718],
                    help="10 seeds by default; the manuscript's three are the "
                         "first three, so a 3-seed run is a prefix of a 10-seed one")
    ap.add_argument("--num_clients", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--input_dim", type=int, default=45)
    ap.add_argument("--timing_reps", type=int, default=5)
    ap.add_argument("--out", default="results_paper")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level, Path(args.out) / "study.log")
    logger.info("model=%s  seeds=%s  rounds=%d  clients=%d",
                args.model, args.seeds, args.rounds, args.num_clients)

    names = list(BLOCKS) if args.block == "all" else [args.block]
    for b in names:
        logger.info("=== block: %s ===", b)
        BLOCKS[b](args)


if __name__ == "__main__":
    main()
