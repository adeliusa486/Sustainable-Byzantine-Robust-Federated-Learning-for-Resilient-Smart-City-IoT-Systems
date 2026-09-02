"""
The redesigned experimental study: a compact 20-client controlled federation.
=============================================================================

The 100-client campaign stays where it is and is reported as large-cohort
context. The main experiments move to 20 clients so that attack strength and
heterogeneity can be varied without a full re-run each time.

    python experiments/run_redesign.py --exp E1 E2 E3
    python experiments/run_redesign.py --exp all --seeds 42 123 456

Experiments
-----------
  E1  benign baseline, 7 methods
  E2  label-flipping robustness at 20% and 30%
  E3  Gaussian-noise robustness at 20% and 30%
  E4  heterogeneity stress, alpha = 0.1 against alpha = 0.5
  E5  AMFTA component ablation
  E6  sustainability: uniform participation against AMFTA-S, benign
  E7  joint test: AMFTA-ND against AMFTA-S under 20% label flipping
  E8  resource-spoofing sensitivity
  E9  aggregation timing at 20 / 50 / 100 clients

Blocks are resumable: a configuration already present in the CSV is skipped.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from amfta.utils.logging_utils import setup_logging
from training.federated_runner import FederatedRunner, RunConfig

logger = logging.getLogger(__name__)

CORE = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
FIELDS = ["exp", "method", "attack", "rho", "alpha", "spoof", "seed", "round",
          "accuracy", "f1", "precision", "recall"]


def path_for(out: Path, exp: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{exp}.csv"


def done_keys(p: Path) -> set:
    if not p.exists():
        return set()
    with p.open(newline="", encoding="utf8") as f:
        return {(r["method"], r["attack"], r["rho"], r["alpha"], r["spoof"], r["seed"])
                for r in csv.DictReader(f)}


def append(p: Path, rows: list) -> None:
    new = not p.exists()
    with p.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def run(exp, method, attack, rho, alpha, seed, args, spoof=0.0, extra=None):
    cfg = RunConfig(
        method=method, model_class="logistic", model_input_dim=args.input_dim,
        num_clients=args.clients, num_rounds=args.rounds,
        local_epochs=args.local_epochs, local_lr=args.lr,
        local_batch_size=args.batch_size,
        byzantine_fraction=rho, attack_type=attack,
        alpha_dirichlet=alpha, repartition=True,
        trim_fraction=max(rho, 0.01), spoof_fraction=spoof,
        seed=seed, results_dir=str(Path(args.out) / "_runs"),
        log_interval=args.rounds + 1,
        **(extra or {}),
    )
    t0 = time.time()
    hist = FederatedRunner(cfg).run()
    logger.info("  %s %s rho=%.2f a=%.2f spoof=%.2f seed=%d -> %.1f min",
                exp, method, rho, alpha, spoof, seed, (time.time() - t0) / 60)
    return [{"exp": exp, "method": method, "attack": attack, "rho": f"{rho:.2f}",
             "alpha": f"{alpha:.2f}", "spoof": f"{spoof:.2f}", "seed": seed,
             "round": h.get("round"), "accuracy": h.get("accuracy"),
             "f1": h.get("f1"), "precision": h.get("precision"),
             "recall": h.get("recall")} for h in hist]


def sweep(exp, combos, args):
    p = path_for(Path(args.out), exp)
    seen = set() if args.force else done_keys(p)
    total = len(combos) * len(args.seeds)
    n = 0
    for method, attack, rho, alpha, spoof, extra in combos:
        for seed in args.seeds:
            n += 1
            key = (method, attack, f"{rho:.2f}", f"{alpha:.2f}", f"{spoof:.2f}", str(seed))
            if key in seen:
                logger.info("[%d/%d] skip %s", n, total, key)
                continue
            logger.info("[%d/%d] %s", n, total, key)
            try:
                append(p, run(exp, method, attack, rho, alpha, seed, args, spoof, extra))
            except Exception as e:
                import traceback
                logger.error("FAILED %s: %s\n%s", key, e, traceback.format_exc())
    logger.info("%s -> %s", exp, p)


def E1(a): sweep("E1", [(m, "none", 0.0, 0.5, 0.0, None) for m in CORE], a)
def E2(a): sweep("E2", [(m, "label_flipping", r, 0.5, 0.0, None)
                        for m in CORE for r in (0.20, 0.30)], a)
def E3(a): sweep("E3", [(m, "gaussian_noise", r, 0.5, 0.0, None)
                        for m in CORE for r in (0.20, 0.30)], a)
def E4(a): sweep("E4", [(m, att, 0.20, 0.1, 0.0, None)
                        for m in CORE for att in ("label_flipping", "gaussian_noise")], a)


def E5(a):
    """Component ablation. Each variant removes one part of the trust rule."""
    sweep("E5", [
        ("fedavg",    "label_flipping", 0.20, 0.5, 0.0, None),
        ("normclip",  "label_flipping", 0.20, 0.5, 0.0, None),
        ("amfta_noq", "label_flipping", 0.20, 0.5, 0.0, {"disable_factor_h": True}),
        ("amfta_noq", "label_flipping", 0.20, 0.5, 0.0, None),
        ("amfta",     "label_flipping", 0.20, 0.5, 0.0, None),
    ], a)


def E6(a): sweep("E6", [("amfta_noq", "none", 0.0, 0.5, 0.0, None),
                        ("amfta_s",   "none", 0.0, 0.5, 0.0, None)], a)
def E7(a): sweep("E7", [("amfta_noq", "label_flipping", 0.20, 0.5, 0.0, None),
                        ("amfta_s",   "label_flipping", 0.20, 0.5, 0.0, None)], a)
def E8(a): sweep("E8", [("amfta_s", "label_flipping", 0.20, 0.5, s, None)
                        for s in (0.0, 0.5, 1.0)], a)


def E9(a):
    """Aggregation timing only; no training and no dataset needed."""
    import torch
    from amfta.aggregation.baselines import build_aggregator
    import amfta.aggregation.extra_baselines  # noqa: F401
    from amfta.models.logistic import build_global_model

    torch.set_num_threads(1)
    model = build_global_model("logistic", input_dim=a.input_dim)
    tpl = {k: v.clone() for k, v in model.state_dict().items()}
    rows = []
    for N in (20, 50, 100):
        for method in CORE:
            torch.manual_seed(42)
            upd = {i: {k: torch.randn_like(v.float().cpu()) * 0.01
                       for k, v in tpl.items()} for i in range(N)}
            if method in ("amfta", "amfta_noq"):
                from amfta.aggregation.amfta import AMFTAAggregator
                kw = {} if method == "amfta" else {"alpha_s": 0.57, "alpha_h": 0.43,
                                                   "alpha_q": 0.0}
                agg = AMFTAAggregator(num_clients=N,
                                      use_quality_eval=(method == "amfta"), **kw)
            elif method == "krum":
                agg = build_aggregator("krum", {"num_byzantine": int(0.3 * N)})
            else:
                agg = build_aggregator(method)
            kw2 = {}
            if method == "fltrust":
                kw2["root_update"] = {k: torch.randn_like(v.float()) * 0.01
                                      for k, v in tpl.items()}
            for _ in range(3):
                agg.aggregate(model, upd, **kw2)
            samples = []
            for _ in range(25):
                t0 = time.perf_counter()
                agg.aggregate(model, upd, **kw2)
                samples.append(1000 * (time.perf_counter() - t0))
            samples.sort()
            ms = samples[len(samples) // 2]
            rows.append({"exp": "E9", "method": method, "attack": "none",
                         "rho": "0.00", "alpha": "0.50", "spoof": "0.00",
                         "seed": 42, "round": N, "accuracy": ms,
                         "f1": "", "precision": "", "recall": ""})
            logger.info("  N=%3d %-13s %8.2f ms", N, method, ms)
    append(path_for(Path(a.out), "E9"), rows)


EXPS = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5,
        "E6": E6, "E7": E7, "E8": E8, "E9": E9}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--input_dim", type=int, default=41)
    ap.add_argument("--out", default="results_redesign")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level, Path(args.out) / "redesign.log")
    names = list(EXPS) if "all" in args.exp else args.exp
    logger.info("clients=%d rounds=%d epochs=%d seeds=%s",
                args.clients, args.rounds, args.local_epochs, args.seeds)
    for e in names:
        logger.info("=== %s ===", e)
        EXPS[e](args)


if __name__ == "__main__":
    main()
