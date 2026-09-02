"""
Standalone Inference Script
=============================

Load a trained AMFTA global model checkpoint and classify network flows.

Usage:
    # Single sample from CLI
    python scripts/inference.py --checkpoint checkpoints/amfta_best.pt \
        --features 0.12 0.34 0.05 ... (45 values)

    # From CSV file
    python scripts/inference.py --checkpoint checkpoints/amfta_best.pt \
        --input data/raw/NF-TON-IoT.csv --output results/predictions.csv

    # Without checkpoint (fresh model — for pipeline testing)
    python scripts/inference.py --use_synthetic --n_samples 100
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

from amfta.models.local_mlp import LocalMLP
from amfta.utils.logging_utils import setup_logging

logger = logging.getLogger("amfta.inference")


def load_model(checkpoint_path: str | None) -> LocalMLP:
    """Load model from checkpoint or return a fresh initialised model."""
    model = LocalMLP()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        logger.info("Model loaded from: %s", checkpoint_path)
    else:
        logger.warning(
            "No checkpoint at '%s'. Using freshly initialised model.",
            checkpoint_path
        )
    model.eval()
    return model


def predict_batch(
    model: LocalMLP,
    X: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batch inference.

    Returns
    -------
    (probabilities, predictions) — both shape (N,)
    """
    all_probs = []
    n = len(X)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(X[start : start + batch_size].astype(np.float32))
            probs = model(batch).numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs)
    preds_arr = (probs_arr >= threshold).astype(int)
    return probs_arr, preds_arr


def main():
    parser = argparse.ArgumentParser(
        description="AMFTA inference — classify IoT network flows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV file with network flow features")
    parser.add_argument("--output", type=str, default="results/predictions.csv",
                        help="Output CSV file for predictions")
    parser.add_argument("--features", type=float, nargs="+", default=None,
                        help="Single sample: 45 feature values (min-max normalised)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for attack classification")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic data for testing")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of synthetic samples")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load model
    model = load_model(args.checkpoint)
    logger.info("Model ready: %s", model)

    # ── Single sample from --features ──────────────────────────────────────
    if args.features is not None:
        if len(args.features) != 45:
            logger.error("Expected 45 features, got %d", len(args.features))
            sys.exit(1)
        x = np.array([args.features], dtype=np.float32)
        probs, preds = predict_batch(model, x, args.threshold)
        print(f"\nPrediction:")
        print(f"  Attack probability : {probs[0]:.4f}")
        print(f"  Classification     : {'ATTACK' if preds[0] else 'NORMAL'}")
        print(f"  Threshold          : {args.threshold}")
        return

    # ── Synthetic data ──────────────────────────────────────────────────────
    if args.use_synthetic:
        from amfta.data.partitioning import generate_synthetic_data
        X, y_true = generate_synthetic_data(args.n_samples, n_features=45, seed=42)
        logger.info("Generated %d synthetic samples", len(X))
    elif args.input:
        import pandas as pd
        df = pd.read_csv(args.input, low_memory=False)
        # Drop label/identifier columns if present
        drop_cols = [c for c in ["label", "Label", "Attack", "type", "src_ip", "dst_ip",
                                  "src_port", "dst_port", "Timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_true = None
        logger.info("Loaded %d samples from %s", len(X), args.input)
    else:
        logger.error("Provide --features, --input, or --use_synthetic.")
        parser.print_help()
        sys.exit(1)

    # ── Run inference ───────────────────────────────────────────────────────
    start = time.perf_counter()
    probs, preds = predict_batch(model, X, args.threshold)
    elapsed = time.perf_counter() - start

    attack_count  = int(preds.sum())
    normal_count  = int((preds == 0).sum())
    throughput    = len(X) / elapsed

    print(f"\n{'='*50}")
    print(f"  Inference Results")
    print(f"{'='*50}")
    print(f"  Samples processed : {len(X):,}")
    print(f"  Attacks detected  : {attack_count:,} ({100*attack_count/len(X):.1f}%)")
    print(f"  Normal traffic    : {normal_count:,} ({100*normal_count/len(X):.1f}%)")
    print(f"  Throughput        : {throughput:,.0f} samples/sec")
    print(f"  Total latency     : {elapsed*1000:.1f} ms")

    if y_true is not None:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, zero_division=0)
        print(f"\n  Accuracy (vs ground truth) : {acc:.4f}")
        print(f"  F1-Score (vs ground truth) : {f1:.4f}")

    # ── Save predictions ────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "attack_probability", "predicted_label"])
        for i, (p, pred) in enumerate(zip(probs, preds)):
            writer.writerow([i, f"{p:.6f}", int(pred)])

    logger.info("Predictions saved to: %s", args.output)


if __name__ == "__main__":
    main()
