#!/usr/bin/env python3
"""
Dataset Setup Helper
====================

Guides users through setting up the TON_IoT dataset or generating
a synthetic substitute for quick prototyping.

Usage:
    python scripts/setup_data.py --synthetic          # Quick synthetic data
    python scripts/setup_data.py --verify             # Verify existing data
    python scripts/setup_data.py --preprocess         # Run full pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("amfta.setup")

TON_IOT_URL = "https://research.unsw.edu.au/projects/toniot-datasets"
EXPECTED_FILE = "data/raw/NF-TON-IoT.csv"


def verify_data():
    """Check whether processed data already exists."""
    required = [
        "data/processed/train.npz",
        "data/processed/val.npz",
        "data/processed/test.npz",
        "data/server/val_buffer.npz",
    ]
    all_ok = True
    for path in required:
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {path}")
        if not exists:
            all_ok = False
    return all_ok


def create_synthetic_data(n_samples: int = 100_000, n_clients: int = 100, seed: int = 42):
    """Generate and save synthetic TON_IoT-like data for quick experiments."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from amfta.data.partitioning import generate_synthetic_data, dirichlet_partition, save_partitions

    logger.info("Generating %d synthetic samples...", n_samples)
    X, y = generate_synthetic_data(n_samples, n_features=45, seed=seed)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=seed)

    # Server buffer
    rng = np.random.RandomState(seed)
    buf_idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)

    # Save processed
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/server", exist_ok=True)
    os.makedirs("data/partitions", exist_ok=True)

    np.savez_compressed("data/processed/train.npz", X=X_train, y=y_train)
    np.savez_compressed("data/processed/val.npz",   X=X_val,   y=y_val)
    np.savez_compressed("data/processed/test.npz",  X=X_test,  y=y_test)
    np.savez_compressed("data/server/val_buffer.npz",
                        X=X_val[buf_idx], y=y_val[buf_idx])

    logger.info("Saved: train=%d  val=%d  test=%d  buffer=%d",
                len(X_train), len(X_val), len(X_test), len(buf_idx))

    # Partition for FL clients
    logger.info("Partitioning into %d non-IID clients (Dirichlet α=0.5)...", n_clients)
    for s in [42, 123, 456, 789, 1024]:
        parts = dirichlet_partition(X_train, y_train, n_clients, alpha=0.5, seed=s)
        save_partitions(parts, Path("data/partitions"), seed=s)

    logger.info("✓ Synthetic dataset ready. Run experiments with --use_synthetic flag.")


def main():
    parser = argparse.ArgumentParser(description="AMFTA data setup utility")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data (no download required)")
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--verify", action="store_true",
                        help="Verify that processed data exists")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run full preprocessing on existing raw CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.verify:
        print("\nData verification:")
        ok = verify_data()
        sys.exit(0 if ok else 1)

    if args.synthetic:
        create_synthetic_data(args.n_samples, args.n_clients, args.seed)
        return

    if args.preprocess:
        if not Path("data/raw/NF-TON-IoT.csv").exists() and not Path("data/raw/NF-TON-IoT.parquet").exists() and not list(Path("data/raw").glob("*.parquet")):
            print(f"\n  ERROR: Dataset not found in data/raw/")
            print(f"\n  To download TON_IoT dataset:")
            print(f"    1. Visit: {TON_IOT_URL}")
            print(f"    2. Download 'Network Flow (NF-TON-IoT)' variant")
            print(f"    3. Place NF-TON-IoT.csv or .parquet in data/raw/")
            print(f"    4. Re-run: python scripts/setup_data.py --preprocess")
            print(f"\n  Or use synthetic data: python scripts/setup_data.py --synthetic")
            sys.exit(1)

        from amfta.data.preprocessing import preprocess
        from amfta.data.partitioning import dirichlet_partition, save_partitions

        data = preprocess()
        logger.info("Preprocessing complete.")

        for seed in [42, 123, 456, 789, 1024]:
            parts = dirichlet_partition(
                data["X_train"], data["y_train"],
                num_clients=100, alpha=0.5, seed=seed,
            )
            save_partitions(parts, Path("data/partitions"), seed=seed)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
