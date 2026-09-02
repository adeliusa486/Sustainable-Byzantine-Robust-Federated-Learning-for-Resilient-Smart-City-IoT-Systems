"""
Dirichlet Non-IID Data Partitioning
======================================

Splits the training dataset into N heterogeneous client partitions using
Dirichlet sampling, simulating real-world non-IID conditions in smart city
IoT deployments.

Dirichlet(α) controls heterogeneity:
  α → 0 : extreme heterogeneity (each client sees only one class)
  α = 0.5: moderate heterogeneity (paper setting)
  α → ∞ : IID distribution (equal class proportions everywhere)

The paper uses α=0.5 with N=100 clients.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ClientDataset = Dict[int, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Core Dirichlet partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 10,
    min_samples_per_class: int = 2,
) -> ClientDataset:
    """Partition dataset into num_clients non-IID splits via Dirichlet sampling.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)   Feature matrix.
    y : np.ndarray, shape (N,)     Binary labels {0, 1}.
    num_clients : int              Number of FL clients.
    alpha : float                  Dirichlet concentration parameter.
    seed : int                     RNG seed for reproducibility.
    min_samples_per_client : int   Minimum total samples per client.
    min_samples_per_class  : int   Minimum samples of each class per client
                                   (prevents zero-class clients).

    Returns
    -------
    dict {client_id (int): (X_client, y_client)}
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError(f"Expected at least 2 classes; got {n_classes}")

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)

        # Sample Dirichlet proportions
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Convert fractions to sample counts
        counts = (proportions * n_cls).astype(int)

        # Fix rounding: ensure sum == n_cls
        diff = n_cls - counts.sum()
        if diff != 0:
            # Distribute rounding error to the clients with largest fractional parts
            fracs = proportions * n_cls - counts
            adjust_idx = np.argsort(fracs)[::-1][: abs(diff)]
            counts[adjust_idx] += int(np.sign(diff))

        # Assign indices
        start = 0
        for client_id, count in enumerate(counts):
            # Enforce minimum per class
            actual_count = max(count, min_samples_per_class)
            actual_count = min(actual_count, n_cls - start)
            client_indices[client_id].extend(cls_idx[start : start + actual_count].tolist())
            start += actual_count
            if start >= n_cls:
                break

    # Build client datasets
    client_data: ClientDataset = {}
    for cid in range(num_clients):
        idxs = np.array(client_indices[cid])
        if len(idxs) < min_samples_per_client:
            # Pad by sampling with replacement from all data
            extra = rng.choice(len(X), size=min_samples_per_client - len(idxs), replace=True)
            idxs = np.concatenate([idxs, extra])

        rng.shuffle(idxs)
        client_data[cid] = (X[idxs], y[idxs])

    _log_partition_stats(client_data)
    return client_data


def _log_partition_stats(client_data: ClientDataset) -> None:
    """Log summary statistics of the partition."""
    sizes = [len(v[0]) for v in client_data.values()]
    attack_fracs = []
    for X_c, y_c in client_data.values():
        if len(y_c) > 0:
            attack_fracs.append(y_c.mean())

    logger.info(
        "Partition stats — clients: %d | sizes: [%d, %d] mean=%.0f | "
        "attack frac: [%.2f, %.2f] mean=%.2f",
        len(client_data),
        min(sizes), max(sizes), np.mean(sizes),
        min(attack_fracs), max(attack_fracs), np.mean(attack_fracs),
    )


# ---------------------------------------------------------------------------
# Byzantine client assignment
# ---------------------------------------------------------------------------

def assign_byzantine_clients(
    num_clients: int,
    byzantine_fraction: float,
    seed: int = 42,
) -> set:
    """Randomly select Byzantine client IDs.

    Parameters
    ----------
    num_clients       : int   Total number of FL clients.
    byzantine_fraction: float Fraction of clients that are Byzantine (e.g. 0.30).
    seed              : int   RNG seed.

    Returns
    -------
    set of int  Client IDs designated as Byzantine.
    """
    rng = np.random.RandomState(seed)
    num_byzantine = int(num_clients * byzantine_fraction)
    byzantine_ids = set(rng.choice(num_clients, num_byzantine, replace=False).tolist())
    logger.info(
        "Byzantine assignment: %d/%d clients (%.0f%%) — IDs: %s",
        num_byzantine, num_clients, 100 * byzantine_fraction,
        sorted(list(byzantine_ids))[:10],
    )
    return byzantine_ids


# ---------------------------------------------------------------------------
# Save / Load partitions
# ---------------------------------------------------------------------------

def save_partitions(
    client_data: ClientDataset,
    partition_dir: Path,
    seed: int,
) -> None:
    """Save each client's dataset to a .npz file."""
    seed_dir = Path(partition_dir) / f"seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)

    for cid, (X_c, y_c) in client_data.items():
        path = seed_dir / f"client_{cid:03d}.npz"
        np.savez_compressed(path, X=X_c, y=y_c)

    logger.info("Saved %d client partitions to %s", len(client_data), seed_dir)


def load_partitions(
    partition_dir: Path,
    seed: int,
    num_clients: int = 100,
) -> ClientDataset:
    """Load per-client .npz files from disk."""
    seed_dir = Path(partition_dir) / f"seed_{seed}"

    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Partition directory not found: {seed_dir}. "
            "Run: python -m amfta.data.partitioning"
        )

    client_data: ClientDataset = {}
    for cid in range(num_clients):
        path = seed_dir / f"client_{cid:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing partition file: {path}")
        npz = np.load(path)
        client_data[cid] = (npz["X"], npz["y"])

    logger.info("Loaded %d client partitions from %s", num_clients, seed_dir)
    return client_data


# ---------------------------------------------------------------------------
# Generate synthetic data (for testing without TON_IoT)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 10_000,
    n_features: int = 45,
    attack_fraction: float = 0.62,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data mimicking TON_IoT.

    Used for smoke tests and CI pipelines that cannot access the real dataset.
    Feature distributions approximate the normalised [0,1] range.

    Parameters
    ----------
    n_samples       : int   Total samples to generate.
    n_features      : int   Number of features (45 for TON_IoT).
    attack_fraction : float Class 1 (attack) proportion. Default 0.62.
    seed            : int   Random seed.

    Returns
    -------
    (X, y) — float32 feature matrix and binary labels.
    """
    rng = np.random.RandomState(seed)
    n_attack = int(n_samples * attack_fraction)
    n_normal = n_samples - n_attack

    # Normal traffic: lower feature values (more benign patterns)
    X_normal = rng.beta(a=2, b=5, size=(n_normal, n_features)).astype(np.float32)
    y_normal = np.zeros(n_normal, dtype=np.int64)

    # Attack traffic: higher/more varied feature values
    X_attack = rng.beta(a=5, b=2, size=(n_attack, n_features)).astype(np.float32)
    y_attack = np.ones(n_attack, dtype=np.int64)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from amfta.data.preprocessing import load_processed

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="Partition TON_IoT for federated learning.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--partition_dir", default="data/partitions")
    args = parser.parse_args()

    data = load_processed(args.processed_dir)
    X_train, y_train = data["X_train"], data["y_train"]

    for seed in args.seeds:
        logger.info("Partitioning with seed=%d ...", seed)
        client_data = dirichlet_partition(
            X_train, y_train,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=seed,
        )
        save_partitions(client_data, Path(args.partition_dir), seed)
