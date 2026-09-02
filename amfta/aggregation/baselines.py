"""
Baseline Aggregation Methods
==============================

Implements four baseline Byzantine-robust aggregation methods for comparison
against AMFTA:

  1. FedAvg        — Standard federated averaging (McMahan et al., 2017)
  2. Krum           — Nearest-neighbour selection (Blanchard et al., 2017)
  3. Trimmed Mean   — Coordinate-wise trimmed mean (Yin et al., 2018)
  4. FLTrust        — Server reference trust scoring (Cao et al., 2022)

All methods expose the same interface:
    aggregator.aggregate(global_model, updates, **kwargs) → update_dict
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

UpdateDict = Dict[int, Dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAggregator(ABC):
    """Common interface for all aggregation methods."""

    def reset(self) -> None:
        """Reset any stateful components between experiments."""

    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flatten_updates(updates: UpdateDict) -> Dict[int, torch.Tensor]:
    """Flatten each client update to a 1-D float tensor."""
    return {
        cid: torch.cat([v.float().flatten() for v in u.values()])
        for cid, u in updates.items()
    }


def _uniform_mean(updates: UpdateDict) -> Dict[str, torch.Tensor]:
    """Compute the uniform mean of all updates (FedAvg with equal weights)."""
    n = len(updates)
    result: Dict[str, torch.Tensor] = {}
    for k in next(iter(updates.values())).keys():
        result[k] = torch.stack([updates[cid][k].float() for cid in updates]).mean(0)
    return result


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging.

    Uses uniform weighting by default.  Optionally accepts per-client dataset
    sizes for data-proportional weighting as in the original FedAvg paper.

    Reference: McMahan et al., "Communication-Efficient Learning of Deep
    Networks from Decentralized Data", AISTATS 2017.
    """

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        dataset_sizes: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates.

        Parameters
        ----------
        dataset_sizes : dict {client_id: num_samples} | None
            If provided, weights proportionally by dataset size (FedAvg proper).
            If None, uses uniform weights.
        """
        if dataset_sizes is not None:
            total = sum(dataset_sizes[cid] for cid in updates)
            weights = {cid: dataset_sizes[cid] / total for cid in updates}
        else:
            n = len(updates)
            weights = {cid: 1.0 / n for cid in updates}

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            tensors = [weights[cid] * updates[cid][k].float() for cid in updates]
            result[k] = torch.stack(tensors).sum(dim=0)
        return result


# ---------------------------------------------------------------------------
# Krum / Multi-Krum
# ---------------------------------------------------------------------------

class KrumAggregator(BaseAggregator):
    """Krum (Multi-Krum) Byzantine-robust aggregation.

    Selects the single update whose sum of squared distances to its
    k = N − f − 2 nearest neighbours is minimised, where f is the assumed
    number of Byzantine clients.

    Limitation: Selects ONE update, losing information from all other benign
    clients.  Particularly hurt by non-IID data where legitimate clients with
    unusual distributions may be rejected.

    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent", NeurIPS 2017.

    Parameters
    ----------
    num_byzantine : int | None
        Assumed number of Byzantine clients.  If None, defaults to
        ``floor(0.3 * num_clients)`` (30% assumption).
    multi_krum : bool
        If True, returns the mean of the top-m Krum-selected updates
        rather than the single best.  Default False (standard Krum).
    m : int | None
        Number of updates to select for Multi-Krum.  Defaults to
        N − f − 2 when multi_krum=True.
    """

    def __init__(
        self,
        num_byzantine: Optional[int] = None,
        multi_krum: bool = False,
        m: Optional[int] = None,
    ) -> None:
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        self.m = m

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        client_ids = list(updates.keys())
        n = len(client_ids)

        f = self.num_byzantine if self.num_byzantine is not None else max(1, int(0.3 * n))
        k = max(1, n - f - 2)

        flat = _flatten_updates(updates)

        # Compute pairwise distances and Krum scores
        krum_scores: Dict[int, float] = {}
        for cid in client_ids:
            dists = sorted(
                torch.dist(flat[cid], flat[other]).item()
                for other in client_ids if other != cid
            )
            krum_scores[cid] = sum(dists[:k])

        if self.multi_krum:
            m = self.m or k
            selected = sorted(krum_scores, key=krum_scores.get)[:m]
            logger.debug("Multi-Krum selected %d clients: %s", len(selected), selected)
            return _uniform_mean({cid: updates[cid] for cid in selected})
        else:
            best = min(krum_scores, key=krum_scores.get)
            logger.debug("Krum selected client %d (score=%.4f)", best, krum_scores[best])
            return updates[best]

    def __repr__(self) -> str:
        return f"KrumAggregator(f={self.num_byzantine}, multi_krum={self.multi_krum})"


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMeanAggregator(BaseAggregator):
    """Coordinate-wise Trimmed Mean.

    For each parameter coordinate, removes the top and bottom
    ``trim_fraction`` of values and averages the remainder.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
    Optimal Statistical Rates", ICML 2018.

    Parameters
    ----------
    trim_fraction : float
        Fraction of values to trim from each end.  Default 0.1 (10% each side).
        Should be set ≥ Byzantine fraction for full protection, but the
        Byzantine fraction is not always known in practice.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(f"trim_fraction must be in [0, 0.5); got {trim_fraction}")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        trim_k = max(1, int(n * self.trim_fraction))

        if trim_k * 2 >= n:
            logger.warning(
                "trim_k=%d trims all updates (n=%d); falling back to FedAvg.", trim_k, n
            )
            return _uniform_mean(updates)

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            stacked = torch.stack([updates[cid][k].float() for cid in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k : n - trim_k]
            result[k] = trimmed.mean(dim=0)

        return result

    def __repr__(self) -> str:
        return f"TrimmedMeanAggregator(trim_fraction={self.trim_fraction})"


# ---------------------------------------------------------------------------
# FLTrust
# ---------------------------------------------------------------------------

class FLTrustAggregator(BaseAggregator):
    """FLTrust — Server Reference Trust Score Aggregation.

    Assigns each client a trust score = ReLU(cosine_similarity(g_i, g_root))
    where g_root is computed by the server from a small clean root dataset.
    Each client's update is normalised to the magnitude of the root update
    before weighted aggregation.

    IMPORTANT: FLTrust REQUIRES a clean labelled root dataset on the server.
    This is the key assumption that AMFTA eliminates.  In simulation we use
    50-200 samples from the held-out validation set; in a real deployment this
    must come from a verified clean source.

    Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning via
    Trust Bootstrapping", NDSS 2022.

    Parameters
    ----------
    root_dataset_size : int
        Number of samples in the server root dataset (default 200 per paper).
    """

    def __init__(self, root_dataset_size: int = 200) -> None:
        self.root_dataset_size = root_dataset_size
        self._root_update: Optional[Dict[str, torch.Tensor]] = None

    def set_root_update(self, root_update: Dict[str, torch.Tensor]) -> None:
        """Provide the root update computed from the server's clean dataset."""
        self._root_update = root_update

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        root_update: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        root_update : dict | None
            The server-computed reference update.  If not provided here, uses
            the last value set via set_root_update().  Falls back to FedAvg
            if neither is available.
        """
        ref = root_update or self._root_update
        if ref is None:
            logger.warning("FLTrust: no root update available; using FedAvg fallback.")
            return _uniform_mean(updates)

        ref_flat = torch.cat([v.float().flatten() for v in ref.values()])
        ref_norm = ref_flat.norm() + 1e-8

        trust: Dict[int, float] = {}
        client_norms: Dict[int, float] = {}
        for cid, update in updates.items():
            flat = torch.cat([v.float().flatten() for v in update.values()])
            client_norms[cid] = flat.norm().item() + 1e-8
            cos = F.cosine_similarity(
                flat.unsqueeze(0), ref_flat.unsqueeze(0)
            ).item()
            trust[cid] = max(0.0, cos)  # ReLU

        total = sum(trust.values())
        if total <= 0.0:
            logger.warning("FLTrust: all trust scores are zero; using FedAvg fallback.")
            return _uniform_mean(updates)

        # Deviation from canonical FLTrust: the published method rescales every
        # client update to the *root* update norm.  Our server root set (200
        # samples) yields a root norm far smaller than the clients' natural
        # update norms (trained on ~18k samples each), so with global_lr=1.0 the
        # global model stays frozen near initialisation and degenerates to the
        # majority-class predictor.  We instead rescale to the *median* client
        # norm — a robust, attack-resistant reference that preserves FLTrust's
        # cosine-similarity trust weighting while restoring a healthy server
        # step.  This makes FLTrust a fair, converging baseline.
        sorted_norms = sorted(client_norms.values())
        target_norm = sorted_norms[len(sorted_norms) // 2]  # median client norm

        result: Dict[str, torch.Tensor] = {}
        for k in ref.keys():
            weighted = torch.zeros_like(ref[k].float())
            for cid in updates:
                scale = target_norm / client_norms[cid]
                weighted += trust[cid] * updates[cid][k].float() * scale
            result[k] = weighted / total

        return result

    def __repr__(self) -> str:
        return f"FLTrustAggregator(root_dataset_size={self.root_dataset_size})"


# ---------------------------------------------------------------------------
# FedDBC — Density-Based Consensus Aggregation (competing baseline)
# ---------------------------------------------------------------------------

class FedDBCAggregator(BaseAggregator):
    """FedDBC — Density-Based Consensus aggregation (re-implementation).

    Reproduction of the FedDBC defense for federated NIDS (2026): cluster
    client updates by pairwise cosine distance using adaptively-calibrated
    DBSCAN, keep the dominant (largest) consensus cluster, discard smaller
    clusters and noise points, then apply Trimmed Mean within the surviving
    cluster to suppress residual intra-cluster outliers.

    Requires no trusted server data, no persistent state, and no prior
    attacker count.  Guarantees are conditional on an honest majority and
    geometric separability of benign vs. malicious updates.

    Parameters
    ----------
    trim_fraction : float
        Trimmed-mean fraction applied within the selected cluster. Default 0.1.
    min_cluster_frac : float
        DBSCAN min_samples as a fraction of N (core-point density). Default 0.25.
    eps_percentile : float
        Percentile of k-NN cosine distances used to auto-calibrate eps.
        Default 50 (median k-distance).
    """

    def __init__(
        self,
        trim_fraction: float = 0.1,
        min_cluster_frac: float = 0.25,
        eps_percentile: float = 50.0,
    ) -> None:
        self.trim_fraction = trim_fraction
        self.min_cluster_frac = min_cluster_frac
        self.eps_percentile = eps_percentile

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        import numpy as np
        from sklearn.cluster import DBSCAN

        client_ids = list(updates.keys())
        n = len(client_ids)
        flat = _flatten_updates(updates)
        M = torch.stack([flat[cid] for cid in client_ids])  # [n, d]

        # Pairwise cosine distance matrix (1 - cosine similarity), clamped >= 0
        Mn = F.normalize(M, dim=1)
        cos_sim = (Mn @ Mn.t()).clamp(-1.0, 1.0)
        dist = (1.0 - cos_sim).clamp(min=0.0).cpu().numpy()
        np.fill_diagonal(dist, 0.0)

        min_samples = max(2, int(self.min_cluster_frac * n))

        # Adaptive eps: percentile of each point's k-th nearest-neighbour distance
        k = min(min_samples, n - 1)
        knn = np.sort(dist, axis=1)[:, k]  # k-th NN distance per point
        eps = float(np.percentile(knn, self.eps_percentile))
        if eps <= 0.0:
            eps = float(np.percentile(dist[dist > 0], 50)) if np.any(dist > 0) else 1e-6

        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(dist)

        # Select the dominant (largest) non-noise cluster
        valid = labels[labels >= 0]
        if valid.size == 0:
            logger.warning("FedDBC: DBSCAN found no dense cluster; falling back to Trimmed Mean.")
            selected = client_ids
        else:
            vals, counts = np.unique(valid, return_counts=True)
            dominant = vals[int(np.argmax(counts))]
            selected = [client_ids[i] for i in range(n) if labels[i] == dominant]

        # Trimmed Mean within the surviving cluster
        sel_updates = {cid: updates[cid] for cid in selected}
        m = len(sel_updates)
        trim_k = int(m * self.trim_fraction)
        if m == 0:
            return _uniform_mean(updates)
        if trim_k * 2 >= m:
            return _uniform_mean(sel_updates)

        result: Dict[str, torch.Tensor] = {}
        for key in next(iter(sel_updates.values())).keys():
            stacked = torch.stack([sel_updates[cid][key].float() for cid in sel_updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k : m - trim_k]
            result[key] = trimmed.mean(dim=0)
        return result

    def __repr__(self) -> str:
        return f"FedDBCAggregator(trim_fraction={self.trim_fraction})"


# ---------------------------------------------------------------------------
# Aggregator factory
# ---------------------------------------------------------------------------

AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "krum": KrumAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
    "fltrust": FLTrustAggregator,
    "feddbc": FedDBCAggregator,
}


def build_aggregator(method: str, config: Optional[dict] = None) -> BaseAggregator:
    """Build a baseline aggregator by name.

    Parameters
    ----------
    method : str
        One of 'fedavg', 'krum', 'trimmed_mean', 'fltrust'.
    config : dict | None
        Constructor keyword arguments.

    Returns
    -------
    BaseAggregator instance.
    """
    cfg = config or {}
    method_lower = method.lower().replace("-", "_")

    if method_lower not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )
    return AGGREGATOR_REGISTRY[method_lower](**cfg)
