"""
Aggregation rules reported in the manuscript but absent from this package.
==========================================================================

The manuscript reports four server-side rules that had no implementation here:

  * ``normclip``  — median-norm rescaling followed by a uniform mean, with no
                    trust weighting. This is the ablation arm that isolates how
                    much of the method's robustness comes from clipping rather
                    than from the trust engine.
  * ``median``    — coordinate-wise median (Yin et al., ICML 2018), the
                    companion order statistic to Trimmed Mean.
  * ``multikrum`` — the averaging variant of Krum. ``KrumAggregator`` already
                    supports ``multi_krum=True``; this only exposes it under
                    its own name so it can be selected as a method.
  * ``foolsgold`` — inter-client cosine dissimilarity weighting (Fung et al.,
                    RAID 2020), the closest prior mechanism to Factor I.

Importing this module registers all four in ``AGGREGATOR_REGISTRY``, so
``build_aggregator('median')`` works without editing ``baselines.py``.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

from amfta.aggregation.baselines import (
    AGGREGATOR_REGISTRY,
    BaseAggregator,
    KrumAggregator,
    UpdateDict,
    _flatten_updates,
    _uniform_mean,
)

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Shared helper: median-norm rescaling
# ---------------------------------------------------------------------------

def rescale_to_median_norm(updates: UpdateDict) -> UpdateDict:
    """Rescale every update to the cohort's median L2 norm.

    This is Equation (12) of the manuscript, factored out so that the
    ablation arm and the trust aggregator apply exactly the same operation.
    """
    flat = _flatten_updates(updates)
    norms = {cid: float(v.norm()) for cid, v in flat.items()}
    ordered = sorted(norms.values())
    target = ordered[len(ordered) // 2]

    rescaled: UpdateDict = {}
    for cid, u in updates.items():
        factor = target / (norms[cid] + EPS)
        rescaled[cid] = {k: v.float() * factor for k, v in u.items()}
    return rescaled


# ---------------------------------------------------------------------------
# NormClip-Only
# ---------------------------------------------------------------------------

class NormClipOnlyAggregator(BaseAggregator):
    """Median-norm rescaling plus a uniform mean. No trust weighting.

    Isolates the contribution of magnitude control from the contribution of
    the trust factors. Against magnitude-based poisoning this should recover
    most of the trust aggregator's performance; against normal-magnitude
    poisoning such as label flipping it should not.

    Reference for the defense in isolation: Sun et al., "Can You Really
    Backdoor Federated Learning?", 2019.
    """

    def aggregate(self, global_model: nn.Module, updates: UpdateDict,
                  **kwargs) -> Dict[str, torch.Tensor]:
        return _uniform_mean(rescale_to_median_norm(updates))

    def __repr__(self) -> str:
        return "NormClipOnlyAggregator()"


# ---------------------------------------------------------------------------
# Coordinate-wise median
# ---------------------------------------------------------------------------

class CoordinateMedianAggregator(BaseAggregator):
    """Coordinate-wise median.

    Unlike Trimmed Mean, needs no trim fraction and therefore no estimate of
    the attacker fraction, which is why the manuscript reports it separately:
    Trimmed Mean's collapse under one-sided magnitude contamination is partly
    an artefact of trimming symmetric tails.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
    Optimal Statistical Rates", ICML 2018.
    """

    def aggregate(self, global_model: nn.Module, updates: UpdateDict,
                  **kwargs) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            stacked = torch.stack([updates[cid][k].float() for cid in updates])
            result[k] = stacked.median(dim=0).values
        return result

    def __repr__(self) -> str:
        return "CoordinateMedianAggregator()"


# ---------------------------------------------------------------------------
# Multi-Krum
# ---------------------------------------------------------------------------

class MultiKrumAggregator(KrumAggregator):
    """Krum with averaging over the m best-scoring updates.

    Present only so that ``multikrum`` is selectable as a method name; the
    behaviour is ``KrumAggregator(multi_krum=True)``. Like Krum it requires
    an estimate of the Byzantine count, which the manuscript marks in the
    oracle column.
    """

    def __init__(self, num_byzantine=None, m=None, **kwargs) -> None:
        super().__init__(num_byzantine=num_byzantine, multi_krum=True, m=m, **kwargs)

    def __repr__(self) -> str:
        return f"MultiKrumAggregator(num_byzantine={self.num_byzantine})"


# ---------------------------------------------------------------------------
# FoolsGold
# ---------------------------------------------------------------------------

class FoolsGoldAggregator(BaseAggregator):
    """FoolsGold: penalise clients whose updates are mutually too similar.

    Designed against sybil coalitions that submit correlated updates. It is
    the closest prior mechanism to the manuscript's Factor I in that it needs
    no server data, but it scores clients against *each other* rather than
    against a reference direction, and its target is collusion rather than
    independent adversaries. That difference is why the manuscript expects it
    to do reasonably against label flipping and poorly against independent
    Gaussian noise.

    Implementation follows Fung et al., "The Limitations of Federated Learning
    in Sybil Settings", RAID 2020: maximum pairwise cosine similarity per
    client, pardoning, logit rescaling.

    Parameters
    ----------
    use_history : bool
        Accumulate updates across rounds before scoring, as in the original.
    kappa : float
        Confidence parameter of the logit rescaling step.
    """

    def __init__(self, use_history: bool = True, kappa: float = 1.0) -> None:
        self.use_history = use_history
        self.kappa = kappa
        self._history: Dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self._history = {}

    def aggregate(self, global_model: nn.Module, updates: UpdateDict,
                  **kwargs) -> Dict[str, torch.Tensor]:
        flat = _flatten_updates(updates)
        cids = list(flat.keys())

        if self.use_history:
            for cid in cids:
                prev = self._history.get(cid)
                self._history[cid] = flat[cid].clone() if prev is None else prev + flat[cid]
            feats = {cid: self._history[cid] for cid in cids}
        else:
            feats = flat

        n = len(cids)
        if n < 2:
            return _uniform_mean(updates)

        M = torch.stack([feats[c] for c in cids])
        M = M / (M.norm(dim=1, keepdim=True) + EPS)
        cs = M @ M.T
        cs.fill_diagonal_(-1.0)

        # v_i = max_j cos(i, j)
        v = cs.max(dim=1).values.clamp(min=0.0)

        # Pardoning: a client whose own maximum similarity is lower than that
        # of the client it resembles is rescaled down.
        cs_p = cs.clone()
        for i in range(n):
            for j in range(n):
                if i != j and v[j] > v[i] and v[j] > EPS:
                    cs_p[i, j] = cs_p[i, j] * v[i] / v[j]

        wv = 1.0 - cs_p.max(dim=1).values.clamp(min=0.0)
        wv = wv.clamp(0.0, 1.0)
        if wv.max() > EPS:
            wv = wv / wv.max()
        wv = wv.clamp(max=1.0 - 1e-5)

        # Logit rescaling
        wv = torch.where(wv > EPS, wv, torch.full_like(wv, EPS))
        wv = self.kappa * (torch.log(wv / (1.0 - wv) + EPS) + 0.5)
        wv = torch.nan_to_num(wv, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if float(wv.sum()) <= EPS:
            logger.warning("FoolsGold zeroed every weight; falling back to uniform mean.")
            return _uniform_mean(updates)

        wv = wv / wv.sum()
        weights = {cid: float(wv[i]) for i, cid in enumerate(cids)}

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            result[k] = sum(weights[cid] * updates[cid][k].float() for cid in cids)
        return result

    def __repr__(self) -> str:
        return f"FoolsGoldAggregator(use_history={self.use_history})"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AGGREGATOR_REGISTRY.update({
    "normclip": NormClipOnlyAggregator,
    "normclip_only": NormClipOnlyAggregator,
    "median": CoordinateMedianAggregator,
    "multikrum": MultiKrumAggregator,
    "foolsgold": FoolsGoldAggregator,
})
