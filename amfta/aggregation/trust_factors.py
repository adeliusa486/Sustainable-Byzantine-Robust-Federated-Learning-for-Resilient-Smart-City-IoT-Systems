"""
AMFTA Trust Factor Computation
================================

Implements the three complementary trust signals used in AMFTA:

  Factor I  — Gradient Similarity  (S_i^{(t)})
    Cosine similarity of each client's update to the population centroid.
    Normalised from [-1, 1] to [0, 1].

  Factor II — Historical Reputation (H_i^{(t)})
    Exponential Moving Average of similarity scores over training rounds.
    H_i^{(t)} = β · H_i^{(t-1)} + (1-β) · S_i^{(t)},  β=0.9

  Factor III — Contribution Quality (Q_i^{(t)})
    Leave-one-out accuracy delta on a 500-sample server validation buffer.
    Applied only to *borderline* clients to limit compute overhead.

Reference: Algorithm 1 in the AMFTA paper.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Type aliases
UpdateDict = Dict[int, Dict[str, torch.Tensor]]
ScoreDict = Dict[int, float]


# ---------------------------------------------------------------------------
# Factor I — Gradient Similarity
# ---------------------------------------------------------------------------

def compute_gradient_similarity(
    updates: UpdateDict,
    eps: float = 1e-8,
) -> ScoreDict:
    """Compute cosine similarity of each client update to the population centroid.

    Parameters
    ----------
    updates : dict  {client_id: {param_name: tensor}}
        Model update deltas from all participating clients.
    eps : float
        Numerical stability constant to avoid division-by-zero.

    Returns
    -------
    dict {client_id: float}
        Similarity scores in [0, 1].  Higher → more aligned with population.

    Notes
    -----
    The centroid is computed as the simple mean of *all* client updates,
    including Byzantine clients.  This means the centroid is a noisy reference
    under high Byzantine rates (>40%).  The historical reputation factor (II)
    compensates for this limitation by tracking consistency over time.
    """
    client_ids = list(updates.keys())
    if not client_ids:
        raise ValueError("updates dict is empty")

    # Flatten each update to a 1-D vector
    flat_updates: Dict[int, torch.Tensor] = {}
    for cid in client_ids:
        vecs = [v.float().flatten() for v in updates[cid].values()]
        flat_updates[cid] = torch.cat(vecs)

    # Population centroid (mean of all updates)
    centroid = torch.stack(list(flat_updates.values())).mean(dim=0)

    similarity: ScoreDict = {}
    for cid, vec in flat_updates.items():
        cos_raw = F.cosine_similarity(
            vec.unsqueeze(0), centroid.unsqueeze(0), eps=eps
        ).item()
        # Map [-1, 1] → [0, 1]
        similarity[cid] = (cos_raw + 1.0) / 2.0

    return similarity


# ---------------------------------------------------------------------------
# Factor II — Historical Reputation (EMA)
# ---------------------------------------------------------------------------

class ReputationTracker:
    """Per-client EMA reputation tracker.

    Maintains H_i^{(t)} = β · H_i^{(t-1)} + (1-β) · S_i^{(t)} for all clients.

    Parameters
    ----------
    beta : float
        EMA decay factor.  β=0.9 → ~10-round effective memory window.
    init_value : float
        Neutral starting reputation for new clients.  Default 0.5 (mid-range).
    """

    def __init__(self, beta: float = 0.9, init_value: float = 0.5) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        self.beta = beta
        self.init_value = init_value
        self._history: Dict[int, float] = {}

    # ------------------------------------------------------------------

    def update(self, similarity: ScoreDict) -> ScoreDict:
        """Update EMA for all clients and return current reputation scores.

        Parameters
        ----------
        similarity : dict {client_id: float}
            Current-round gradient similarity scores.

        Returns
        -------
        dict {client_id: float}
            Updated reputation scores H_i^{(t)}.
        """
        for cid, s in similarity.items():
            if cid not in self._history:
                self._history[cid] = self.init_value
            self._history[cid] = (
                self.beta * self._history[cid] + (1 - self.beta) * s
            )
        return dict(self._history)

    def get(self, client_id: int) -> float:
        """Return current reputation for a single client."""
        return self._history.get(client_id, self.init_value)

    def reset(self) -> None:
        """Reset all reputation history (e.g. between experiments)."""
        self._history.clear()

    def state_dict(self) -> dict:
        return {"beta": self.beta, "init_value": self.init_value, "history": dict(self._history)}

    def load_state_dict(self, sd: dict) -> None:
        self.beta = sd["beta"]
        self.init_value = sd["init_value"]
        self._history = dict(sd["history"])

    @property
    def effective_memory_rounds(self) -> float:
        """Approximate number of rounds retained in effective memory."""
        return 1.0 / (1.0 - self.beta)

    def __repr__(self) -> str:
        return (
            f"ReputationTracker(beta={self.beta}, "
            f"clients={len(self._history)}, "
            f"effective_memory={self.effective_memory_rounds:.1f} rounds)"
        )


# ---------------------------------------------------------------------------
# Preliminary Score (Factors I + II only, before quality evaluation)
# ---------------------------------------------------------------------------

def compute_preliminary_scores(
    similarity: ScoreDict,
    history: ScoreDict,
    alpha_s: float = 0.4,
    alpha_h: float = 0.3,
) -> ScoreDict:
    """Compute the two-factor preliminary score used for borderline detection.

    T̂_i = (α_s · S_i + α_h · H_i) / (α_s + α_h)

    Normalised by the sum of weights so the result remains in [0, 1].

    Parameters
    ----------
    similarity : dict {client_id: float}   Factor I scores.
    history    : dict {client_id: float}   Factor II scores.
    alpha_s    : float                     Weight for gradient similarity.
    alpha_h    : float                     Weight for historical reputation.

    Returns
    -------
    dict {client_id: float}  Preliminary trust scores in [0, 1].
    """
    denom = alpha_s + alpha_h
    return {
        cid: (alpha_s * similarity[cid] + alpha_h * history.get(cid, 0.5)) / denom
        for cid in similarity
    }


# ---------------------------------------------------------------------------
# Borderline Detection
# ---------------------------------------------------------------------------

def identify_borderline_clients(
    preliminary: ScoreDict,
    tau_lower: float = 0.35,
    tau_upper: float = 0.55,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Classify clients into clearly benign, borderline, and clearly suspect.

    Parameters
    ----------
    preliminary : dict {client_id: float}  Two-factor preliminary scores.
    tau_lower   : float  Lower borderline threshold.
    tau_upper   : float  Upper borderline threshold.

    Returns
    -------
    (benign_ids, borderline_ids, suspect_ids) — three disjoint sets.
    """
    benign: Set[int] = set()
    borderline: Set[int] = set()
    suspect: Set[int] = set()

    for cid, score in preliminary.items():
        if score > tau_upper:
            benign.add(cid)
        elif score < tau_lower:
            suspect.add(cid)
        else:
            borderline.add(cid)

    logger.debug(
        "Borderline detection — benign: %d, borderline: %d, suspect: %d",
        len(benign), len(borderline), len(suspect),
    )
    return benign, borderline, suspect


# ---------------------------------------------------------------------------
# Factor III — Contribution Quality (Leave-One-Out)
# ---------------------------------------------------------------------------

def compute_contribution_quality(
    global_model: nn.Module,
    updates: UpdateDict,
    val_buffer: Tuple[torch.Tensor, torch.Tensor],
    borderline_ids: Set[int],
    quality_default_benign: float = 1.0,
    quality_default_suspect: float = 0.0,
    suspect_ids: Optional[Set[int]] = None,
) -> ScoreDict:
    """Estimate each borderline client's marginal contribution to model accuracy.

    Q_i^{(t)} = A(w + ḡ^{(t)}) − A(w + ḡ_{−i}^{(t)})

    ḡ^{(t)} is computed as the **uniform mean** of all updates (not the final
    trust-weighted aggregation) to avoid a circular dependency.  This resolves
    the ambiguity noted in the reproducibility analysis of the original paper.

    Clearly benign clients receive Q = quality_default_benign (1.0).
    Clearly suspect clients receive Q = quality_default_suspect (0.0).
    Only borderline clients undergo the expensive leave-one-out evaluation.

    Parameters
    ----------
    global_model  : nn.Module      Current global model (read-only; not mutated).
    updates       : UpdateDict     All client model updates.
    val_buffer    : (X, y)         500-sample validation buffer tensors.
    borderline_ids: set[int]       Clients requiring quality evaluation.
    quality_default_benign  : float  Q assigned to clearly benign clients.
    quality_default_suspect : float  Q assigned to clearly suspect clients.
    suspect_ids   : set[int]|None    Suspect client IDs (optional; Q set to 0).

    Returns
    -------
    dict {client_id: float}  Quality scores in [0, 1].
    """
    suspect_ids = suspect_ids or set()
    all_ids = list(updates.keys())
    X_val, y_val = val_buffer
    quality: ScoreDict = {}

    # --- Helper: evaluate accuracy after applying a uniform-mean update ---
    def _evaluate_with_update(selected_ids: list[int]) -> float:
        """Temporarily apply mean update from selected clients, evaluate, restore."""
        if not selected_ids:
            return 0.0

        # Compute uniform mean update
        mean_update: Dict[str, torch.Tensor] = {}
        for k in updates[selected_ids[0]].keys():
            stacked = torch.stack([updates[cid][k].float() for cid in selected_ids])
            mean_update[k] = stacked.mean(dim=0)

        # Save original state, apply update, evaluate, restore
        original_state = copy.deepcopy(global_model.state_dict())
        new_state = {
            k: original_state[k].float() + mean_update[k] for k in original_state
        }
        global_model.load_state_dict(new_state)

        global_model.eval()
        with torch.no_grad():
            preds = (global_model(X_val.float()) >= 0.5).float()
            acc = (preds == y_val.float()).float().mean().item()

        global_model.load_state_dict(original_state)
        return acc

    # --- Baseline: accuracy with ALL clients' updates ---
    acc_full = _evaluate_with_update(all_ids)

    # --- Assign quality scores ---
    for cid in all_ids:
        if cid in borderline_ids:
            remaining = [x for x in all_ids if x != cid]
            acc_without = _evaluate_with_update(remaining)
            delta = acc_full - acc_without
            # Clip to [0, 1]: positive = helpful, negative → 0
            quality[cid] = float(max(0.0, min(1.0, delta)))
        elif cid in suspect_ids:
            quality[cid] = quality_default_suspect
        else:
            quality[cid] = quality_default_benign

    logger.debug(
        "Quality eval — evaluated %d borderline clients out of %d total",
        len(borderline_ids), len(all_ids),
    )
    return quality


# ---------------------------------------------------------------------------
# Final Trust Score Combination
# ---------------------------------------------------------------------------

def combine_trust_scores(
    similarity: ScoreDict,
    history: ScoreDict,
    quality: ScoreDict,
    alpha_s: float = 0.4,
    alpha_h: float = 0.3,
    alpha_q: float = 0.3,
) -> ScoreDict:
    """Combine the three trust factors into the final trust score.

    T_i^{(t)} = α_s · S_i + α_h · H_i + α_q · Q_i

    Parameters
    ----------
    similarity : dict {cid: float}  Factor I (gradient similarity).
    history    : dict {cid: float}  Factor II (historical reputation).
    quality    : dict {cid: float}  Factor III (contribution quality).
    alpha_s    : float              Weight for similarity.  Default 0.4.
    alpha_h    : float              Weight for history.     Default 0.3.
    alpha_q    : float              Weight for quality.     Default 0.3.

    Returns
    -------
    dict {client_id: float}  Final trust scores.  Typically in (0, 1].
    """
    if abs(alpha_s + alpha_h + alpha_q - 1.0) > 1e-6:
        raise ValueError(
            f"Trust weights must sum to 1.0; got {alpha_s}+{alpha_h}+{alpha_q}"
            f"={alpha_s+alpha_h+alpha_q:.4f}"
        )

    trust: ScoreDict = {}
    for cid in similarity:
        h = history.get(cid, 0.5)
        q = quality.get(cid, 0.5)
        trust[cid] = alpha_s * similarity[cid] + alpha_h * h + alpha_q * q

    return trust
