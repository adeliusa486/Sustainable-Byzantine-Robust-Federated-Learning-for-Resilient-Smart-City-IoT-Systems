"""
AMFTAAggregator — Drop-in Byzantine-Robust Aggregation Module
=============================================================

Implements the complete AMFTA pipeline as described in Algorithm 1:

  1. Receive model updates {g_i^{(t)}} from all N clients
  2. Factor I  : Gradient similarity S_i (cosine to centroid)
  3. Factor II : Historical reputation H_i (EMA across rounds)
  4. Borderline detection (τ_l=0.35, τ_u=0.55)
  5. Factor III: Contribution quality Q_i (leave-one-out, selective)
  6. Final trust T_i = α_s·S + α_h·H + α_q·Q
  7. Trust-weighted soft aggregation: ḡ = Σ T_i·g_i / Σ T_i

Usage
-----
    aggregator = AMFTAAggregator(num_clients=100)
    aggregated_update = aggregator.aggregate(global_model, updates, val_buffer)
    global_model.apply_update(aggregated_update)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from amfta.aggregation.trust_factors import (
    ReputationTracker,
    combine_trust_scores,
    compute_contribution_quality,
    compute_gradient_similarity,
    compute_preliminary_scores,
    identify_borderline_clients,
)

logger = logging.getLogger(__name__)

UpdateDict = Dict[int, Dict[str, torch.Tensor]]
ScoreDict = Dict[int, float]


# ---------------------------------------------------------------------------
# Round-level diagnostics dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoundDiagnostics:
    """Stores per-round trust engine diagnostics for logging and analysis."""
    round_number: int = 0
    similarity_scores: Dict[int, float] = field(default_factory=dict)
    reputation_scores: Dict[int, float] = field(default_factory=dict)
    quality_scores: Dict[int, float] = field(default_factory=dict)
    trust_scores: Dict[int, float] = field(default_factory=dict)
    num_benign: int = 0
    num_borderline: int = 0
    num_suspect: int = 0
    mean_trust: float = 0.0
    min_trust: float = 0.0
    max_trust: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "num_benign": self.num_benign,
            "num_borderline": self.num_borderline,
            "num_suspect": self.num_suspect,
            "mean_trust": self.mean_trust,
            "min_trust": self.min_trust,
            "max_trust": self.max_trust,
            "trust_scores": self.trust_scores,
        }


# ---------------------------------------------------------------------------
# AMFTA Aggregator
# ---------------------------------------------------------------------------

class AMFTAAggregator:
    """Adaptive Multi-Factor Trust Aggregation engine.

    Parameters
    ----------
    num_clients : int
        Total number of FL clients (N=100 in paper).
    alpha_s : float
        Weight for gradient similarity factor.  Default 0.4.
    alpha_h : float
        Weight for historical reputation factor. Default 0.3.
    alpha_q : float
        Weight for contribution quality factor.  Default 0.3.
    beta : float
        EMA decay for reputation tracker.  Default 0.9.
    tau_lower : float
        Lower borderline threshold.  Default 0.35.
    tau_upper : float
        Upper borderline threshold.  Default 0.55.
    eps : float
        Cosine similarity stability constant.  Default 1e-8.
    use_quality_eval : bool
        Enable Factor III (contribution quality).  Default True.
        Set False to run AMFTA-SH variant (ablation).
    """

    def __init__(
        self,
        num_clients: int = 100,
        alpha_s: float = 0.4,
        alpha_h: float = 0.3,
        alpha_q: float = 0.3,
        beta: float = 0.9,
        tau_lower: float = 0.35,
        tau_upper: float = 0.55,
        eps: float = 1e-8,
        use_quality_eval: bool = True,
    ) -> None:
        if abs(alpha_s + alpha_h + alpha_q - 1.0) > 1e-6:
            raise ValueError("alpha_s + alpha_h + alpha_q must equal 1.0")

        self.num_clients = num_clients
        self.alpha_s = alpha_s
        self.alpha_h = alpha_h
        self.alpha_q = alpha_q
        self.tau_lower = tau_lower
        self.tau_upper = tau_upper
        self.eps = eps
        self.use_quality_eval = use_quality_eval

        # Reputation tracker (stateful across rounds)
        self.reputation = ReputationTracker(beta=beta)

        # Round counter and diagnostics log
        self._round: int = 0
        self._diagnostics: List[RoundDiagnostics] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        val_buffer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the complete AMFTA trust pipeline and return the aggregated update.

        Parameters
        ----------
        global_model : nn.Module
            The current global model (used for quality evaluation; not mutated).
        updates : UpdateDict
            {client_id: {param_name: delta_tensor}} from all clients.
        val_buffer : (X_val, y_val) | None
            Server-side validation buffer for quality evaluation.
            Required when use_quality_eval=True and borderline clients exist.

        Returns
        -------
        dict {param_name: tensor}
            Trust-weighted aggregated update ready to apply to global_model.
        """
        self._round += 1

        if not updates:
            raise ValueError("No updates received for aggregation.")

        diag = RoundDiagnostics(round_number=self._round)

        # ── Factor I: Gradient Similarity ──────────────────────────────────
        similarity = compute_gradient_similarity(updates, eps=self.eps)
        diag.similarity_scores = similarity

        # ── Factor II: Historical Reputation ───────────────────────────────
        history = self.reputation.update(similarity)
        diag.reputation_scores = history

        # ── Preliminary scores (for borderline detection) ──────────────────
        preliminary = compute_preliminary_scores(
            similarity, history, self.alpha_s, self.alpha_h
        )

        benign_ids, borderline_ids, suspect_ids = identify_borderline_clients(
            preliminary, self.tau_lower, self.tau_upper
        )
        diag.num_benign = len(benign_ids)
        diag.num_borderline = len(borderline_ids)
        diag.num_suspect = len(suspect_ids)

        # ── Factor III: Contribution Quality ───────────────────────────────
        if self.use_quality_eval and borderline_ids:
            if val_buffer is None:
                logger.warning(
                    "Borderline clients detected but val_buffer is None. "
                    "Defaulting quality scores to 0.5 for borderline clients."
                )
                quality = {
                    cid: (1.0 if cid in benign_ids else (0.0 if cid in suspect_ids else 0.5))
                    for cid in updates
                }
            else:
                quality = compute_contribution_quality(
                    global_model, updates, val_buffer,
                    borderline_ids=borderline_ids,
                    suspect_ids=suspect_ids,
                )
        else:
            # No borderline clients — assign Q by category
            quality = {
                cid: (1.0 if cid in benign_ids else 0.0)
                for cid in updates
            }
            if not self.use_quality_eval:
                # AMFTA-SH ablation: assign neutral quality
                quality = {cid: 0.5 for cid in updates}

        diag.quality_scores = quality

        # ── Final Trust Scores ─────────────────────────────────────────────
        trust = combine_trust_scores(
            similarity, history, quality,
            alpha_s=self.alpha_s,
            alpha_h=self.alpha_h,
            alpha_q=self.alpha_q,
        )
        diag.trust_scores = trust

        trust_vals = list(trust.values())
        diag.mean_trust = sum(trust_vals) / len(trust_vals)
        diag.min_trust = min(trust_vals)
        diag.max_trust = max(trust_vals)
        self._diagnostics.append(diag)

        logger.info(
            "Round %3d | benign=%d borderline=%d suspect=%d | "
            "mean_trust=%.3f min=%.3f max=%.3f",
            self._round,
            diag.num_benign, diag.num_borderline, diag.num_suspect,
            diag.mean_trust, diag.min_trust, diag.max_trust,
        )

        # ── Trust-Weighted Aggregation ─────────────────────────────────────
        return self._soft_aggregate(updates, trust)

    # ------------------------------------------------------------------
    # Soft trust-weighted aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_aggregate(
        updates: UpdateDict,
        trust: ScoreDict,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Compute the trust-weighted mean of all client updates.

        ḡ^{(t)} = Σ_i T_i^{(t)} · ĝ_i^{(t)} / Σ_i T_i^{(t)}

        Magnitude normalisation (critical for stability)
        ------------------------------------------------
        Trust scores only re-weight the *direction* of each update; they do not
        bound its *magnitude*.  Under label-flipping attacks, poisoned clients
        produce updates with disproportionately large norms.  Even at a small
        trust weight, a few such high-norm updates can dominate the weighted
        sum and flip the global model between the two trivial solutions
        (all-attack / all-normal), causing the round-to-round oscillation
        observed in early experiments.

        To prevent this, every client update is rescaled toward a common
        magnitude reference before trust-weighting (analogous to the root-norm
        rescaling used by FLTrust).  This bounds the influence of any single
        client to its trust weight, restoring convergence.  No client is
        hard-excluded — all contribute proportionally to trust.

        """
        client_ids = list(updates.keys())
        param_keys = list(next(iter(updates.values())).keys())

        # --- Robust per-client magnitude normalisation ---
        norms = {
            cid: torch.sqrt(
                sum((updates[cid][k].float() ** 2).sum() for k in param_keys)
            ).item()
            for cid in client_ids
        }
        sorted_norms = sorted(norms.values())
        target_norm = sorted_norms[len(sorted_norms) // 2]  # median norm
        scale = {
            cid: (target_norm / (norms[cid] + eps)) for cid in client_ids
        }

        total_trust = sum(trust.values())
        if total_trust <= 0.0:
            logger.warning("Total trust is zero; falling back to uniform aggregation.")
            total_trust = float(len(client_ids))
            trust = {cid: 1.0 for cid in client_ids}

        aggregated: Dict[str, torch.Tensor] = {}
        for k in param_keys:
            weighted_sum = sum(
                trust[cid] * scale[cid] * updates[cid][k].float()
                for cid in client_ids
            )
            aggregated[k] = weighted_sum / total_trust

        return aggregated

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset AMFTA state (reputation history and round counter).

        Call between independent experiments to avoid history leakage.
        """
        self.reputation.reset()
        self._round = 0
        self._diagnostics.clear()

    def get_diagnostics(self) -> List[dict]:
        """Return list of per-round diagnostic dicts for analysis."""
        return [d.to_dict() for d in self._diagnostics]

    def get_trust_history(self) -> Dict[int, List[float]]:
        """Return per-client trust score trajectory across all rounds."""
        if not self._diagnostics:
            return {}

        client_ids = list(self._diagnostics[0].trust_scores.keys())
        history: Dict[int, List[float]] = {cid: [] for cid in client_ids}

        for diag in self._diagnostics:
            for cid in client_ids:
                history[cid].append(diag.trust_scores.get(cid, 0.0))

        return history

    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "round": self._round,
            "reputation": self.reputation.state_dict(),
            "config": {
                "num_clients": self.num_clients,
                "alpha_s": self.alpha_s,
                "alpha_h": self.alpha_h,
                "alpha_q": self.alpha_q,
                "tau_lower": self.tau_lower,
                "tau_upper": self.tau_upper,
                "eps": self.eps,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state from a checkpoint."""
        self._round = sd["round"]
        self.reputation.load_state_dict(sd["reputation"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_round(self) -> int:
        return self._round

    def __repr__(self) -> str:
        return (
            f"AMFTAAggregator("
            f"α_s={self.alpha_s}, α_h={self.alpha_h}, α_q={self.alpha_q}, "
            f"β={self.reputation.beta}, "
            f"τ=[{self.tau_lower},{self.tau_upper}], "
            f"round={self._round})"
        )
