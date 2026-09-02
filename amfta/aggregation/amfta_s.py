"""
AMFTA-S: resource-aware scheduling wrapped around the trust engine.
===================================================================

The manuscript specifies this method but the codebase never contained it, so
Experiments 6, 7 and 8 of the redesigned study could not be run.

Two layers, kept apart on purpose:

  security        AMFTA-ND scores the update, and nothing about the client's
                  hardware enters that score.
  sustainability  a resource score decides how *often* a client is asked to
                  work and how much local work it does, and nothing about the
                  client's trustworthiness enters that decision.

They meet only here, in the controller, which samples participation and applies
an importance correction so that a resource-poor honest client's expected
contribution is unchanged (Proposition 1). That same correction is what a client
understating its resources exploits, which is why the participation floor
``p_min`` bounds the amplification at ``1/p_min`` (Proposition 2).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from amfta.aggregation.amfta import AMFTAAggregator

logger = logging.getLogger(__name__)

EPS = 1e-12


class ResourceProfile:
    """Per-client device characteristics used only by the sustainability layer.

    ``budget_j`` is what makes the score fair: cost is expressed as a fraction
    of what the device itself can afford, so a battery node is not ranked
    cheaper than a mains gateway merely because it draws less absolute power.
    """

    TIERS = {
        "A": dict(throughput=1.0e10, power=6.00, e_bit=5.0e-8, e_fix=5.0e-2, budget=1.0e7),
        "B": dict(throughput=5.0e8,  power=1.50, e_bit=1.0e-6, e_fix=3.0e-1, budget=1.0e6),
        "C": dict(throughput=2.0e7,  power=0.09, e_bit=2.0e-5, e_fix=1.5e-1, budget=3.24e4),
    }
    SHARES = ("A", 0.20), ("B", 0.45), ("C", 0.35)

    def __init__(self, num_clients: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        counts = {k: int(np.floor(f * num_clients)) for k, f in self.SHARES}
        counts["C"] += num_clients - sum(counts.values())
        tiers = np.concatenate([np.full(c, k) for k, c in counts.items()])
        rng.shuffle(tiers)
        self.tier = {i: str(tiers[i]) for i in range(num_clients)}

    def params(self, cid: int) -> dict:
        return self.TIERS[self.tier[cid]]


def compute_scores(n_samples: Dict[int, int], profile: ResourceProfile,
                   d: int, epochs: int, batch: int,
                   lam: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
                   s_min: float = 0.05,
                   misreport: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """Sustainability score in [s_min, 1]; 1 means cheap to ask, given the budget.

    ``misreport`` scales a client's declared cost, and models the resource
    spoofing of Experiment 8: a factor below one understates the true cost.
    """
    cids = sorted(n_samples)
    flops, energy, comm, latency = {}, {}, {}, {}
    for c in cids:
        p = profile.params(c)
        n = n_samples[c]
        f = epochs * d * (3.0 * n + 4.0 * np.ceil(n / batch))
        t = f / p["throughput"]
        bits = 2 * 4 * d * 8
        e_comm = p["e_fix"] + p["e_bit"] * bits
        e_tot = t * p["power"] + e_comm
        k = (misreport or {}).get(c, 1.0)
        flops[c], latency[c] = f * k, t * k
        energy[c] = (e_tot / p["budget"]) * k
        comm[c] = (e_comm / p["budget"]) * k

    def norm_log(v: Dict[int, float]) -> Dict[int, float]:
        a = np.log10(np.maximum(np.array([v[c] for c in cids]), 1e-30))
        rng = a.max() - a.min()
        z = np.zeros_like(a) if rng == 0 else (a - a.min()) / rng
        return {c: float(z[i]) for i, c in enumerate(cids)}

    ne, nc, nf, nt = norm_log(energy), norm_log(comm), norm_log(flops), norm_log(latency)
    le, lc, lf, lt = lam
    return {c: float(np.clip(1.0 - (le * ne[c] + lc * nc[c] + lf * nf[c] + lt * nt[c]),
                             s_min, 1.0)) for c in cids}


def schedule(scores: Dict[int, float], p_min: float = 0.25, gamma: float = 1.0,
             base_epochs: int = 5, e_min: int = 1) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Participation probability and local epoch count, both server-assigned.

    The floor does two jobs: it keeps every client contributing evidence often
    enough for the reputation EMA to stay meaningful, and it caps the
    importance weight at 1/p_min.
    """
    mean = float(np.mean(list(scores.values()))) or 1.0
    p = {c: float(np.clip((s / mean) ** gamma, p_min, 1.0)) for c, s in scores.items()}
    e = {c: int(np.clip(round(base_epochs * s / mean), e_min, base_epochs))
         for c, s in scores.items()}
    return p, e


class AMFTASAggregator(AMFTAAggregator):
    """AMFTA-ND plus the importance correction for resource-aware sampling.

    ``participation`` holds the probabilities the controller assigned this
    round. Only clients that were actually sampled appear in ``updates``; each
    is reweighted by 1/p_i so the expected contribution matches what full
    participation would have produced.
    """

    def __init__(self, *args, p_min: float = 0.25, **kwargs):
        kwargs.setdefault("use_quality_eval", False)
        kwargs.setdefault("alpha_s", 0.57)
        kwargs.setdefault("alpha_h", 0.43)
        kwargs.setdefault("alpha_q", 0.0)
        super().__init__(*args, **kwargs)
        self.p_min = p_min
        self.participation: Dict[int, float] = {}

    def aggregate(self, global_model: nn.Module, updates, val_buffer=None):
        if not self.participation:
            return super().aggregate(global_model, updates, val_buffer=None)

        # trust is computed exactly as in AMFTA-ND, on the participating set
        out = super().aggregate(global_model, updates, val_buffer=None)
        trust = (self._diagnostics[-1].trust_scores
                 if getattr(self, "_diagnostics", None) else None)

        if not trust:
            # the parent did not expose per-client trust; fall back to the
            # uncorrected aggregate rather than guessing at weights
            logger.warning("AMFTA-S: per-client trust unavailable; "
                           "importance correction skipped this round.")
            return out

        w = {}
        for cid in updates:
            p = max(self.participation.get(cid, 1.0), self.p_min)
            w[cid] = float(trust.get(cid, 0.0)) / p
        total = sum(w.values())
        if total <= EPS:
            return out

        # median-norm rescaling, then the importance-corrected weighted mean
        flat = {c: torch.cat([v.float().flatten() for v in u.values()])
                for c, u in updates.items()}
        norms = {c: float(v.norm()) for c, v in flat.items()}
        target = sorted(norms.values())[len(norms) // 2]

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())):
            acc = None
            for cid, u in updates.items():
                scale = target / (norms[cid] + 1e-8)
                term = u[k].float() * scale * (w[cid] / total)
                acc = term if acc is None else acc + term
            result[k] = acc
        return result
