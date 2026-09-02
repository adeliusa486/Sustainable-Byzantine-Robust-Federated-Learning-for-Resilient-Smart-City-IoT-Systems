"""
AGR-tailored adaptive attack (Fang et al., USENIX Security 2020).
=================================================================

The manuscript reports an adaptive attack in which

    "Colluding adversaries observe the AGR and craft updates that maximize
     deviation of the aggregate while remaining inside the region the rule
     accepts ... the coalition computes the aggregate that would result from
     honest behavior, chooses a malicious direction, and scales its
     perturbation to the largest magnitude that the defense still accepts."

No such attack existed in this package; the implemented families were label
flipping, Gaussian noise, sign flipping and mimicry. This module supplies it.

Mechanism
---------
Given the honest updates of the current round, the coalition:

  1. estimates the honest consensus direction ``mu_h`` and its dispersion;
  2. sets the malicious direction to ``-mu_h`` (steepest damage to the
     aggregate) or to the sign-inverted coordinate-wise mean, per ``mode``;
  3. binary-searches the perturbation magnitude ``gamma`` for the largest
     value that the target aggregation rule still accepts, where acceptance
     is evaluated with the actual server-side rule supplied by the caller;
  4. emits identical (or jittered) updates for every compromised client.

Because the coalition is scored against a centroid it partially controls,
the attack is strongest against similarity-to-centroid rules, which is the
worst case the manuscript sets out to test.

Threat-model parameters that the manuscript says must be specified
------------------------------------------------------------------
attacker knowledge   : full knowledge of honest updates in the current round
                       (``full``) or of the compromised subset only (``partial``)
initialisation       : gamma_init
optimisation         : binary search on gamma, ``search_steps`` halvings
constraints          : gamma in [gamma_min, gamma_init]
stopping criterion   : first gamma accepted by the AGR, or gamma_min
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from amfta.attacks.base import BaseAttack, register_attack

EPS = 1e-8


def _flatten(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in sd.values()])


def _unflatten_like(flat: torch.Tensor,
                    template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out, off = {}, 0
    for k, v in template.items():
        n = v.numel()
        out[k] = flat[off:off + n].view(v.shape).clone()
        off += n
    return out


@register_attack("adaptive")
class AdaptiveAGRAttack(BaseAttack):
    """Optimisation-based attack tailored to the aggregation rule.

    Parameters
    ----------
    mode : {'directed', 'sign'}
        'directed' perturbs along ``-mu_h``; 'sign' along the inverted
        coordinate-wise sign of ``mu_h`` (the Fang "full-knowledge" variant).
    gamma_init : float
        Initial perturbation scale, in multiples of ``||mu_h||``.
    gamma_min : float
        Lower bound below which the attack gives up and submits ``-mu_h``
        scaled to the honest median norm.
    search_steps : int
        Number of binary-search halvings.
    jitter : float
        Relative Gaussian jitter added per compromised client, so that the
        coalition is not trivially detectable by exact duplication.
    knowledge : {'full', 'partial'}
        Whether the coalition sees all honest updates or only its own.
    """

    def __init__(
        self,
        mode: str = "directed",
        gamma_init: float = 10.0,
        gamma_min: float = 0.05,
        search_steps: int = 12,
        jitter: float = 0.02,
        knowledge: str = "full",
        seed: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.gamma_init = gamma_init
        self.gamma_min = gamma_min
        self.search_steps = search_steps
        self.jitter = jitter
        self.knowledge = knowledge
        self.seed = seed

    # ------------------------------------------------------------------
    # Coalition-level crafting
    # ------------------------------------------------------------------

    def craft_coalition(
        self,
        global_model: nn.Module,
        honest_updates: Dict[int, Dict[str, torch.Tensor]],
        byzantine_ids: Sequence[int],
        accepts: Optional[Callable[[Dict[int, Dict[str, torch.Tensor]]], bool]] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Craft one poisoned update per compromised client.

        Parameters
        ----------
        honest_updates : the honest clients' updates for this round
        byzantine_ids  : ids to emit updates for
        accepts        : optional oracle returning True when the aggregate
                         produced with a candidate coalition still lies within
                         the region the AGR accepts. When None, the magnitude
                         is fixed at the honest median norm, which is the
                         norm-clipping-aware setting.
        """
        if not honest_updates:
            raise ValueError("adaptive attack needs at least one honest update")

        template = next(iter(honest_updates.values()))
        H = torch.stack([_flatten(u) for u in honest_updates.values()])
        mu_h = H.mean(dim=0)
        median_norm = float(H.norm(dim=1).median())

        if self.mode == "sign":
            direction = -torch.sign(mu_h)
        else:
            direction = -mu_h
        dn = direction.norm()
        direction = direction / (dn + EPS)

        gamma = self.gamma_init * median_norm

        if accepts is not None:
            lo, hi = self.gamma_min * median_norm, gamma
            best = lo
            for _ in range(self.search_steps):
                mid = 0.5 * (lo + hi)
                cand = self._emit(direction * mid, template, byzantine_ids)
                if accepts(cand):
                    best, lo = mid, mid
                else:
                    hi = mid
                if hi - lo < 1e-6 * median_norm:
                    break
            gamma = best
        else:
            # Without an acceptance oracle the informative setting is the
            # largest magnitude that median-norm rescaling leaves untouched.
            gamma = median_norm

        return self._emit(direction * gamma, template, byzantine_ids)

    def _emit(self, payload: torch.Tensor, template: Dict[str, torch.Tensor],
              byzantine_ids: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
        g = torch.Generator(device=payload.device)
        if self.seed is not None:
            g.manual_seed(self.seed)
        out = {}
        for cid in byzantine_ids:
            v = payload
            if self.jitter > 0:
                noise = torch.randn(payload.shape, generator=g,
                                    device=payload.device)
                v = payload + self.jitter * payload.norm() * noise / (noise.norm() + EPS)
            out[cid] = _unflatten_like(v, template)
        return out

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        honest_updates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Per-client entry point.

        The adaptive attack is inherently coalition-level: it needs the round's
        honest updates. The runner therefore calls ``craft_coalition`` once per
        round. This method exists so the class satisfies ``BaseAttack``, and it
        degrades to a single-client directed perturbation when the coalition
        context is unavailable.
        """
        if honest_updates:
            crafted = self.craft_coalition(global_model, honest_updates, [0])
            return crafted[0]

        # No coalition context: fall back to an inverted honest update.
        from amfta.attacks.base import _train_model
        X, y = local_data
        model = copy.deepcopy(global_model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model = _train_model(model, X.float(), y.float(), epochs, lr, batch_size)
        return {k: -(model.state_dict()[k] - before[k]) for k in before}

    def __repr__(self) -> str:
        return (f"AdaptiveAGRAttack(mode={self.mode}, knowledge={self.knowledge}, "
                f"gamma_init={self.gamma_init})")
