"""
Attenuated Mimicry (Adaptive) Attack
====================================

An *adaptive* model-poisoning attack designed to evade both direction-based
(cosine similarity / clustering) and magnitude-based (norm clipping) defenses.

The attacker computes both an honest update g_honest and a malicious
(label-flipped) update g_mal, then submits a stealthy blend:

    g_i = normalise( s * g_honest + (1 - s) * g_mal ) * ||g_honest||

The blend direction is pulled toward the honest gradient (raising cosine
similarity to the benign consensus so it survives clustering / similarity
filters), while the rescaling to the honest norm removes the magnitude
signature that norm-aware defenses rely on.  The residual malicious component
still drifts the global model.  This is the hardest case for any single-signal
defense and mirrors the "attenuated mimicry / multi-cluster" stress tests used
by recent NIDS aggregation work (FedDBC 2026).

Parameter ``stealth`` in [0, 1]:
  - 1.0  → fully honest (no attack)
  - 0.0  → undisguised label-flip (rescaled to honest norm)
  - 0.5  → balanced stealth/impact (default)
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn

from amfta.attacks.base import BaseAttack, _train_model, register_attack


@register_attack("mimicry")
class MimicryAttack(BaseAttack):
    """Adaptive attack blending honest and malicious updates, norm-camouflaged."""

    def __init__(self, stealth: float = 0.5) -> None:
        if not 0.0 <= stealth <= 1.0:
            raise ValueError(f"stealth must be in [0, 1]; got {stealth}")
        self.stealth = stealth

    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        X, y = local_data
        X, y = X.float(), y.float()

        # Honest update
        m_h = copy.deepcopy(global_model)
        w0 = {k: v.clone() for k, v in m_h.state_dict().items()}
        m_h = _train_model(m_h, X, y, epochs, lr, batch_size)
        g_honest = {k: m_h.state_dict()[k] - w0[k] for k in w0}

        # Malicious (label-flipped) update
        m_m = copy.deepcopy(global_model)
        m_m = _train_model(m_m, X, 1.0 - y, epochs, lr, batch_size)
        g_mal = {k: m_m.state_dict()[k] - w0[k] for k in w0}

        s = self.stealth
        blend = {k: s * g_honest[k] + (1.0 - s) * g_mal[k] for k in w0}

        # Rescale blend to the honest-update norm (magnitude camouflage)
        honest_norm = torch.sqrt(sum((g_honest[k].float() ** 2).sum() for k in w0))
        blend_norm = torch.sqrt(sum((blend[k].float() ** 2).sum() for k in w0)) + 1e-8
        scale = (honest_norm / blend_norm).item()

        return {k: blend[k] * scale for k in w0}

    def __repr__(self) -> str:
        return f"MimicryAttack(stealth={self.stealth})"
