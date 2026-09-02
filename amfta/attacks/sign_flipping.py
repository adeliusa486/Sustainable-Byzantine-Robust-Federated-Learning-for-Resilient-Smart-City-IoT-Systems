"""
Sign-Flipping (Collusion) Attack
================================

A coordinated model-poisoning attack in which Byzantine clients compute an
honest local update and then *negate* it (optionally amplified), pushing the
global model in the opposite direction.  Because all colluding attackers flip
the same honest gradient, their poisoned updates are mutually *consistent*
(geometrically coherent), which lets them survive naive outlier filters that
assume malicious updates are scattered.

This is the canonical model-poisoning attack used to stress Byzantine-robust
aggregators (cf. Blanchard et al. 2017; Cao et al. 2022; FedDBC 2026) and is
a stronger, more standard benchmark than label flipping.

Attack mechanism:
  1. Receive global model w(t-1)
  2. Train honestly on (X, y) to obtain g_honest
  3. Submit g_i = -scale * g_honest

Detectability: directionally opposite to honest updates (detectable by cosine
similarity), but colluding attackers form a tight cluster, which is precisely
what density / clustering defenses must contend with.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn

from amfta.attacks.base import BaseAttack, _train_model, register_attack


@register_attack("sign_flipping")
class SignFlippingAttack(BaseAttack):
    """Byzantine attack: negate the honest update (collusion-coherent).

    Parameters
    ----------
    scale : float
        Amplification factor applied to the negated honest update.
        Default 1.0 (pure sign flip).  Larger values increase impact but
        also increase update magnitude (more detectable by norm-based defenses).
    """

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0; got {scale}")
        self.scale = scale

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

        model = copy.deepcopy(global_model)
        w_before = {k: v.clone() for k, v in model.state_dict().items()}
        model = _train_model(model, X, y, epochs, lr, batch_size)

        update = {
            k: -self.scale * (model.state_dict()[k] - w_before[k]) for k in w_before
        }
        return update

    def __repr__(self) -> str:
        return f"SignFlippingAttack(scale={self.scale})"
