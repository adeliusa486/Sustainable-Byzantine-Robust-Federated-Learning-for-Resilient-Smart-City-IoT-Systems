"""
Label Flipping Attack
======================

Byzantine clients train on inverted labels (0→1, 1→0) to push the global
model toward the inverse classification boundary.  This is a strong, simple
attack that is particularly effective in binary classification tasks.

Attack mechanism:
  1. Receive global model w(t-1)
  2. Flip all local labels: y' = 1 - y
  3. Train on (X, y') for E epochs with SGD
  4. Submit update g_i = w_trained - w(t-1) to server

Detectability: Produces updates directionally opposite to honest updates,
making them easily detected by cosine similarity (Factor I of AMFTA).
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn

from amfta.attacks.base import BaseAttack, _train_model, register_attack


@register_attack("label_flipping")
class LabelFlippingAttack(BaseAttack):
    """Byzantine attack: flip all binary labels before local training.

    Parameters
    ----------
    flip_fraction : float
        Fraction of labels to flip. Default 1.0 (full flip as in paper).
        Set < 1.0 for partial label flipping (more stealthy variant).
    """

    def __init__(self, flip_fraction: float = 1.0) -> None:
        if not 0.0 < flip_fraction <= 1.0:
            raise ValueError(f"flip_fraction must be in (0, 1]; got {flip_fraction}")
        self.flip_fraction = flip_fraction

    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce a label-flipping poisoned update."""
        X, y = local_data
        X, y = X.float(), y.float()

        # Flip labels
        if self.flip_fraction == 1.0:
            y_poisoned = 1.0 - y
        else:
            mask = torch.rand(len(y)) < self.flip_fraction
            y_poisoned = y.clone()
            y_poisoned[mask] = 1.0 - y_poisoned[mask]

        # Deep copy model to avoid mutating the original
        model = copy.deepcopy(global_model)
        w_before = {k: v.clone() for k, v in model.state_dict().items()}

        # Train on poisoned labels
        model = _train_model(model, X, y_poisoned, epochs, lr, batch_size)

        # Compute delta
        update = {k: model.state_dict()[k] - w_before[k] for k in w_before}
        return update

    def __repr__(self) -> str:
        return f"LabelFlippingAttack(flip_fraction={self.flip_fraction})"
