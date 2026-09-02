"""
Gaussian Noise Attack
======================

Byzantine clients replace their computed gradient with pure Gaussian noise.
The noise magnitude σ is scaled relative to the mean norm of honest updates
to make it realistic (not trivially detectable by magnitude alone).

Attack mechanism:
  g_i^{(t)} ~ N(0, σ² · I)

As σ → ∞ this becomes an unbounded attack; as σ → 0 it approaches a
benign zero-update.  The paper uses σ scaled to match legitimate update
magnitudes (exact value not specified; we implement a configurable scale).

Note: The paper does not specify the exact σ value used.  Our default
uses σ=1.0 (unit Gaussian) which is a reasonable middle ground.  Use
the `scale_to_honest` flag to auto-scale σ based on received updates.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from amfta.attacks.base import BaseAttack, register_attack


@register_attack("gaussian_noise")
class GaussianNoiseAttack(BaseAttack):
    """Byzantine attack: submit random Gaussian noise as model update.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian noise.  Default 1.0.
    scale_to_honest : bool
        If True, sigma is multiplied by the mean L2 norm of the honest
        updates passed via the ``honest_norm_ref`` kwarg to get_update.
        Enables more realistic noise magnitude.
    seed : int | None
        RNG seed for reproducible noise.  Default None (non-deterministic).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        scale_to_honest: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.sigma = sigma
        self.scale_to_honest = scale_to_honest
        self.seed = seed

    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        honest_norm_ref: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce a Gaussian noise poisoned update.

        Parameters
        ----------
        honest_norm_ref : float | None
            Reference L2 norm for scaling sigma.  Used when scale_to_honest=True.
            Typically set to the mean norm of honest updates from the previous round.
        """
        effective_sigma = self.sigma
        if self.scale_to_honest and honest_norm_ref is not None:
            effective_sigma = self.sigma * honest_norm_ref

        noise_update: Dict[str, torch.Tensor] = {}
        for k, v in global_model.state_dict().items():
            generator = torch.Generator(device=v.device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
            noise_update[k] = torch.randn(v.shape, generator=generator, device=v.device) * effective_sigma

        return noise_update

    def __repr__(self) -> str:
        return (
            f"GaussianNoiseAttack(sigma={self.sigma}, "
            f"scale_to_honest={self.scale_to_honest})"
        )


# ---------------------------------------------------------------------------
# Honest client (no attack — used as a pass-through for benign clients)
# ---------------------------------------------------------------------------

class HonestClient(BaseAttack):
    """Placeholder attack class representing an honest client.

    Performs standard local training and returns the true update.
    """

    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        from amfta.attacks.base import _train_model
        X, y = local_data
        model = copy.deepcopy(global_model)
        w_before = {k: v.clone() for k, v in model.state_dict().items()}
        model = _train_model(model, X.float(), y.float(), epochs, lr, batch_size)
        return {k: model.state_dict()[k] - w_before[k] for k in w_before}
