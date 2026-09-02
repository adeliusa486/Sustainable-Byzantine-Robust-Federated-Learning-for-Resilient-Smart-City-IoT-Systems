"""Base class and registry for Byzantine attack strategies."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BaseAttack(ABC):
    """Abstract base class for Byzantine client attack strategies.

    Each attack produces a poisoned model update (gradient delta) that the
    Byzantine client submits to the server in place of its honest update.
    """

    @abstractmethod
    def get_update(
        self,
        global_model: nn.Module,
        local_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce a malicious model update.

        Parameters
        ----------
        global_model : nn.Module
            The current global model (copied internally; not mutated).
        local_data   : (X, y) tensors
            Client's local dataset.
        epochs       : int   Local training epochs.
        lr           : float Local learning rate.
        batch_size   : int   Mini-batch size.

        Returns
        -------
        dict {param_name: delta_tensor}
            Poisoned model update to send to the server.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Helpers shared across attacks
# ---------------------------------------------------------------------------

def _train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> nn.Module:
    """Standard local SGD training loop."""
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss()

    model.train()
    X = X.float().to(device)
    y = y.float().to(device)
    num_samples = len(X)
    for _ in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        X_shuff = X[indices]
        y_shuff = y[indices]
        for i in range(0, num_samples, batch_size):
            X_b = X_shuff[i:i+batch_size]
            y_b = y_shuff[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Attack registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {}


def register_attack(name: str):
    """Decorator to register an attack class by name."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_attack(name: str, **kwargs) -> BaseAttack:
    """Instantiate an attack by name.

    Parameters
    ----------
    name : str  One of 'label_flipping', 'gaussian_noise', 'none'.
    **kwargs    Passed to the attack constructor.
    """
    key = name.lower().replace("-", "_")
    if key == "none":
        from amfta.attacks.honest import HonestClient
        return HonestClient()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown attack '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**kwargs)
