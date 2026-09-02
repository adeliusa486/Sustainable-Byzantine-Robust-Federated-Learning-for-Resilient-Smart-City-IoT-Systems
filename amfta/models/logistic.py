"""
Logistic Regression — the convex model the manuscript evaluates.
================================================================

The manuscript scopes every result to a convex model:

    "Every method trains an identical logistic-regression model,
     y = sigma(w^T x) ... a convex model isolates the behavior of the
     aggregation rule from optimization pathologies of deep networks,
     and makes the geometry of the update space interpretable."

Only ``LocalMLP`` existed in this package, so that configuration could not be
run. This module supplies it.

Parameter count
---------------
For an input of ``input_dim`` features the model has ``input_dim + 1``
parameters (weights plus bias). The processed TON_IoT tensors in
``data/processed`` carry 45 features, giving d = 46. The manuscript states
d = 41, which corresponds to 40 features; if you intend to reproduce that
figure exactly you must also reproduce the feature-selection step that
reduces 45 columns to 40, and record it in the preprocessing pipeline.
``num_parameters()`` always reports the true count for the model actually
built, so the manuscript's Table can be filled from the code rather than
asserted independently of it.

This class subclasses ``LocalMLP`` purely to inherit the shared parameter
plumbing (``apply_update``, ``get_flat_params``, ``get_update_dict``); it
replaces the network with a single affine layer and a sigmoid.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from amfta.models.local_mlp import LocalMLP


class LogisticRegression(LocalMLP):
    """Binary logistic regression: ``y = sigma(w^T x + b)``.

    Parameters
    ----------
    input_dim : int
        Number of input features. The model has ``input_dim + 1`` parameters.
    """

    def __init__(self, input_dim: int = 45, **_ignored) -> None:
        # Build the parent, then replace its network. The parent's __init__
        # signature is reused so that config_dict()/from_config() keep working.
        nn.Module.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

        self.input_dim = input_dim
        self.hidden1 = 0          # kept for config_dict() compatibility
        self.hidden2 = 0

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def config_dict(self) -> dict:
        return {"model": "logistic", "input_dim": self.input_dim}

    def __repr__(self) -> str:
        return (
            f"LogisticRegression(input_dim={self.input_dim}, "
            f"d={self.num_parameters()})"
        )


def build_global_model(model_class: str, input_dim: int,
                       hidden1: int = 64, hidden2: int = 32):
    """Construct the global model named by ``model_class``.

    Parameters
    ----------
    model_class : {'logistic', 'mlp'}
        'logistic' reproduces the manuscript's stated configuration;
        'mlp' reproduces the model the released experiments actually used.
    """
    mc = model_class.lower()
    if mc in ("logistic", "logreg", "lr"):
        return LogisticRegression(input_dim=input_dim)
    if mc == "mlp":
        return LocalMLP(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2)
    raise ValueError(f"Unknown model_class '{model_class}'; use 'logistic' or 'mlp'.")
