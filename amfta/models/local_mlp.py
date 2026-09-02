"""
Local MLP — Lightweight 3-Layer Binary Classifier for IoT Intrusion Detection.

Architecture: 45 → 64 → 32 → 1  (Binary classification via Sigmoid)

Designed to be deployable on constrained IoT edge devices.  The parameter
count (~5 057 for the default config) is intentionally small so that model
updates can be transmitted efficiently over bandwidth-limited wireless links.

Note on parameter count discrepancy
------------------------------------
The AMFTA paper states 3 393 parameters; the formula
  (45×64+64) + (64×32+32) + (32×1+1) = 2 944 + 2 080 + 33 = 5 057
yields a different value.  We implement the architecture as written
(45→64→32→1) and expose the actual count via ``LocalMLP.num_parameters()``.
If a smaller model is required to match the paper's count, set
``hidden1=48, hidden2=24`` (≈ 3 409 params).  The default here follows the
explicit layer dimensions stated in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class LocalMLP(nn.Module):
    """Lightweight 3-layer MLP for binary network traffic classification.

    Parameters
    ----------
    input_dim : int
        Number of input features. Default 45 (TON_IoT network flow features).
    hidden1 : int
        Size of first hidden layer. Default 64.
    hidden2 : int
        Size of second hidden layer. Default 32.
    dropout : float
        Dropout probability applied after each hidden layer. Default 0.0
        (disabled — keeps training behaviour deterministic for FL).
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)

        # Store config for serialisation
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Attack probability in [0, 1].
        """
        return self.net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a single flat vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Set parameters from a flat vector (inverse of get_flat_params)."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(flat[offset : offset + size].view(p.shape))
            offset += size

    def get_update_dict(self, reference: "LocalMLP") -> dict[str, torch.Tensor]:
        """Compute gradient delta: self.state_dict() − reference.state_dict().

        Used by clients to compute their model update to send to the server.
        """
        ref_sd = reference.state_dict()
        self_sd = self.state_dict()
        return {k: self_sd[k] - ref_sd[k] for k in self_sd}

    def apply_update(self, update: dict[str, torch.Tensor], lr: float = 1.0) -> None:
        """Apply an aggregated update delta to the current model in-place.

        Parameters
        ----------
        update : dict mapping parameter name → delta tensor
        lr     : global learning rate (η_global). Default 1.0 per paper.
        """
        sd = self.state_dict()
        for k, delta in update.items():
            sd[k] = sd[k] + lr * delta
        self.load_state_dict(sd)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def config_dict(self) -> dict:
        """Return constructor kwargs for reconstruction."""
        return {
            "input_dim": self.input_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
        }

    @classmethod
    def from_config(cls, config: dict) -> "LocalMLP":
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"LocalMLP(input_dim={self.input_dim}, "
            f"hidden1={self.hidden1}, hidden2={self.hidden2}, "
            f"params={self.num_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(config: Optional[dict] = None) -> LocalMLP:
    """Build a LocalMLP from an optional config dict.

    Falls back to paper defaults when config is None or keys are absent.
    """
    cfg = config or {}
    return LocalMLP(
        input_dim=cfg.get("input_dim", 45),
        hidden1=cfg.get("hidden1", 64),
        hidden2=cfg.get("hidden2", 32),
        dropout=cfg.get("dropout", 0.0),
    )


if __name__ == "__main__":
    model = build_model()
    print(model)
    x = torch.randn(8, 45)
    out = model(x)
    print(f"Output shape : {out.shape}")
    print(f"Output range : [{out.min():.4f}, {out.max():.4f}]")
