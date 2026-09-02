"""
Shared pytest fixtures for AMFTA test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def set_deterministic():
    """Session-level determinism setup."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def tiny_model():
    """Minimal model for fast tests."""
    from amfta.models.local_mlp import LocalMLP
    torch.manual_seed(42)
    return LocalMLP(input_dim=10, hidden1=8, hidden2=4)


@pytest.fixture
def tiny_updates(tiny_model):
    """Three-client updates for the tiny model."""
    sd = tiny_model.state_dict()
    return {
        0: {k: torch.zeros_like(v.float()) + 0.01 for k, v in sd.items()},
        1: {k: torch.zeros_like(v.float()) + 0.01 for k, v in sd.items()},
        2: {k: torch.zeros_like(v.float()) - 0.05 for k, v in sd.items()},
    }
