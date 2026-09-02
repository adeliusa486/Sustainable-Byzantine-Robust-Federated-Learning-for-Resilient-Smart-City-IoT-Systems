"""
Reproducibility Utilities
==========================

Centralised seed management and determinism configuration for all experiments.

Principles:
  - All experiments must set seeds BEFORE any data loading or model init.
  - PyTorch determinism flags are enabled by default.
  - Seeds for 5 experiment runs: {42, 123, 456, 789, 1024} (from paper).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

PAPER_SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all RNG seeds for reproducibility.

    Parameters
    ----------
    seed        : int   Seed value.
    deterministic: bool Enable PyTorch deterministic algorithms.
                        Slightly slower but required for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # PyTorch < 1.8
            pass

    logger.debug("Seed set to %d (deterministic=%s)", seed, deterministic)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available compute device.

    Parameters
    ----------
    prefer_gpu : bool  Use CUDA if available.

    Returns
    -------
    torch.device  'cuda' or 'cpu'.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    return device


class ExperimentContext:
    """Context manager that sets seeds and restores RNG state afterwards.

    Usage:
        with ExperimentContext(seed=42):
            run_experiment(...)
    """

    def __init__(self, seed: int, deterministic: bool = True) -> None:
        self.seed = seed
        self.deterministic = deterministic
        self._np_state: Optional[np.random.RandomState] = None
        self._torch_state = None

    def __enter__(self) -> "ExperimentContext":
        self._np_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, *args) -> None:
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        if self._torch_state is not None:
            torch.set_rng_state(self._torch_state)
