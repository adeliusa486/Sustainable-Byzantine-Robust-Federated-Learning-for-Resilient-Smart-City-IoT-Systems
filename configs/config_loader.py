"""
Configuration Loader
=====================

Loads YAML experiment configs with dot-access and CLI override support.
Falls back to pure YAML if OmegaConf is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Dict subclass with dot-access notation for nested keys."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return DotDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path = "configs/config.yaml") -> DotDict:
    """Load a YAML config file.

    Parameters
    ----------
    path : str | Path   Path to the YAML config.

    Returns
    -------
    DotDict  Dot-access config object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return DotDict(raw or {})


def merge_cli_overrides(config: DotDict, overrides: dict) -> DotDict:
    """Merge CLI argument overrides into config using dot-path notation.

    Example:
        merge_cli_overrides(cfg, {"federated.num_rounds": 50})
    """
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def config_from_env(config: DotDict) -> DotDict:
    """Override config fields from environment variables.

    Recognised env vars:
        AMFTA_NUM_CLIENTS, AMFTA_NUM_ROUNDS, AMFTA_BYZ_FRACTION,
        AMFTA_ATTACK_TYPE, AMFTA_SEED, AMFTA_USE_SYNTHETIC,
        AMFTA_RESULTS_DIR
    """
    env_map = {
        "AMFTA_NUM_CLIENTS":    ("federated", "num_clients", int),
        "AMFTA_NUM_ROUNDS":     ("federated", "num_rounds", int),
        "AMFTA_BYZ_FRACTION":   ("byzantine", "fraction", float),
        "AMFTA_ATTACK_TYPE":    ("byzantine", "attack_type", str),
        "AMFTA_SEED":           ("experiment", "seed", int),
        "AMFTA_USE_SYNTHETIC":  ("data", "use_synthetic", lambda x: x.lower() == "true"),
        "AMFTA_RESULTS_DIR":    ("output", "results_dir", str),
    }
    for env_var, (section, key, cast) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            config.setdefault(section, {})[key] = cast(val)

    return config
