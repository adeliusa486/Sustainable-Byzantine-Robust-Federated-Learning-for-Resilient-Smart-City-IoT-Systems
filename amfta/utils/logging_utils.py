"""
Experiment Logging Utilities
==============================

Structured logging for federated learning experiments.  Writes per-round
metrics to CSV for easy analysis and to JSON for structured export.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    name: str = "amfta",
) -> logging.Logger:
    """Configure root logger with console and optional file handler.

    Parameters
    ----------
    log_level : str   Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    log_file  : Path  If provided, also writes to this file.
    name      : str   Logger name.

    Returns
    -------
    logging.Logger  Configured logger instance.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        os.makedirs(log_file.parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    return logging.getLogger(name)


class ExperimentLogger:
    """Logs per-round metrics to CSV and JSON for offline analysis.

    Parameters
    ----------
    experiment_name : str   Identifier for the experiment.
    results_dir     : Path  Directory to write output files.
    """

    def __init__(self, experiment_name: str, results_dir: Path = Path("results")) -> None:
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.results_dir / f"{experiment_name}_{ts}.csv"
        self.json_path = self.results_dir / f"{experiment_name}_{ts}.json"

        self._rows: List[Dict[str, Any]] = []
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None

        self._logger = logging.getLogger(f"amfta.experiment.{experiment_name}")

    def log_round(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a single training round."""
        row = {"round": round_num, **metrics}
        self._rows.append(row)

        # Lazy CSV initialisation (need fieldnames from first row)
        if self._csv_writer is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()

        # Write only scalar-valued fields to CSV
        scalar_row = {
            k: v for k, v in row.items() if isinstance(v, (int, float, str, bool))
        }
        self._csv_writer.writerow(scalar_row)
        self._csv_file.flush()

        self._logger.info(
            "Round %3d | acc=%.4f f1=%.4f precision=%.4f recall=%.4f",
            round_num,
            metrics.get("accuracy", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
        )

    def save(self) -> None:
        """Flush all results to JSON."""
        with open(self.json_path, "w") as f:
            json.dump(self._rows, f, indent=2, default=str)
        self._logger.info("Results saved -> %s, %s", self.csv_path, self.json_path)

    def close(self) -> None:
        """Close open file handles."""
        if self._csv_file is not None:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save()
        self.close()

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._rows)
