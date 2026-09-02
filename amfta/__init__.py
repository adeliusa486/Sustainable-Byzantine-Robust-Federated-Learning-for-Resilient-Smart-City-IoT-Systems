"""
AMFTA — Adaptive Multi-Factor Trust Aggregation
for Byzantine-Resilient Federated Learning in Non-IID Smart City IoT Networks

IEEE Internet of Things Journal Submission

This package provides:
  - AMFTA trust-weighted aggregation engine
  - Byzantine attack simulators (label flipping, Gaussian noise)
  - Baseline aggregators (FedAvg, Krum, Trimmed Mean, FLTrust)
  - Data preprocessing and Dirichlet non-IID partitioning
  - Federated training orchestration
  - Evaluation metrics and visualisation

Quick start:
    from amfta.aggregation.amfta import AMFTAAggregator
    from amfta.models.local_mlp import LocalMLP
    from amfta.data.partitioning import dirichlet_partition
"""

__version__ = "1.0.0"
__author__ = "AMFTA Research Team"
__license__ = "MIT"

from amfta.models.local_mlp import LocalMLP
from amfta.aggregation.amfta import AMFTAAggregator

__all__ = ["LocalMLP", "AMFTAAggregator", "__version__"]
