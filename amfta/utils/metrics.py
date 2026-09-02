"""
Evaluation Metrics for FL Intrusion Detection
===============================================

Provides per-round model evaluation returning accuracy, F1, precision,
recall, and AUC.  F1 is the primary metric used in the paper due to the
class imbalance in TON_IoT (38% normal / 62% attack).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

MetricsDict = Dict[str, float]


def evaluate_model(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> MetricsDict:
    """Evaluate a binary classifier on a dataset.

    Parameters
    ----------
    model     : nn.Module    Trained model.
    X         : array-like   Feature matrix.
    y         : array-like   Ground-truth binary labels.
    threshold : float        Decision threshold for binary classification.
    batch_size: int          Inference batch size.
    device    : torch.device Target device.  None = auto-detect.

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, auc
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X.astype(np.float32))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y.astype(np.int64))

    dataset = TensorDataset(X.float(), y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs: list = []
    all_labels: list = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            probs = model(X_b).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_b.numpy().tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= threshold).astype(int)

    metrics: MetricsDict = {
        "accuracy":  float(accuracy_score(labels_arr, preds_arr)),
        "f1":        float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall":    float(recall_score(labels_arr, preds_arr, zero_division=0)),
    }

    # AUC requires probabilities, not hard predictions
    try:
        metrics["auc"] = float(roc_auc_score(labels_arr, probs_arr))
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


@torch.no_grad()
def evaluate_model_fast(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 65536,
    device: Optional[torch.device] = None,
) -> MetricsDict:
    """Fully-vectorised GPU evaluation for large test sets.

    Computes accuracy/precision/recall/f1 by accumulating TP/FP/TN/FN counts
    on the GPU in large batches.  Avoids the per-sample Python-list / sklearn
    path in ``evaluate_model``, which is ~100x slower on multi-million-sample
    test sets.  Numerically identical results (same 0.5 threshold).

    Parameters
    ----------
    model : nn.Module       Trained model.
    X, y  : torch.Tensor    Test features / binary labels (CPU or GPU).
    threshold : float       Decision threshold.
    batch_size : int        Inference batch size.
    device : torch.device   Compute device. None = auto-detect.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    tp = fp = tn = fn = 0
    n = X.shape[0]
    for i in range(0, n, batch_size):
        X_b = X[i:i + batch_size].to(device, non_blocking=True).float()
        y_b = y[i:i + batch_size].to(device, non_blocking=True).float()
        preds = (model(X_b) >= threshold).float()
        tp += torch.sum((preds == 1) & (y_b == 1)).item()
        fp += torch.sum((preds == 1) & (y_b == 0)).item()
        tn += torch.sum((preds == 0) & (y_b == 0)).item()
        fn += torch.sum((preds == 0) & (y_b == 1)).item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_confusion(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> np.ndarray:
    """Return confusion matrix [[TN FP] [FN TP]]."""
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X.astype(np.float32))
    else:
        X_t = X.float()

    model.eval()
    with torch.no_grad():
        preds = (model(X_t) >= threshold).long().numpy()
    labels = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    return confusion_matrix(labels, preds)


def aggregate_metrics(results: list[MetricsDict]) -> Dict[str, Tuple[float, float]]:
    """Compute mean ± std over multiple seed runs.

    Parameters
    ----------
    results : list of MetricsDict  One dict per seed.

    Returns
    -------
    dict {metric: (mean, std)}
    """
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: (float(np.mean([r[k] for r in results])),
            float(np.std([r[k] for r in results])))
        for k in keys
    }


def format_metrics(metrics: MetricsDict, prefix: str = "") -> str:
    """Return a formatted one-line string of metrics."""
    parts = [f"{prefix}{k}={v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
