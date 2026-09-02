"""
Federated Learning Training Orchestrator
==========================================

Implements the complete FL training loop for AMFTA and all baseline methods.
Simulates 100 IoT clients on a single machine (single-GPU or CPU).

Architecture:
  - Server-side: aggregator receives gradient deltas, applies trust engine,
    updates global model, broadcasts back.
  - Client-side: each client deep-copies global model, trains locally for
    E=5 epochs on its private partition, returns gradient delta.
  - Byzantine clients: replaced by attack strategy (label flipping / noise).

Usage:
    from training.federated_runner import FederatedRunner, RunConfig
    config = RunConfig(method="amfta", byzantine_fraction=0.30, seed=42)
    runner = FederatedRunner(config)
    history = runner.run()
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amfta.aggregation.amfta import AMFTAAggregator
from amfta.aggregation.amfta_s import (
    AMFTASAggregator, ResourceProfile, compute_scores, schedule,
)
from amfta.aggregation.baselines import (
    FedAvgAggregator,
    FedDBCAggregator,
    FLTrustAggregator,
    KrumAggregator,
    TrimmedMeanAggregator,
    build_aggregator,
)
from amfta.attacks.base import BaseAttack, get_attack
from amfta.data.partitioning import (
    assign_byzantine_clients,
    generate_synthetic_data,
    dirichlet_partition,
)
from amfta.models.local_mlp import LocalMLP
from amfta.models.logistic import build_global_model
import amfta.aggregation.extra_baselines  # noqa: F401  (registers rules)
import amfta.attacks.adaptive  # noqa: F401  (registers 'adaptive')
from amfta.aggregation.baselines import build_aggregator as _build_by_name
from amfta.utils.logging_utils import ExperimentLogger
from amfta.utils.metrics import evaluate_model, evaluate_model_fast
from amfta.utils.reproducibility import set_seed, get_device

logger = logging.getLogger(__name__)

UpdateDict = Dict[int, Dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for a single FL experiment run."""

    # Method
    method: str = "amfta"             # 'amfta', 'fedavg', 'krum', 'trimmed_mean', 'fltrust'

    # FL parameters
    num_clients: int = 100
    num_rounds: int = 100
    local_epochs: int = 5
    local_lr: float = 0.01
    local_batch_size: int = 2048
    global_lr: float = 1.0            # η_global — scales aggregated update

    # Byzantine parameters
    byzantine_fraction: float = 0.30
    attack_type: str = "label_flipping"  # 'label_flipping', 'gaussian_noise', 'none'

    # AMFTA-specific
    alpha_s: float = 0.4
    alpha_h: float = 0.3
    alpha_q: float = 0.3
    beta: float = 0.9
    tau_lower: float = 0.35
    tau_upper: float = 0.55

    # FLTrust-specific
    fltrust_root_size: int = 200

    # Krum-specific
    krum_multi: bool = False

    # TrimmedMean-specific
    trim_fraction: float = 0.10

    # Data
    seed: int = 42
    alpha_dirichlet: float = 0.5
    partition_dir: str = "data/partitions"
    processed_dir: str = "data/processed"
    server_dir: str = "data/server"
    use_synthetic: bool = False        # Use synthetic data (for testing without TON_IoT)
    n_synthetic: int = 50_000

    # Model
    repartition: bool = False  # regenerate the split at alpha_dirichlet
    p_min: float = 0.25          # participation floor for AMFTA-S
    sched_gamma: float = 1.0     # participation exponent
    spoof_fraction: float = 0.0  # share of Byzantine clients understating cost
    spoof_factor: float = 0.05   # how far they understate it
    model_class: str = "mlp"   # 'logistic' reproduces the manuscript
    model_input_dim: int = 41
    model_hidden1: int = 64
    model_hidden2: int = 32

    # Logging
    results_dir: str = "results"
    log_interval: int = 10
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"

    # Ablation
    disable_factor_h: bool = False     # AMFTA-S ablation
    disable_factor_q: bool = False     # AMFTA-SH ablation


# ---------------------------------------------------------------------------
# Client local training
# ---------------------------------------------------------------------------

def local_train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Train a copy of the model on local data and return the gradient delta.

    Parameters
    ----------
    model  : nn.Module     Current global model (deep-copied internally).
    X, y   : tensors       Local dataset.
    epochs : int           Local training epochs.
    lr     : float         SGD learning rate.
    batch_size : int       Mini-batch size.
    device : torch.device  Compute device.

    Returns
    -------
    dict {param_name: delta_tensor}  Update = w_trained − w_received.
    """
    model = copy.deepcopy(model).to(device)
    w_before = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss()

    model.train()
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

    update = {k: model.state_dict()[k] - w_before[k] for k in w_before}
    return update


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class FederatedRunner:
    """Orchestrates a complete federated learning experiment.

    Parameters
    ----------
    config : RunConfig  Experiment configuration.
    """

    def __init__(self, config: RunConfig) -> None:
        self.cfg = config
        self.device = get_device(prefer_gpu=True)
        set_seed(config.seed)

        # Build global model
        self.global_model = build_global_model(
            config.model_class,
            input_dim=config.model_input_dim,
            hidden1=config.model_hidden1,
            hidden2=config.model_hidden2,
        ).to(self.device)
        logger.info(
            "Global model: %r (d=%d)",
            self.global_model, self.global_model.num_parameters(),
        )

        # Build aggregator
        self.aggregator = self._build_aggregator()

        # Load data
        self._load_data()

        # Resource-aware schedule (AMFTA-S only). The sustainability layer
        # sees client resource reports and nothing else; it never sees trust.
        import numpy as _np
        self._rng = _np.random.default_rng(config.seed + 9973)
        self._sampled_this_round = []
        self.participation = {}
        self.local_epochs_per_client = {}
        if config.method == "amfta_s":
            n_samples = {cid: len(y) for cid, (_, y) in self.client_tensors.items()}
            self.resource_profile = ResourceProfile(config.num_clients, seed=config.seed)
            misreport = None
            if config.spoof_fraction > 0:
                byz = sorted(assign_byzantine_clients(
                    config.num_clients, config.byzantine_fraction, config.seed))
                k = int(round(config.spoof_fraction * len(byz)))
                misreport = {c: config.spoof_factor for c in byz[:k]}
            scores = compute_scores(
                n_samples, self.resource_profile,
                d=self.global_model.num_parameters(),
                epochs=config.local_epochs, batch=config.local_batch_size,
                misreport=misreport)
            self.participation, self.local_epochs_per_client = schedule(
                scores, p_min=config.p_min, gamma=config.sched_gamma,
                base_epochs=config.local_epochs)
            logger.info("AMFTA-S schedule: mean p=%.3f, mean local epochs=%.2f",
                        sum(self.participation.values()) / len(self.participation),
                        sum(self.local_epochs_per_client.values()) / len(self.local_epochs_per_client))

        # Assign Byzantine clients
        self.byzantine_ids: Set[int] = assign_byzantine_clients(
            config.num_clients, config.byzantine_fraction, config.seed
        )

        # Build attack strategy
        self.attack: BaseAttack = get_attack(config.attack_type)

        logger.info("FederatedRunner initialised: %s", self)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load/generate training partitions, test set, and val buffer."""
        cfg = self.cfg

        if cfg.use_synthetic:
            logger.warning(
                "Using SYNTHETIC data. Results will not match paper figures. "
                "Download TON_IoT and remove --use_synthetic for real experiments."
            )
            X_all, y_all = generate_synthetic_data(
                n_samples=cfg.n_synthetic,
                n_features=cfg.model_input_dim,
                seed=cfg.seed,
            )
            # 70/10/20 split
            n = len(X_all)
            n_train = int(0.70 * n)
            n_val = int(0.10 * n)
            X_train, y_train = X_all[:n_train], y_all[:n_train]
            X_val   = X_all[n_train : n_train + n_val]
            y_val   = y_all[n_train : n_train + n_val]
            X_test  = X_all[n_train + n_val :]
            y_test  = y_all[n_train + n_val :]

            # Server val buffer
            buf_idx = np.random.choice(len(X_val), size=min(500, len(X_val)), replace=False)
            X_buf, y_buf = X_val[buf_idx], y_val[buf_idx]

            # Partition training data
            self.client_data = dirichlet_partition(
                X_train, y_train,
                num_clients=cfg.num_clients,
                alpha=cfg.alpha_dirichlet,
                seed=cfg.seed,
            )
        else:
            from amfta.data.partitioning import load_partitions
            from amfta.data.preprocessing import load_processed

            data = load_processed(
                processed_dir=Path(cfg.processed_dir),
                server_dir=Path(cfg.server_dir),
            )
            X_test, y_test = data["X_test"], data["y_test"]
            X_buf, y_buf = data["X_val_buffer"], data["y_val_buffer"]

            if cfg.repartition:
                # Regenerate the client split at the requested concentration.
                # The stored partitions under data/partitions were generated at
                # alpha=0.5 only, so any other heterogeneity level must be
                # produced here.
                logger.info(
                    "Repartitioning %d clients at Dirichlet alpha=%.3f (seed %d)",
                    cfg.num_clients, cfg.alpha_dirichlet, cfg.seed,
                )
                self.client_data = dirichlet_partition(
                    data["X_train"], data["y_train"],
                    num_clients=cfg.num_clients,
                    alpha=cfg.alpha_dirichlet,
                    seed=cfg.seed,
                )
            else:
                self.client_data = load_partitions(
                    partition_dir=Path(cfg.partition_dir),
                    seed=cfg.seed,
                    num_clients=cfg.num_clients,
                )

        # Convert to tensors
        self.X_test = torch.from_numpy(X_test.astype(np.float32))
        self.y_test = torch.from_numpy(y_test.astype(np.float32))
        self.val_buffer = (
            torch.from_numpy(X_buf.astype(np.float32)).to(self.device),
            torch.from_numpy(y_buf.astype(np.float32)).to(self.device),
        )

        # Convert client data to tensors
        self.client_tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for cid, (Xc, yc) in self.client_data.items():
            self.client_tensors[cid] = (
                torch.from_numpy(Xc.astype(np.float32)).to(self.device),
                torch.from_numpy(yc.astype(np.float32)).to(self.device),
            )

        logger.info(
            "Data loaded — test: %d samples, val_buffer: %d samples, clients: %d",
            len(self.X_test), len(self.val_buffer[0]), len(self.client_tensors),
        )

    # ------------------------------------------------------------------
    # Aggregator factory
    # ------------------------------------------------------------------

    def _build_aggregator(self):
        cfg = self.cfg
        method = cfg.method.lower()

        if method == "amfta_s":
            return AMFTASAggregator(num_clients=cfg.num_clients,
                                    beta=cfg.beta, p_min=cfg.p_min)

        if method in ("amfta", "amfta_noq"):
            # Handle ablation variants. 'amfta_noq' is the no-trusted-data
            # ablation: Factor III (server validation buffer) is disabled, so
            # the method uses only gradient similarity + reputation. This
            # answers the "why does AMFTA need trusted validation data?" review
            # question by showing performance without any trusted data.
            use_q = (method != "amfta_noq") and (not cfg.disable_factor_q)
            agg = AMFTAAggregator(
                num_clients=cfg.num_clients,
                alpha_s=cfg.alpha_s,
                alpha_h=cfg.alpha_h,
                alpha_q=cfg.alpha_q,
                beta=cfg.beta,
                tau_lower=cfg.tau_lower,
                tau_upper=cfg.tau_upper,
                use_quality_eval=use_q,
            )
            if cfg.disable_factor_h:
                # AMFTA-S: freeze reputation to neutral 0.5
                agg.reputation.beta = 0.0  # H_i(t) = 0.5 always
            return agg

        elif method == "fedavg":
            return FedAvgAggregator()

        elif method == "krum":
            f = int(cfg.num_clients * cfg.byzantine_fraction)
            return KrumAggregator(num_byzantine=f, multi_krum=cfg.krum_multi)

        elif method == "trimmed_mean":
            return TrimmedMeanAggregator(trim_fraction=cfg.trim_fraction)

        elif method == "feddbc":
            return FedDBCAggregator(trim_fraction=cfg.trim_fraction)

        elif method == "fltrust":
            return FLTrustAggregator(root_dataset_size=cfg.fltrust_root_size)

        elif method in ("normclip", "normclip_only", "median", "foolsgold"):
            return _build_by_name(method)

        elif method == "multikrum":
            f = int(cfg.num_clients * cfg.byzantine_fraction)
            return _build_by_name("multikrum", {"num_byzantine": f})

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # FLTrust root update generation
    # ------------------------------------------------------------------

    def _compute_root_update(self) -> Dict[str, torch.Tensor]:
        """Generate FLTrust root update from server-side validation buffer."""
        X_root, y_root = self.val_buffer
        # Use only fltrust_root_size samples
        n = min(self.cfg.fltrust_root_size, len(X_root))
        root_update = local_train(
            self.global_model,
            X_root[:n], y_root[:n],
            epochs=self.cfg.local_epochs,
            lr=self.cfg.local_lr,
            device=self.device,
        )
        return root_update

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Execute the full federated training loop.

        Returns
        -------
        List of per-round metric dicts.
        """
        cfg = self.cfg
        exp_name = f"{cfg.method}_byz{cfg.byzantine_fraction}_{cfg.attack_type}_seed{cfg.seed}"
        history: List[Dict[str, Any]] = []

        with ExperimentLogger(exp_name, results_dir=Path(cfg.results_dir)) as exp_log:
            for round_t in range(1, cfg.num_rounds + 1):

                # ── Phase A: Client local training ─────────────────────────
                updates: UpdateDict = {}

                # Under AMFTA-S the sustainability layer decides who is asked
                # to work this round and how much work each does. Every other
                # method trains the full cohort for a fixed number of epochs.
                if self.participation:
                    invited = [c for c in range(cfg.num_clients)
                               if self._rng.random() < self.participation.get(c, 1.0)]
                    if not invited:                      # never skip a whole round
                        invited = [max(self.participation,
                                       key=self.participation.get)]
                    self._sampled_this_round = invited
                else:
                    self._sampled_this_round = list(range(cfg.num_clients))

                for cid in self._sampled_this_round:
                    X_c, y_c = self.client_tensors[cid]
                    epochs = self.local_epochs_per_client.get(cid, cfg.local_epochs)

                    if cid in self.byzantine_ids:
                        update = self.attack.get_update(
                            self.global_model, (X_c, y_c),
                            epochs=epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                        )
                    else:
                        update = local_train(
                            self.global_model, X_c, y_c,
                            epochs=epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                            device=self.device,
                        )

                    updates[cid] = update

                if self.participation:
                    # the aggregator needs the assigned probabilities, not the
                    # realised draw, to weight by 1/p_i
                    self.aggregator.participation = {
                        c: self.participation[c] for c in self._sampled_this_round}

                # ── Phase A2: coalition-level attacks ──────────────────────
                # The adaptive attack is defined over the whole coalition: it
                # needs the round's honest updates to choose a direction and
                # to scale the perturbation. Per-client attacks are already
                # applied in Phase A.
                if cfg.attack_type == "adaptive" and self.byzantine_ids:
                    from amfta.attacks.adaptive import AdaptiveAGRAttack
                    honest = {c: u for c, u in updates.items()
                              if c not in self.byzantine_ids}
                    if honest:
                        crafter = self.attack if isinstance(
                            self.attack, AdaptiveAGRAttack) else AdaptiveAGRAttack()
                        updates.update(
                            crafter.craft_coalition(
                                self.global_model, honest,
                                sorted(self.byzantine_ids),
                            )
                        )

                # ── Phase B: Aggregation ───────────────────────────────────
                if cfg.method == "fltrust":
                    root_update = self._compute_root_update()
                    agg_update = self.aggregator.aggregate(
                        self.global_model, updates, root_update=root_update
                    )
                elif cfg.method in ("amfta", "amfta_noq"):
                    # amfta_noq passes val_buffer but never reads it
                    # (use_quality_eval=False) — the no-trusted-data ablation.
                    vb = None if cfg.method == "amfta_noq" else self.val_buffer
                    agg_update = self.aggregator.aggregate(
                        self.global_model, updates, val_buffer=vb
                    )
                else:
                    agg_update = self.aggregator.aggregate(self.global_model, updates)

                # ── Phase C: Global model update ───────────────────────────
                self.global_model.apply_update(agg_update, lr=cfg.global_lr)

                # ── Phase D: Evaluation ────────────────────────────────────
                metrics = evaluate_model_fast(
                    self.global_model, self.X_test, self.y_test, device=self.device
                )

                # Collect trust diagnostics (AMFTA only)
                diagnostics = {}
                if cfg.method in ("amfta", "amfta_noq") and hasattr(self.aggregator, "_diagnostics"):
                    if self.aggregator._diagnostics:
                        last_diag = self.aggregator._diagnostics[-1]
                        diagnostics = {
                            "num_benign":    last_diag.num_benign,
                            "num_borderline": last_diag.num_borderline,
                            "num_suspect":   last_diag.num_suspect,
                            "mean_trust":    last_diag.mean_trust,
                        }

                row = {"round": round_t, **metrics, **diagnostics}
                history.append(row)
                exp_log.log_round(round_t, row)

                if round_t % cfg.log_interval == 0:
                    logger.info(
                        "[Round %3d/%d] acc=%.4f f1=%.4f %s",
                        round_t, cfg.num_rounds,
                        metrics["accuracy"], metrics["f1"],
                        " ".join(f"{k}={v:.1f}" for k, v in diagnostics.items()),
                    )

                if cfg.save_checkpoints and round_t % 20 == 0:
                    self._save_checkpoint(round_t)

        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, round_t: int) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = ckpt_dir / f"{self.cfg.method}_round{round_t:04d}.pt"
        torch.save({
            "round": round_t,
            "model_state": self.global_model.state_dict(),
            "config": vars(self.cfg),
        }, path)
        logger.info("Checkpoint saved: %s", path)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FederatedRunner("
            f"method={self.cfg.method}, "
            f"n_clients={self.cfg.num_clients}, "
            f"byz={self.cfg.byzantine_fraction:.0%}, "
            f"attack={self.cfg.attack_type}, "
            f"seed={self.cfg.seed})"
        )
