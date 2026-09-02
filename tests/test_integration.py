"""
Integration Tests — Full FL Round Simulation
=============================================

Verifies that a complete federated learning round (client training →
aggregation → global model update → evaluation) runs without errors
for all methods using synthetic data.

These tests are slower than unit tests (~5-30 seconds per test).
Run with: pytest tests/test_integration.py -v
"""

import pytest
import torch
import numpy as np

from amfta.aggregation.amfta import AMFTAAggregator
from amfta.aggregation.baselines import (
    FedAvgAggregator, KrumAggregator,
    TrimmedMeanAggregator, FLTrustAggregator,
)
from amfta.attacks.label_flipping import LabelFlippingAttack
from amfta.attacks.gaussian_noise import GaussianNoiseAttack
from amfta.data.partitioning import (
    dirichlet_partition, generate_synthetic_data, assign_byzantine_clients
)
from amfta.models.local_mlp import LocalMLP
from amfta.utils.metrics import evaluate_model
from amfta.utils.reproducibility import set_seed
from training.federated_runner import local_train


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_CLIENTS   = 10
N_SAMPLES   = 2000
N_FEATURES  = 45
N_ROUNDS    = 3
BYZ_FRAC    = 0.30


@pytest.fixture(scope="module")
def synthetic_data():
    set_seed(42)
    X, y = generate_synthetic_data(N_SAMPLES, N_FEATURES, seed=42)
    return X, y


@pytest.fixture(scope="module")
def client_partitions(synthetic_data):
    X, y = synthetic_data
    n = int(0.8 * len(X))
    return dirichlet_partition(X[:n], y[:n], num_clients=N_CLIENTS, alpha=0.5, seed=42)


@pytest.fixture(scope="module")
def val_buffer(synthetic_data):
    X, y = synthetic_data
    n = int(0.8 * len(X))
    X_val, y_val = X[n:], y[n:]
    return (
        torch.from_numpy(X_val[:100].astype(np.float32)),
        torch.from_numpy(y_val[:100].astype(np.float32)),
    )


@pytest.fixture(scope="module")
def test_data(synthetic_data):
    X, y = synthetic_data
    n = int(0.8 * len(X))
    return (
        torch.from_numpy(X[n:].astype(np.float32)),
        torch.from_numpy(y[n:].astype(np.float32)),
    )


@pytest.fixture(scope="module")
def byzantine_ids():
    return assign_byzantine_clients(N_CLIENTS, BYZ_FRAC, seed=42)


# ---------------------------------------------------------------------------
# Helper: run N rounds and return metrics
# ---------------------------------------------------------------------------

def run_fl_rounds(
    aggregator, client_partitions, byzantine_ids,
    val_buffer, test_data, n_rounds=N_ROUNDS,
    attack=None,
):
    set_seed(0)
    global_model = LocalMLP(input_dim=N_FEATURES)
    attack = attack or LabelFlippingAttack()
    device = torch.device("cpu")

    # Convert partitions to tensors once
    client_tensors = {
        cid: (
            torch.from_numpy(Xc.astype(np.float32)),
            torch.from_numpy(yc.astype(np.float32)),
        )
        for cid, (Xc, yc) in client_partitions.items()
    }

    history = []
    for round_t in range(1, n_rounds + 1):
        updates = {}
        for cid in range(N_CLIENTS):
            X_c, y_c = client_tensors[cid]
            if cid in byzantine_ids:
                updates[cid] = attack.get_update(global_model, (X_c, y_c), epochs=1)
            else:
                updates[cid] = local_train(global_model, X_c, y_c, epochs=1,
                                           lr=0.01, device=device)

        # Aggregation
        if isinstance(aggregator, AMFTAAggregator):
            agg_update = aggregator.aggregate(global_model, updates, val_buffer=val_buffer)
        elif isinstance(aggregator, FLTrustAggregator):
            root_upd = local_train(
                global_model, val_buffer[0][:50], val_buffer[1][:50], epochs=1,
                lr=0.01, device=device,
            )
            agg_update = aggregator.aggregate(global_model, updates, root_update=root_upd)
        else:
            agg_update = aggregator.aggregate(global_model, updates)

        global_model.apply_update(agg_update)
        X_test, y_test = test_data
        metrics = evaluate_model(global_model, X_test, y_test)
        history.append(metrics)

    return history, global_model


# ---------------------------------------------------------------------------
# Integration tests per method
# ---------------------------------------------------------------------------

class TestFedAvgIntegration:
    def test_runs_without_error(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = FedAvgAggregator()
        history, model = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        assert len(history) == N_ROUNDS

    def test_model_params_finite(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = FedAvgAggregator()
        _, model = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        for p in model.parameters():
            assert torch.isfinite(p).all()


class TestAMFTAIntegration:
    def test_runs_without_error(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = AMFTAAggregator(num_clients=N_CLIENTS)
        history, model = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        assert len(history) == N_ROUNDS

    def test_diagnostics_populated_per_round(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = AMFTAAggregator(num_clients=N_CLIENTS)
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        diags = agg.get_diagnostics()
        assert len(diags) == N_ROUNDS
        for d in diags:
            assert d["num_benign"] + d["num_borderline"] + d["num_suspect"] == N_CLIENTS

    def test_accuracy_non_trivial(self, client_partitions, byzantine_ids, val_buffer, test_data):
        """After FL training (even a few rounds), accuracy should be > 50%."""
        agg = AMFTAAggregator(num_clients=N_CLIENTS)
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data, n_rounds=5
        )
        assert history[-1]["accuracy"] > 0.5, (
            f"Expected accuracy > 0.5 after training; got {history[-1]['accuracy']:.4f}"
        )


class TestKrumIntegration:
    def test_runs_without_error(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = KrumAggregator(num_byzantine=int(N_CLIENTS * BYZ_FRAC))
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        assert len(history) == N_ROUNDS


class TestTrimmedMeanIntegration:
    def test_runs_without_error(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = TrimmedMeanAggregator(trim_fraction=0.1)
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        assert len(history) == N_ROUNDS


class TestFLTrustIntegration:
    def test_runs_without_error(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = FLTrustAggregator()
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data
        )
        assert len(history) == N_ROUNDS


class TestAttackIntegration:
    def test_gaussian_noise_attack(self, client_partitions, byzantine_ids, val_buffer, test_data):
        agg = AMFTAAggregator(num_clients=N_CLIENTS)
        attack = GaussianNoiseAttack(sigma=0.1)
        history, _ = run_fl_rounds(
            agg, client_partitions, byzantine_ids, val_buffer, test_data, attack=attack
        )
        assert len(history) == N_ROUNDS
