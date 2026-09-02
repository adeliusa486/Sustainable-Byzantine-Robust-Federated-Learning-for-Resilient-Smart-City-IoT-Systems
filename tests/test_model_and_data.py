"""
Unit Tests — Model, Attacks, and Data Pipeline
================================================
"""

import numpy as np
import pytest
import torch
from amfta.attacks.gaussian_noise import GaussianNoiseAttack
from amfta.attacks.label_flipping import LabelFlippingAttack
from amfta.data.partitioning import (
    assign_byzantine_clients,
    dirichlet_partition,
    generate_synthetic_data,
)
from amfta.models.local_mlp import LocalMLP, build_model


# ===========================================================================
# LocalMLP Tests
# ===========================================================================

class TestLocalMLP:
    def test_default_output_shape(self):
        model = LocalMLP()
        x = torch.randn(8, 45)
        out = model(x)
        assert out.shape == (8,)

    def test_output_in_zero_one(self):
        model = LocalMLP()
        x = torch.randn(32, 45)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_custom_dimensions(self):
        model = LocalMLP(input_dim=20, hidden1=16, hidden2=8)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4,)

    def test_num_parameters_positive(self):
        model = LocalMLP()
        assert model.num_parameters() > 0

    def test_flat_params_round_trip(self):
        model = LocalMLP()
        flat = model.get_flat_params()
        model2 = LocalMLP()
        model2.set_flat_params(flat)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1.data, p2.data)

    def test_apply_update(self):
        model = LocalMLP()
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        update = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        model.apply_update(update, lr=1.0)
        for k in sd_before:
            assert torch.allclose(model.state_dict()[k], sd_before[k])

    def test_config_dict_reconstruction(self):
        model = LocalMLP(input_dim=30, hidden1=48, hidden2=24)
        cfg = model.config_dict()
        model2 = LocalMLP.from_config(cfg)
        assert model2.input_dim == 30
        assert model2.hidden1 == 48

    def test_build_model_defaults(self):
        model = build_model()
        assert model.input_dim == 45

    def test_single_sample_inference(self):
        model = LocalMLP()
        x = torch.rand(1, 45)
        out = model(x)
        assert out.shape == (1,)

    def test_gradient_flows(self):
        model = LocalMLP()
        x = torch.rand(8, 45)
        y = torch.rand(8)
        loss = torch.nn.BCELoss()(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ===========================================================================
# Attack Tests
# ===========================================================================

class TestLabelFlippingAttack:
    def test_update_keys_match_model(self):
        model = LocalMLP()
        X = torch.rand(50, 45)
        y = torch.randint(0, 2, (50,)).float()
        attack = LabelFlippingAttack()
        update = attack.get_update(model, (X, y), epochs=1)
        assert set(update.keys()) == set(model.state_dict().keys())

    def test_does_not_mutate_global_model(self):
        model = LocalMLP()
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        X = torch.rand(20, 45)
        y = torch.zeros(20)
        attack = LabelFlippingAttack()
        attack.get_update(model, (X, y), epochs=1)
        for k in sd_before:
            assert torch.allclose(model.state_dict()[k], sd_before[k])

    def test_partial_flip_fraction(self):
        attack = LabelFlippingAttack(flip_fraction=0.5)
        model = LocalMLP()
        X = torch.rand(40, 45)
        y = torch.zeros(40)
        update = attack.get_update(model, (X, y), epochs=1)
        assert update is not None

    def test_invalid_flip_fraction(self):
        with pytest.raises(ValueError):
            LabelFlippingAttack(flip_fraction=0.0)
        with pytest.raises(ValueError):
            LabelFlippingAttack(flip_fraction=1.5)


class TestGaussianNoiseAttack:
    def test_update_is_noise(self):
        model = LocalMLP()
        X, y = torch.rand(20, 45), torch.rand(20)
        attack = GaussianNoiseAttack(sigma=1.0, seed=42)
        update = attack.get_update(model, (X, y))
        assert set(update.keys()) == set(model.state_dict().keys())

    def test_deterministic_with_seed(self):
        model = LocalMLP()
        X, y = torch.rand(20, 45), torch.rand(20)
        a1 = GaussianNoiseAttack(sigma=1.0, seed=7)
        a2 = GaussianNoiseAttack(sigma=1.0, seed=7)
        u1 = a1.get_update(model, (X, y))
        u2 = a2.get_update(model, (X, y))
        for k in u1:
            assert torch.allclose(u1[k], u2[k])

    def test_zero_sigma_gives_zero_update(self):
        model = LocalMLP()
        X, y = torch.rand(20, 45), torch.rand(20)
        attack = GaussianNoiseAttack(sigma=0.0, seed=0)
        update = attack.get_update(model, (X, y))
        for v in update.values():
            assert torch.allclose(v, torch.zeros_like(v))


# ===========================================================================
# Data Partitioning Tests
# ===========================================================================

class TestDataPartitioning:
    def test_generate_synthetic(self):
        X, y = generate_synthetic_data(n_samples=1000, n_features=45, seed=42)
        assert X.shape == (1000, 45)
        assert y.shape == (1000,)
        assert set(np.unique(y)) == {0, 1}

    def test_synthetic_feature_range(self):
        X, _ = generate_synthetic_data(n_samples=500, n_features=10)
        assert (X >= 0).all() and (X <= 1).all()

    def test_dirichlet_partition_num_clients(self):
        X, y = generate_synthetic_data(1000)
        partitions = dirichlet_partition(X, y, num_clients=10, alpha=0.5, seed=42)
        assert len(partitions) == 10

    def test_dirichlet_all_data_assigned(self):
        X, y = generate_synthetic_data(1000)
        partitions = dirichlet_partition(X, y, num_clients=10, alpha=0.5, seed=42)
        for cid, (Xc, yc) in partitions.items():
            assert len(Xc) > 0, f"Client {cid} has 0 samples"

    def test_dirichlet_min_samples(self):
        X, y = generate_synthetic_data(500)
        partitions = dirichlet_partition(
            X, y, num_clients=5, alpha=0.5, seed=0,
            min_samples_per_client=10,
        )
        for cid, (Xc, _) in partitions.items():
            assert len(Xc) >= 10, f"Client {cid} has fewer than min_samples"

    def test_dirichlet_reproducibility(self):
        X, y = generate_synthetic_data(500)
        p1 = dirichlet_partition(X, y, num_clients=5, seed=42)
        p2 = dirichlet_partition(X, y, num_clients=5, seed=42)
        for cid in p1:
            assert np.array_equal(p1[cid][1], p2[cid][1])

    def test_assign_byzantine_clients(self):
        byz = assign_byzantine_clients(100, 0.30, seed=42)
        assert len(byz) == 30
        assert all(0 <= cid < 100 for cid in byz)

    def test_assign_byzantine_zero_rate(self):
        byz = assign_byzantine_clients(100, 0.0, seed=42)
        assert len(byz) == 0
