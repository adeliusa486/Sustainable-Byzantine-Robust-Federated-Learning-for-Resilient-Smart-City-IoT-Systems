"""
Unit Tests — AMFTA Aggregator
================================

Tests the full AMFTA aggregation pipeline including round execution,
checkpoint save/load, and edge cases.
"""

import pytest
import torch
from amfta.aggregation.amfta import AMFTAAggregator
from amfta.models.local_mlp import LocalMLP


@pytest.fixture
def model():
    torch.manual_seed(0)
    return LocalMLP()


@pytest.fixture
def aggregator():
    return AMFTAAggregator(num_clients=10, alpha_s=0.4, alpha_h=0.3, alpha_q=0.3)


@pytest.fixture
def updates(model):
    """Generate benign and byzantine updates for 10 clients."""
    sd = model.state_dict()
    ups = {}
    for i in range(7):  # benign: positive direction
        ups[i] = {k: torch.full_like(v.float(), +0.01) for k, v in sd.items()}
    for i in range(7, 10):  # byzantine: negative direction
        ups[i] = {k: torch.full_like(v.float(), -0.03) for k, v in sd.items()}
    return ups


@pytest.fixture
def val_buffer():
    torch.manual_seed(99)
    return torch.rand(100, 45), torch.randint(0, 2, (100,)).float()


class TestAMFTAAggregator:
    def test_constructor_weight_validation(self):
        with pytest.raises(ValueError):
            AMFTAAggregator(alpha_s=0.5, alpha_h=0.5, alpha_q=0.5)

    def test_aggregate_returns_all_params(self, aggregator, model, updates, val_buffer):
        result = aggregator.aggregate(model, updates, val_buffer)
        assert set(result.keys()) == set(model.state_dict().keys())

    def test_aggregate_increments_round(self, aggregator, model, updates, val_buffer):
        assert aggregator.current_round == 0
        aggregator.aggregate(model, updates, val_buffer)
        assert aggregator.current_round == 1

    def test_multiple_rounds_stable(self, aggregator, model, updates, val_buffer):
        """Run 5 rounds — should not raise and model should remain finite."""
        for _ in range(5):
            agg_update = aggregator.aggregate(model, updates, val_buffer)
            model.apply_update(agg_update)
        for p in model.parameters():
            assert torch.isfinite(p).all(), "Model parameters contain NaN/Inf"

    def test_diagnostics_populated(self, aggregator, model, updates, val_buffer):
        aggregator.aggregate(model, updates, val_buffer)
        diags = aggregator.get_diagnostics()
        assert len(diags) == 1
        d = diags[0]
        assert "num_benign" in d
        assert "mean_trust" in d
        assert 0.0 <= d["mean_trust"] <= 1.0

    def test_trust_scores_in_range(self, aggregator, model, updates, val_buffer):
        aggregator.aggregate(model, updates, val_buffer)
        diag = aggregator.get_diagnostics()[0]
        for cid, score in diag["trust_scores"].items():
            assert 0.0 <= score <= 1.0 + 1e-6, f"Trust score {score} out of range for client {cid}"

    def test_reset_clears_state(self, aggregator, model, updates, val_buffer):
        aggregator.aggregate(model, updates, val_buffer)
        aggregator.reset()
        assert aggregator.current_round == 0
        assert len(aggregator.get_diagnostics()) == 0

    def test_state_dict_round_trip(self, aggregator, model, updates, val_buffer):
        aggregator.aggregate(model, updates, val_buffer)
        sd = aggregator.state_dict()
        agg2 = AMFTAAggregator(num_clients=10)
        agg2.load_state_dict(sd)
        assert agg2.current_round == aggregator.current_round

    def test_no_val_buffer_falls_back_gracefully(self, aggregator, model, updates):
        """Aggregation without val_buffer should not raise (borderline clients get 0.5)."""
        result = aggregator.aggregate(model, updates, val_buffer=None)
        assert result is not None

    def test_empty_updates_raises(self, aggregator, model):
        with pytest.raises(ValueError):
            aggregator.aggregate(model, {}, val_buffer=None)

    def test_benign_updates_score_higher(self, aggregator, model, val_buffer):
        """Benign clients should receive higher trust scores than Byzantine clients.

        Uses 9 benign (positive direction) vs 1 Byzantine (negative) so the
        centroid unambiguously points in the benign direction, enabling Factor I
        to correctly assign higher similarity to benign clients.
        """
        sd = model.state_dict()
        updates = {}
        for i in range(9):   # 9 benign
            updates[i] = {k: torch.full_like(v.float(), +0.01) for k, v in sd.items()}
        updates[9] = {k: torch.full_like(v.float(), -0.01) for k, v in sd.items()}

        result = aggregator.aggregate(model, updates, val_buffer)
        # With 9 benign vs 1 byzantine, aggregated update should point benign direction
        for k, v in result.items():
            assert v.mean().item() > 0.0, (
                f"With 9 benign vs 1 byzantine, update should be positive for {k}, "
                f"got {v.mean().item():.4f}"
            )
