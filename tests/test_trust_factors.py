"""
Unit Tests — Trust Factor Computation
========================================

Tests for gradient similarity, reputation tracking, quality evaluation,
and final trust score combination.
"""

import pytest
import torch
import numpy as np
from amfta.aggregation.trust_factors import (
    ReputationTracker,
    combine_trust_scores,
    compute_contribution_quality,
    compute_gradient_similarity,
    compute_preliminary_scores,
    identify_borderline_clients,
)
from amfta.models.local_mlp import LocalMLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_updates():
    """Four clients: 2 similar (benign), 2 opposite (byzantine)."""
    torch.manual_seed(0)
    model = LocalMLP()
    sd = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())

    def _make_update(direction: float) -> dict:
        return {
            k: torch.full_like(v.float(), direction * 0.01)
            for k, v in sd.items()
        }

    return {
        0: _make_update(+1.0),  # benign
        1: _make_update(+1.0),  # benign
        2: _make_update(-1.0),  # byzantine
        3: _make_update(-1.0),  # byzantine
    }


@pytest.fixture
def val_buffer():
    torch.manual_seed(42)
    X = torch.rand(100, 45)
    y = torch.randint(0, 2, (100,)).float()
    return X, y


# ---------------------------------------------------------------------------
# Factor I — Gradient Similarity
# ---------------------------------------------------------------------------

class TestGradientSimilarity:
    def test_output_shape(self, simple_updates):
        scores = compute_gradient_similarity(simple_updates)
        assert set(scores.keys()) == set(simple_updates.keys())

    def test_output_range(self, simple_updates):
        scores = compute_gradient_similarity(simple_updates)
        for v in scores.values():
            assert 0.0 <= v <= 1.0, f"Score {v} out of [0, 1]"

    def test_symmetric_updates(self):
        """Identical updates should all score ~0.5 (centroid = each update)."""
        upd = {
            i: {"w": torch.ones(10) * 0.01}
            for i in range(4)
        }
        scores = compute_gradient_similarity(upd)
        for s in scores.values():
            assert abs(s - 1.0) < 1e-5, f"Identical updates should give cosine=1 → score=1"

    def test_opposite_updates_low_score(self):
        """When benign clients dominate centroid magnitude, they score higher than Byzantine.

        Note: When Byzantine updates have much larger magnitude than benign and/or
        Byzantine fraction is >=30%, the centroid can tilt toward Byzantine direction,
        causing benign clients to appear dissimilar.  This is a documented limitation
        of Factor I alone, compensated by Factor II (historical reputation).

        This test uses equal-magnitude updates where benign clients (9) far
        outnumber Byzantine (1) to ensure the centroid points in the benign direction.
        """
        torch.manual_seed(42)
        model = LocalMLP()
        sd = model.state_dict()
        # 9 benign (positive direction) + 1 byzantine (negative, same magnitude)
        # Centroid: (9*0.01 + 1*(-0.01)) / 10 = 0.008 → benign direction
        updates = {}
        for i in range(9):   # benign
            updates[i] = {k: torch.full_like(v.float(), +0.01) for k, v in sd.items()}
        updates[9] = {k: torch.full_like(v.float(), -0.01) for k, v in sd.items()}

        scores = compute_gradient_similarity(updates)
        benign_avg = sum(scores[i] for i in range(9)) / 9
        byz_score  = scores[9]
        assert benign_avg > byz_score, (
            f"Expected benign_avg ({benign_avg:.3f}) > byz_score ({byz_score:.3f}). "
            "When benign clients dominate centroid, Byzantine should score lower."
        )

    def test_empty_updates_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_similarity({})


# ---------------------------------------------------------------------------
# Factor II — Reputation Tracker
# ---------------------------------------------------------------------------

class TestReputationTracker:
    def test_init_value(self):
        tracker = ReputationTracker()
        assert tracker.get(0) == tracker.init_value

    def test_ema_update(self):
        tracker = ReputationTracker(beta=0.9, init_value=0.5)
        sim = {0: 1.0}
        scores = tracker.update(sim)
        # H(1) = 0.9*0.5 + 0.1*1.0 = 0.55
        assert abs(scores[0] - 0.55) < 1e-6

    def test_ema_decays_to_sim(self):
        """After many rounds with constant similarity, history → similarity."""
        tracker = ReputationTracker(beta=0.9)
        for _ in range(200):
            tracker.update({0: 0.8})
        final = tracker.get(0)
        assert abs(final - 0.8) < 0.01, f"Expected ~0.8 after convergence, got {final:.4f}"

    def test_reset_clears_history(self):
        tracker = ReputationTracker()
        tracker.update({0: 1.0})
        tracker.reset()
        assert tracker.get(0) == tracker.init_value
        assert len(tracker._history) == 0

    def test_state_dict_round_trip(self):
        tracker = ReputationTracker(beta=0.8)
        tracker.update({0: 0.9, 1: 0.3})
        sd = tracker.state_dict()
        tracker2 = ReputationTracker()
        tracker2.load_state_dict(sd)
        assert abs(tracker2.get(0) - tracker.get(0)) < 1e-8

    def test_beta_validation(self):
        with pytest.raises(ValueError):
            ReputationTracker(beta=1.0)
        with pytest.raises(ValueError):
            ReputationTracker(beta=-0.1)

    def test_effective_memory(self):
        tracker = ReputationTracker(beta=0.9)
        assert abs(tracker.effective_memory_rounds - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Borderline Detection
# ---------------------------------------------------------------------------

class TestBorderlineDetection:
    def test_all_high_are_benign(self):
        prelim = {0: 0.9, 1: 0.8, 2: 0.7}
        benign, borderline, suspect = identify_borderline_clients(prelim, 0.35, 0.55)
        assert {0, 1, 2} == benign
        assert len(borderline) == 0
        assert len(suspect) == 0

    def test_all_low_are_suspect(self):
        prelim = {0: 0.1, 1: 0.2}
        benign, borderline, suspect = identify_borderline_clients(prelim, 0.35, 0.55)
        assert {0, 1} == suspect
        assert len(benign) == 0

    def test_mixed_classification(self):
        prelim = {0: 0.9, 1: 0.45, 2: 0.1}
        benign, borderline, suspect = identify_borderline_clients(prelim, 0.35, 0.55)
        assert 0 in benign
        assert 1 in borderline
        assert 2 in suspect

    def test_disjoint_partition(self):
        prelim = {i: i / 10 for i in range(10)}
        benign, borderline, suspect = identify_borderline_clients(prelim, 0.35, 0.55)
        all_assigned = benign | borderline | suspect
        assert all_assigned == set(prelim.keys()), "All clients must be assigned"
        assert len(benign & borderline) == 0
        assert len(benign & suspect) == 0
        assert len(borderline & suspect) == 0


# ---------------------------------------------------------------------------
# Factor III — Contribution Quality
# ---------------------------------------------------------------------------

class TestContributionQuality:
    def test_borderline_gets_evaluated(self, val_buffer):
        model = LocalMLP()
        # Only client 1 is borderline
        updates = {
            0: {k: v * 0.0 for k, v in model.state_dict().items()},  # zero update
            1: {k: v * 0.0 for k, v in model.state_dict().items()},  # zero update
        }
        quality = compute_contribution_quality(
            model, updates, val_buffer,
            borderline_ids={1}, suspect_ids=set(),
        )
        assert 0 in quality and 1 in quality
        assert 0.0 <= quality[1] <= 1.0

    def test_suspect_gets_zero(self, val_buffer):
        model = LocalMLP()
        updates = {0: {k: v * 0.0 for k, v in model.state_dict().items()}}
        quality = compute_contribution_quality(
            model, updates, val_buffer,
            borderline_ids=set(), suspect_ids={0},
            quality_default_suspect=0.0,
        )
        assert quality[0] == 0.0

    def test_benign_gets_default(self, val_buffer):
        model = LocalMLP()
        updates = {0: {k: v * 0.0 for k, v in model.state_dict().items()}}
        quality = compute_contribution_quality(
            model, updates, val_buffer,
            borderline_ids=set(), suspect_ids=set(),
            quality_default_benign=1.0,
        )
        assert quality[0] == 1.0


# ---------------------------------------------------------------------------
# Final Trust Score
# ---------------------------------------------------------------------------

class TestCombineTrustScores:
    def test_weights_sum_check(self):
        with pytest.raises(ValueError):
            combine_trust_scores({0: 0.5}, {0: 0.5}, {0: 0.5},
                                  alpha_s=0.5, alpha_h=0.5, alpha_q=0.5)

    def test_valid_combination(self):
        sim = {0: 1.0, 1: 0.0}
        rep = {0: 1.0, 1: 0.0}
        qua = {0: 1.0, 1: 0.0}
        trust = combine_trust_scores(sim, rep, qua,
                                     alpha_s=0.4, alpha_h=0.3, alpha_q=0.3)
        assert trust[0] == pytest.approx(1.0)
        assert trust[1] == pytest.approx(0.0)

    def test_partial_scores(self):
        sim = {0: 0.8}
        rep = {0: 0.6}
        qua = {0: 0.9}
        trust = combine_trust_scores(sim, rep, qua,
                                     alpha_s=0.4, alpha_h=0.3, alpha_q=0.3)
        expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.9
        assert trust[0] == pytest.approx(expected, abs=1e-6)
