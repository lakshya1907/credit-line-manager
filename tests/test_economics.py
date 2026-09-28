import numpy as np
import pytest

from src import economics
from src.config import APR_ANNUAL_SCENARIOS, LGD_SCENARIOS


# ─────────────────────────────────────────────
# balance_under_limit
# ─────────────────────────────────────────────

def test_balance_unchanged_when_limit_unchanged():
    assert economics.balance_under_limit(1000.0, 5000.0, 5000.0) == pytest.approx(1000.0)


def test_balance_grows_when_limit_increases():
    base = economics.balance_under_limit(1000.0, 5000.0, 5000.0)
    higher = economics.balance_under_limit(1000.0, 5000.0, 10000.0)
    assert higher > base


def test_balance_shrinks_when_limit_decreases():
    base = economics.balance_under_limit(1000.0, 5000.0, 5000.0)
    lower = economics.balance_under_limit(1000.0, 5000.0, 2500.0)
    assert lower < base


def test_negative_base_balance_clipped_to_zero():
    assert economics.balance_under_limit(-500.0, 5000.0, 5000.0) == 0.0


def test_scale_floor_prevents_balance_from_collapsing_to_near_zero():
    # A drastic limit cut should still leave at least 20% of the base balance
    # (the `scale = max(scale, 0.2)` floor in balance_under_limit).
    result = economics.balance_under_limit(1000.0, 100000.0, 1.0)
    assert result == pytest.approx(1000.0 * 0.2)


def test_non_positive_limits_are_clipped_to_at_least_one():
    # L0/L1 <= 0 would blow up the log; both are clipped to >= 1.0 first.
    result = economics.balance_under_limit(1000.0, 0.0, 0.0)
    assert result == pytest.approx(1000.0)  # log(1/1) == 0 -> scale == 1


# ─────────────────────────────────────────────
# scenario_eps
# ─────────────────────────────────────────────

def test_scenario_eps_covers_full_apr_x_lgd_grid():
    eps = economics.scenario_eps(0.1, 1000.0)
    assert len(eps) == len(APR_ANNUAL_SCENARIOS) * len(LGD_SCENARIOS)


def test_scenario_eps_all_positive_when_pd_is_zero():
    # No default risk -> every scenario is pure interest revenue, all positive.
    eps = economics.scenario_eps(0.0, 1000.0)
    assert all(e > 0 for e in eps)


def test_scenario_eps_matches_hand_computed_value():
    pd_cal, ead = 0.2, 1000.0
    apr_a, lgd = APR_ANNUAL_SCENARIOS[0], LGD_SCENARIOS[0]
    expected = (apr_a / 12.0) * ead - pd_cal * ead * lgd
    eps = economics.scenario_eps(pd_cal, ead)
    assert eps[0] == pytest.approx(expected)


# ─────────────────────────────────────────────
# robust_ep
# ─────────────────────────────────────────────

def test_robust_ep_worst_case_takes_the_minimum_scenario(monkeypatch):
    monkeypatch.setattr(economics, "ROBUST_MODE", "worst_case")
    ep, eps = economics.robust_ep(0.15, 5000.0)
    assert ep == pytest.approx(min(eps))


def test_robust_ep_expected_mode_takes_the_mean_scenario(monkeypatch):
    monkeypatch.setattr(economics, "ROBUST_MODE", "expected")
    ep, eps = economics.robust_ep(0.15, 5000.0)
    assert ep == pytest.approx(np.mean(eps))


def test_robust_ep_worst_case_is_never_better_than_expected(monkeypatch):
    monkeypatch.setattr(economics, "ROBUST_MODE", "worst_case")
    worst_ep, _ = economics.robust_ep(0.15, 5000.0)
    monkeypatch.setattr(economics, "ROBUST_MODE", "expected")
    expected_ep, _ = economics.robust_ep(0.15, 5000.0)
    assert worst_ep <= expected_ep
