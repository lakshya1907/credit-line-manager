import numpy as np
import pandas as pd
import pytest

from src.decision_engine import recommend_limits
from src.economics import balance_under_limit, robust_ep
from src.config import PD_INCREASE_MAX, LIMIT_MULTIPLIERS


class IdentityCalibrator:
    """A no-op calibrator: apply_calibrator(("isotonic", obj), s) -> s."""
    def predict(self, scores):
        return np.asarray(scores, dtype=float)


PD_CALIBRATOR = ("isotonic", IdentityCalibrator())


class ConstantPDModel:
    """Always predicts the same PD, regardless of the candidate limit."""
    def __init__(self, pd_value):
        self.pd_value = pd_value
        self.n_calls = 0

    def predict_proba(self, df):
        self.n_calls += 1
        n = len(df)
        p = np.full(n, self.pd_value)
        return np.column_stack([1 - p, p])


class ThresholdPDModel:
    """PD jumps above threshold as soon as the limit is raised at all."""
    def __init__(self, base_limit, pd_low, pd_when_increased):
        self.base_limit = base_limit
        self.pd_low = pd_low
        self.pd_when_increased = pd_when_increased

    def predict_proba(self, df):
        limits = df["LIMIT_BAL"].to_numpy(dtype=float)
        p = np.where(limits > self.base_limit + 1e-6, self.pd_when_increased, self.pd_low)
        return np.column_stack([1 - p, p])


class ConstantEADModel:
    def __init__(self, balance):
        self.balance = balance
        self.n_calls = 0

    def predict(self, df):
        self.n_calls += 1
        return np.full(len(df), self.balance)


def make_row(limit_bal, customer_id=None):
    bills = {f"BILL_AMT{i}": 1000.0 * i for i in range(1, 7)}
    data = {
        "LIMIT_BAL": limit_bal,
        **bills,
        **{f"util_{i}": bills[f"BILL_AMT{i}"] / limit_bal for i in range(1, 7)},
        "util_mean": np.mean([bills[f"BILL_AMT{i}"] / limit_bal for i in range(1, 7)]),
        "util_max": max(bills[f"BILL_AMT{i}"] / limit_bal for i in range(1, 7)),
        "util_std": np.std([bills[f"BILL_AMT{i}"] / limit_bal for i in range(1, 7)], ddof=1),
        "util_last": bills["BILL_AMT1"] / limit_bal,
        "util_trend": 0.0,
        "pay_ratio_mean": 0.5,
        "delinq_count_pos": 0,
        "util_x_delinq": 0.0,
        "ratio_x_util": 0.0,
    }
    if customer_id is not None:
        data["customer_id"] = customer_id
    return pd.Series(data)


def test_guardrail_blocks_increase_even_when_it_looks_profitable():
    # PD jumps well above PD_INCREASE_MAX for any candidate that raises the
    # limit, and stays low for the baseline / decreases. Every raised-limit
    # candidate must therefore be skipped by the guardrail, leaving hold
    # (baseline) as the best option -- never "increase".
    L0 = 10000.0
    X = pd.DataFrame([make_row(L0)])
    pd_model = ThresholdPDModel(L0, pd_low=0.001, pd_when_increased=PD_INCREASE_MAX + 0.5)
    ead_model = ConstantEADModel(2000.0)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)

    assert rec.iloc[0]["action"] != "increase"


def test_profitable_increase_is_selected_and_arithmetic_matches_economics():
    # PD stays tiny and constant regardless of limit -> raising the limit
    # (more EAD, same tiny risk) should strictly improve EP, so the engine
    # should pick the largest multiplier.
    L0 = 10000.0
    base_balance = 2000.0
    pd_value = 0.01

    X = pd.DataFrame([make_row(L0, customer_id="cust-1")])
    pd_model = ConstantPDModel(pd_value)
    ead_model = ConstantEADModel(base_balance)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)
    row = rec.iloc[0]

    best_multiplier = max(LIMIT_MULTIPLIERS)
    L1 = L0 * best_multiplier

    assert row["customer_id"] == "cust-1"
    assert row["action"] == "increase"
    assert row["recommended_limit"] == pytest.approx(L1)

    ead0 = balance_under_limit(base_balance, L0, L0)
    ead1 = balance_under_limit(base_balance, L0, L1)
    ep0, _ = robust_ep(pd_value, ead0)
    ep1, _ = robust_ep(pd_value, ead1)

    assert row["ead_current"] == pytest.approx(ead0)
    assert row["ead_recommended"] == pytest.approx(ead1)
    assert row["ep_current"] == pytest.approx(ep0)
    assert row["ep_recommended"] == pytest.approx(ep1)
    assert row["ep_uplift"] == pytest.approx(ep1 - ep0)
    assert row["ead_uplift"] == pytest.approx(ead1 - ead0)
    assert row["el_uplift_proxy"] == pytest.approx(pd_value * ead1 - pd_value * ead0)


def test_ead_model_is_queried_once_per_row_not_once_per_candidate():
    # decision_engine.py computes base_balance once before looping over
    # LIMIT_MULTIPLIERS ("Create once (important optimization)"); it must
    # not re-query the EAD model for every candidate limit.
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.01)
    ead_model = ConstantEADModel(2000.0)

    recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)

    assert ead_model.n_calls == len(X)


def test_customer_id_column_is_preserved_and_ordered_first():
    X = pd.DataFrame([make_row(10000.0, customer_id="abc"), make_row(20000.0, customer_id="xyz")])
    pd_model = ConstantPDModel(0.05)
    ead_model = ConstantEADModel(1000.0)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)

    assert rec.columns[0] == "customer_id"
    assert list(rec["customer_id"]) == ["abc", "xyz"]


def test_no_customer_id_column_when_absent_from_input():
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.05)
    ead_model = ConstantEADModel(1000.0)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)

    assert "customer_id" not in rec.columns


# ─────────────────────────────────────────────
# pd_shock / ead_shock (re-simulation under stress, see src/stress_test.py)
# ─────────────────────────────────────────────

def test_pd_shock_can_flip_a_baseline_increase_away_from_increase():
    # pd_value=0.01 is comfortably under PD_INCREASE_MAX (0.08) and under
    # the worst-case-EP breakeven PD (~0.01875 given this config's APR/LGD
    # grid), so at baseline the engine picks "increase" (matches
    # test_profitable_increase_is_selected_and_arithmetic_matches_economics).
    # An 800% pd_shock (deliberately extreme, chosen only to cross
    # PD_INCREASE_MAX deterministically, not a realistic macro assumption)
    # pushes the shocked PD to 0.09 > PD_INCREASE_MAX, so every raised-limit
    # candidate is rejected by the guardrail on re-simulation -- this is the
    # whole point of re-simulating under stress rather than rescaling an
    # already-decided recommendation, which could never change the action.
    L0 = 10000.0
    pd_value = 0.01
    X = pd.DataFrame([make_row(L0)])
    pd_model = ConstantPDModel(pd_value)
    ead_model = ConstantEADModel(2000.0)

    baseline = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=0.0)
    assert baseline.iloc[0]["action"] == "increase"

    shocked = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=8.0)
    assert shocked.iloc[0]["action"] != "increase"
    assert shocked.iloc[0]["pd_current"] == pytest.approx(min(pd_value * 9.0, 1.0))


def test_pd_shock_is_clipped_to_one():
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.9)
    ead_model = ConstantEADModel(1000.0)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=1.0)

    assert rec.iloc[0]["pd_current"] == pytest.approx(1.0)


def test_ead_shock_scales_base_balance_before_elasticity():
    L0 = 10000.0
    base_balance = 2000.0
    ead_shock = 0.15
    X = pd.DataFrame([make_row(L0)])
    pd_model = ConstantPDModel(0.05)
    ead_model = ConstantEADModel(base_balance)

    rec = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model, ead_shock=ead_shock)
    row = rec.iloc[0]

    expected_ead0 = balance_under_limit(base_balance * (1 + ead_shock), L0, L0)
    assert row["ead_current"] == pytest.approx(expected_ead0)


def test_zero_shock_matches_unshocked_call():
    X = pd.DataFrame([make_row(10000.0)])
    pd_model = ConstantPDModel(0.05)
    ead_model = ConstantEADModel(1000.0)

    unshocked = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model)
    zero_shocked = recommend_limits(X, pd_model, PD_CALIBRATOR, ead_model, pd_shock=0.0, ead_shock=0.0)

    pd.testing.assert_frame_equal(unshocked, zero_shocked)
