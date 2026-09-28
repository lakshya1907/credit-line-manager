import numpy as np
import pandas as pd
import pytest

from src.counterfactual import apply_new_limit_features
from src.features import _slope


@pytest.fixture
def base_row():
    """
    A synthetic post-build_features row. BILL_AMT1..6 are set so the six
    resulting util_i values are NOT all equal (so mean/max/std/trend are
    all distinguishable from each other), and the stale util_mean/max/std/
    last/trend/util_x_delinq are deliberately set to values that do NOT
    match what a correct recompute would give -- if apply_new_limit_features
    ever again contaminates its aggregate recompute with these stale
    values (the bug fixed in src/counterfactual.py), these tests will fail.
    """
    old_limit = 10000.0
    bills = {f"BILL_AMT{i}": v for i, v in zip(range(1, 7), [1000, 1500, 2000, 2500, 3000, 3500])}
    data = {
        "LIMIT_BAL": old_limit,
        **bills,
        "util_1": bills["BILL_AMT1"] / old_limit,
        "util_2": bills["BILL_AMT2"] / old_limit,
        "util_3": bills["BILL_AMT3"] / old_limit,
        "util_4": bills["BILL_AMT4"] / old_limit,
        "util_5": bills["BILL_AMT5"] / old_limit,
        "util_6": bills["BILL_AMT6"] / old_limit,
        # deliberately wrong stale aggregates (would-be contamination values)
        "util_mean": 999.0,
        "util_max": 999.0,
        "util_std": 999.0,
        "util_last": 999.0,
        "util_trend": 999.0,
        "util_x_delinq": 999.0,
        "pay_ratio_mean": 0.4,
        "delinq_count_pos": 2,
    }
    return pd.Series(data)


def test_limit_bal_updated(base_row):
    cf = apply_new_limit_features(base_row, 20000.0)
    assert cf["LIMIT_BAL"] == 20000.0


def test_per_period_utilization_recomputed_against_new_limit(base_row):
    new_limit = 20000.0
    cf = apply_new_limit_features(base_row, new_limit)
    for i in range(1, 7):
        expected = base_row[f"BILL_AMT{i}"] / new_limit
        assert cf[f"util_{i}"] == pytest.approx(expected, rel=1e-6)


def test_util_mean_uses_only_the_six_period_ratios(base_row):
    # Regression test for the bug where util_cols was selected via
    # `c.startswith("util_")`, which also swept in the stale util_mean/
    # max/std/last/trend/util_x_delinq columns still present on the row.
    new_limit = 20000.0
    cf = apply_new_limit_features(base_row, new_limit)
    expected_vals = [base_row[f"BILL_AMT{i}"] / new_limit for i in range(1, 7)]
    assert cf["util_mean"] == pytest.approx(np.mean(expected_vals))
    assert cf["util_max"] == pytest.approx(np.max(expected_vals))
    assert cf["util_std"] == pytest.approx(np.std(expected_vals, ddof=1))


def test_util_last_is_util_1(base_row):
    cf = apply_new_limit_features(base_row, 20000.0)
    assert cf["util_last"] == cf["util_1"]


def test_util_trend_is_recomputed_not_left_stale(base_row):
    new_limit = 20000.0
    cf = apply_new_limit_features(base_row, new_limit)
    expected_vals = np.array([base_row[f"BILL_AMT{i}"] / new_limit for i in range(1, 7)])
    assert cf["util_trend"] == pytest.approx(_slope(expected_vals))
    assert cf["util_trend"] != pytest.approx(999.0)


def test_interactions_recomputed_from_fresh_aggregates(base_row):
    new_limit = 20000.0
    cf = apply_new_limit_features(base_row, new_limit)
    assert cf["util_x_delinq"] == pytest.approx(cf["util_last"] * base_row["delinq_count_pos"])
    assert cf["ratio_x_util"] == pytest.approx(base_row["pay_ratio_mean"] * cf["util_mean"])


def test_increasing_limit_lowers_utilization(base_row):
    cf = apply_new_limit_features(base_row, base_row["LIMIT_BAL"] * 2)
    original_mean = np.mean([base_row[f"util_{i}"] for i in range(1, 7)])
    assert cf["util_mean"] < original_mean
