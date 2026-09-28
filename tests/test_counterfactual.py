import numpy as np
import pandas as pd
import pytest

from src.counterfactual import apply_new_limit_features, build_counterfactual_batch
from src.features import _slope, _slope_batch


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


# ─────────────────────────────────────────────
# _slope_batch (vectorized _slope, used by build_counterfactual_batch)
# ─────────────────────────────────────────────

def test_slope_batch_matches_row_by_row_slope():
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0, 1, size=(50, 6))
    expected = np.array([_slope(row) for row in matrix])
    actual = _slope_batch(matrix)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_slope_batch_handles_flat_rows():
    matrix = np.array([[0.5] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    actual = _slope_batch(matrix)
    assert actual[0] == pytest.approx(0.0)
    assert actual[1] == pytest.approx(_slope(matrix[1]))


# ─────────────────────────────────────────────
# build_counterfactual_batch vs. apply_new_limit_features (equivalence)
# ─────────────────────────────────────────────

def test_build_counterfactual_batch_matches_per_row_application():
    # The whole point of build_counterfactual_batch is to replace n*k
    # per-row apply_new_limit_features calls with one vectorized pass; the
    # two must produce identical features for identical (customer,
    # candidate-limit) pairs.
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(5):
        limit = float(rng.uniform(5000, 30000))
        bills = {f"BILL_AMT{i}": float(rng.uniform(0, limit * 1.5)) for i in range(1, 7)}
        data = {
            "LIMIT_BAL": limit,
            **bills,
            **{f"util_{i}": bills[f"BILL_AMT{i}"] / limit for i in range(1, 7)},
            "util_mean": 0.0, "util_max": 0.0, "util_std": 0.0,
            "util_last": 0.0, "util_trend": 0.0, "util_x_delinq": 0.0,
            "pay_ratio_mean": float(rng.uniform(0, 1)),
            "delinq_count_pos": int(rng.integers(0, 4)),
        }
        rows.append(data)
    df = pd.DataFrame(rows)

    multipliers = np.array([0.8, 1.0, 1.25])
    batch = build_counterfactual_batch(df, multipliers)

    for i in range(len(df)):
        for j, m in enumerate(multipliers):
            expected = apply_new_limit_features(df.iloc[i], df.iloc[i]["LIMIT_BAL"] * m)
            actual = batch.iloc[i * len(multipliers) + j]
            for col in ["LIMIT_BAL", "util_1", "util_2", "util_3", "util_4", "util_5", "util_6",
                        "util_mean", "util_max", "util_std", "util_last", "util_trend",
                        "util_x_delinq", "ratio_x_util"]:
                assert actual[col] == pytest.approx(expected[col], rel=1e-9, abs=1e-9), (
                    f"row {i}, multiplier {m}, column {col}"
                )
