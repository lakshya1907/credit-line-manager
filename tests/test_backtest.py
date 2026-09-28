import numpy as np
import pandas as pd
import pytest

from src.analytics.backtest import build_restricted_features, restricted_feature_backtest


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 40
    data = {"LIMIT_BAL": rng.uniform(5000, 30000, n)}
    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = rng.uniform(0, 20000, n)
        data[f"PAY_AMT{i}"] = rng.uniform(0, 5000, n)
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[c] = rng.integers(-1, 4, n)
    return pd.DataFrame(data)


def test_build_restricted_features_uses_only_the_requested_window(raw_df):
    restricted = build_restricted_features(raw_df, n_months=3)
    expected_util = raw_df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].to_numpy() / raw_df["LIMIT_BAL"].to_numpy()[:, None]
    np.testing.assert_allclose(restricted["util_mean"], expected_util.mean(axis=1), rtol=1e-9)


def test_build_restricted_features_delinq_uses_chronological_status_cols(raw_df):
    restricted = build_restricted_features(raw_df, n_months=2)
    expected = raw_df[["PAY_0", "PAY_2"]].max(axis=1)
    pd.testing.assert_series_equal(restricted["delinq_max"], expected, check_names=False)


def test_build_restricted_features_rejects_invalid_window(raw_df):
    with pytest.raises(ValueError):
        build_restricted_features(raw_df, n_months=0)
    with pytest.raises(ValueError):
        build_restricted_features(raw_df, n_months=7)


def test_build_restricted_features_single_month_has_zero_std(raw_df):
    restricted = build_restricted_features(raw_df, n_months=1)
    assert (restricted["util_std"] == 0.0).all()


def test_restricted_feature_backtest_returns_one_row_per_window(raw_df):
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(raw_df)))
    result = restricted_feature_backtest(raw_df, y, window_months=(3, 1))
    assert list(result["window_months"]) == [3, 1]
    assert {"val_roc_auc", "val_pr_auc", "n_features"}.issubset(result.columns)
