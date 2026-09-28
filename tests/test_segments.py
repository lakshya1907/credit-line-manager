import numpy as np
import pandas as pd
import pytest

from src.analytics.segments import add_segment_columns, segment_summary, disparate_approval_ratio


@pytest.fixture
def raw_X():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6],
        "SEX": [1, 2, 1, 2, 99, 2],          # 99 is an undocumented code -> "unknown"
        "EDUCATION": [1, 2, 3, 4, 2, 1],
        "MARRIAGE": [1, 2, 3, 1, 2, 0],       # 0 is undocumented -> "unknown"
        "AGE": [22, 30, 40, 50, 60, 27],
        "util_mean": [0.1, 0.5, 0.9, 0.2, 0.75, 0.35],
        "delinq_max": [0, 1, 3, 0, 2, 0],
    })


def test_add_segment_columns_maps_documented_codes(raw_X):
    seg = add_segment_columns(raw_X)
    assert list(seg["sex_segment"]) == ["male", "female", "male", "female", "unknown", "female"]
    assert list(seg["education_segment"]) == [
        "graduate_school", "university", "high_school", "other", "university", "graduate_school",
    ]
    assert list(seg["marriage_segment"]) == ["married", "single", "other", "married", "single", "unknown"]


def test_add_segment_columns_age_buckets(raw_X):
    seg = add_segment_columns(raw_X)
    assert list(seg["age_segment"]) == ["<25", "25-34", "35-44", "45-54", "55+", "25-34"]


def test_add_segment_columns_util_and_delinquency_tiers(raw_X):
    seg = add_segment_columns(raw_X)
    assert list(seg["util_tier"]) == [
        "low(<30%)", "medium(30-70%)", "high(>70%)", "low(<30%)", "high(>70%)", "medium(30-70%)",
    ]
    assert list(seg["delinquency_tier"]) == ["none", "mild(1-2mo)", "severe(3mo+)", "none", "mild(1-2mo)", "none"]


def test_add_segment_columns_preserves_row_count(raw_X):
    seg = add_segment_columns(raw_X)
    for col in ["sex_segment", "education_segment", "marriage_segment", "age_segment", "util_tier", "delinquency_tier"]:
        assert seg[col].notna().sum() == len(raw_X)


def make_rec_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "action": ["increase", "increase", "hold", "decrease"],
        "pd_current": [0.05, 0.10, 0.15, 0.20],
        "ep_uplift": [100.0, 50.0, 0.0, 10.0],
    })


def make_segmented_X():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "sex_segment": ["male", "male", "female", "female"],
    })


def test_segment_summary_aggregates_per_segment_value():
    rec_df = make_rec_df()
    segmented_X = make_segmented_X()
    summary = segment_summary(rec_df, segmented_X, "sex_segment")

    male = summary[summary["segment_value"] == "male"].iloc[0]
    assert male["n_customers"] == 2
    assert male["increase_rate"] == pytest.approx(1.0)
    assert male["avg_pd_current"] == pytest.approx((0.05 + 0.10) / 2)
    assert male["total_ep_uplift"] == pytest.approx(150.0)

    female = summary[summary["segment_value"] == "female"].iloc[0]
    assert female["n_customers"] == 2
    assert female["increase_rate"] == pytest.approx(0.0)
    assert female["decrease_rate"] == pytest.approx(0.5)
    assert female["hold_rate"] == pytest.approx(0.5)


def test_segment_summary_requires_customer_id():
    rec_df = make_rec_df().drop(columns=["customer_id"])
    segmented_X = make_segmented_X()
    with pytest.raises(ValueError):
        segment_summary(rec_df, segmented_X, "sex_segment")


def test_segment_summary_requires_segment_column_present():
    rec_df = make_rec_df()
    segmented_X = make_segmented_X()
    with pytest.raises(ValueError):
        segment_summary(rec_df, segmented_X, "not_a_real_segment")


def test_disparate_approval_ratio_full_parity_is_one():
    summary = pd.DataFrame({
        "segment_value": ["male", "female"],
        "n_customers": [100, 100],
        "increase_rate": [0.5, 0.5],
    })
    assert disparate_approval_ratio(summary) == pytest.approx(1.0)


def test_disparate_approval_ratio_flags_a_gap():
    summary = pd.DataFrame({
        "segment_value": ["male", "female"],
        "n_customers": [100, 100],
        "increase_rate": [0.5, 0.3],
    })
    assert disparate_approval_ratio(summary) == pytest.approx(0.6)


def test_disparate_approval_ratio_ignores_small_segments():
    summary = pd.DataFrame({
        "segment_value": ["male", "female", "other"],
        "n_customers": [100, 100, 5],   # "other" too small (<30) to count
        "increase_rate": [0.5, 0.5, 0.0],
    })
    assert disparate_approval_ratio(summary) == pytest.approx(1.0)
