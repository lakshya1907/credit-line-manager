import numpy as np
import pandas as pd
import pytest

from sync_run_to_db import (
    _native, build_backtest_results, build_default_portfolio_run, build_fairness_checks,
    build_model_run, build_policy_comparison_rows, build_recommendations, build_segment_metrics,
    build_stress_test_results,
)


def test_native_converts_numpy_scalars_to_python_scalars():
    # Regression test: psycopg2 doesn't reliably adapt np.float64/np.int64
    # -- found via a real sync failure ("schema \"np\" does not exist")
    # against a live Postgres when a DataFrame-sourced value reached the
    # DB driver un-cast. See _native's docstring.
    assert type(_native(np.float64(1.5))) is float
    assert type(_native(np.int64(3))) is int
    assert _native("plain string") == "plain string"
    assert _native(5) == 5


def make_manifest():
    return {
        "run_id": "run-1",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:00:05+00:00",
        "wall_time_seconds": 5.0,
        "git_commit": "abc123",
        "raw_data_path": "data/raw/uci_credit.csv",
        "config": {"el_budget": 1.0, "pd_increase_max": 0.08, "pd_decrease_min": 0.2},
        "pd_model_metrics": {"val_roc_auc": 0.7, "val_pr_auc": 0.5, "brier_raw": 0.2, "brier_calibrated": 0.1},
        "ead_model_metrics": {"val_mae": 100.0},
        "portfolio_summary": {
            "el_budget": 1.0, "ead_budget": 2.0, "used_el": 0.5, "used_ead": 1.0,
            "n_increase_applied": 1, "n_decrease": 2, "n_hold": 3, "total_ep_uplift": 10.0,
        },
        "stress_test": [],
    }


def test_build_model_run_maps_manifest_fields():
    mr = build_model_run(make_manifest())
    assert mr.run_id == "run-1"
    assert mr.pd_roc_auc == 0.7
    assert mr.ead_mae == 100.0
    assert mr.config == {"el_budget": 1.0, "pd_increase_max": 0.08, "pd_decrease_min": 0.2}


def test_build_default_portfolio_run_uses_config_guardrails():
    pr = build_default_portfolio_run(make_manifest())
    assert pr.policy_name == "default"
    assert pr.n_increase_applied == 1
    assert pr.pd_increase_max == 0.08
    assert pr.pd_decrease_min == 0.2


def test_build_policy_comparison_rows_handles_numpy_dtypes():
    df = pd.DataFrame([{
        "policy": "tight", "el_budget": 1.0, "ead_budget": 2.0, "pd_increase_max": 0.04,
        "pd_decrease_min": 0.2, "used_el": 0.1, "used_ead": 0.2, "n_increase_applied": 5,
        "n_decrease": 6, "n_hold": 7, "total_ep_uplift": 8.0,
    }])
    rows = build_policy_comparison_rows(df)
    assert len(rows) == 1
    assert rows[0].policy_name == "tight"
    assert type(rows[0].n_increase_applied) is int
    assert type(rows[0].used_el) is float


def test_build_recommendations_only_includes_present_columns():
    df = pd.DataFrame([{
        "customer_id": 1, "current_limit": 1000.0, "recommended_limit": 1000.0, "action": "hold",
        "pd_current": 0.1, "pd_recommended": 0.1, "ead_current": 100.0, "ead_recommended": 100.0,
        "ep_current": 0.0, "ep_recommended": 0.0, "ep_uplift": 0.0, "el_uplift_proxy": 0.0, "ead_uplift": 0.0,
        # no top_features/reason_codes/top_shap_value -- e.g. audit not yet run for this row set
    }])
    recs = build_recommendations(df)
    assert len(recs) == 1
    assert recs[0].customer_id == 1
    assert recs[0].top_features is None


def test_build_stress_test_results_casts_numpy_types():
    df = pd.DataFrame([{
        "pd_shock": "+10%", "n_increase": 5, "n_decrease": 6, "n_hold": 7,
        "total_ep_uplift": 8.0, "el_used": 9.0, "el_budget_pct": 10.0,
    }])
    rows = build_stress_test_results(df)
    assert rows[0].pd_shock == "+10%"
    assert type(rows[0].n_increase) is int


def test_build_segment_metrics_maps_all_columns():
    df = pd.DataFrame([{
        "segment": "sex_segment", "segment_value": "female", "n_customers": 100,
        "increase_rate": 0.2, "decrease_rate": 0.5, "hold_rate": 0.3,
        "avg_pd_current": 0.1, "avg_ep_uplift": 50.0, "total_ep_uplift": 5000.0,
    }])
    rows = build_segment_metrics(df)
    assert rows[0].segment_value == "female"
    assert rows[0].n_customers == 100


def test_build_backtest_results_maps_all_columns():
    df = pd.DataFrame([{"window_months": 6, "n_features": 10, "val_roc_auc": 0.77, "val_pr_auc": 0.53}])
    rows = build_backtest_results(df)
    assert rows[0].window_months == 6
    assert rows[0].val_roc_auc == pytest.approx(0.77)


def test_build_fairness_checks_maps_all_columns():
    df = pd.DataFrame([{"segment": "marriage_segment", "min_max_approval_ratio": 0.683, "flag": "REVIEW"}])
    rows = build_fairness_checks(df)
    assert rows[0].flag == "REVIEW"
    assert rows[0].min_max_approval_ratio == pytest.approx(0.683)
