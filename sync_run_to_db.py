"""
sync_run_to_db.py
───────────────────
Load one archived run's files (data/processed/runs/<run_id>/, written by
run_all.py's archive_run() and, if it was run, run_analytics.py) into
Postgres, per the schema in src/db/models.py.

Usage:
    python sync_run_to_db.py                  # syncs the latest run
    python sync_run_to_db.py --run-id <id>     # syncs a specific run
    python sync_run_to_db.py --all             # syncs every archived run not already in the DB

Requires: `alembic upgrade head` to have been run against DATABASE_URL
first (see .env.example / README). Idempotent per run_id -- re-syncing an
already-synced run_id deletes and replaces its rows (via ModelRun's
cascade delete) rather than erroring or duplicating.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.models import (
    BacktestResult, FairnessCheck, ModelRun, PortfolioRun, Recommendation, SegmentMetric, StressTestResult,
)
from src.db.session import get_engine, get_session_factory

RUNS_DIR = "data/processed/runs"
RUNS_INDEX_PATH = "data/processed/runs_index.csv"


def _read_csv_if_exists(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _native(value):
    """psycopg2 doesn't reliably adapt numpy scalar types (np.float64,
    np.int64, ...) -- pandas hands these back from .iterrows()/column
    access even though the DataFrame "looks" like plain floats/ints. Cast
    explicitly rather than relying on any implicit numpy<->DB-driver
    adapter (found the hard way: this broke on a 3-row insert into
    backtest_results with "schema \"np\" does not exist", from psycopg2
    literally stringifying an np.float64 repr into the query)."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_kwargs(row: pd.Series, cols) -> dict:
    return {c: _native(row[c]) for c in cols if c in row.index}


def load_run_manifest(run_id: str) -> dict:
    manifest_path = os.path.join(RUNS_DIR, run_id, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} not found -- is {run_id!r} a real archived run?")
    with open(manifest_path) as f:
        return json.load(f)


def build_model_run(manifest: dict) -> ModelRun:
    return ModelRun(
        run_id=manifest["run_id"],
        started_at=manifest["started_at_utc"],
        finished_at=manifest["finished_at_utc"],
        wall_time_seconds=manifest["wall_time_seconds"],
        git_commit=manifest.get("git_commit"),
        raw_data_path=manifest["raw_data_path"],
        config=manifest["config"],
        pd_roc_auc=manifest["pd_model_metrics"]["val_roc_auc"],
        pd_pr_auc=manifest["pd_model_metrics"]["val_pr_auc"],
        pd_brier_raw=manifest["pd_model_metrics"]["brier_raw"],
        pd_brier_calibrated=manifest["pd_model_metrics"]["brier_calibrated"],
        ead_mae=manifest["ead_model_metrics"]["val_mae"],
    )


def build_default_portfolio_run(manifest: dict) -> PortfolioRun:
    """The portfolio outcome run_all.py itself computed (policy_name =
    "default"), from the manifest's portfolio_summary -- separate from any
    named scenarios in policy_comparison.csv (see build_policy_comparison_rows)."""
    s = manifest["portfolio_summary"]
    cfg = manifest["config"]
    return PortfolioRun(
        policy_name="default",
        el_budget=s["el_budget"], ead_budget=s["ead_budget"],
        pd_increase_max=cfg["pd_increase_max"], pd_decrease_min=cfg["pd_decrease_min"],
        used_el=s["used_el"], used_ead=s["used_ead"],
        n_increase_applied=s["n_increase_applied"], n_decrease=s["n_decrease"], n_hold=s["n_hold"],
        total_ep_uplift=s["total_ep_uplift"],
    )


POLICY_COLS = ["el_budget", "ead_budget", "pd_increase_max", "pd_decrease_min",
               "used_el", "used_ead", "n_increase_applied", "n_decrease", "n_hold", "total_ep_uplift"]
RECOMMENDATION_COLS = ["customer_id", "current_limit", "recommended_limit", "action", "pd_current",
                        "pd_recommended", "ead_current", "ead_recommended", "ep_current", "ep_recommended",
                        "ep_uplift", "el_uplift_proxy", "ead_uplift", "top_features", "reason_codes", "top_shap_value"]
STRESS_COLS = ["pd_shock", "n_increase", "n_decrease", "n_hold", "total_ep_uplift", "el_used", "el_budget_pct"]
SEGMENT_COLS = ["segment", "segment_value", "n_customers", "increase_rate", "decrease_rate",
                "hold_rate", "avg_pd_current", "avg_ep_uplift", "total_ep_uplift"]
BACKTEST_COLS = ["window_months", "n_features", "val_roc_auc", "val_pr_auc"]
FAIRNESS_COLS = ["segment", "min_max_approval_ratio", "flag"]


def build_policy_comparison_rows(policy_df: pd.DataFrame) -> list[PortfolioRun]:
    return [
        PortfolioRun(policy_name=_native(r["policy"]), **_row_kwargs(r, POLICY_COLS))
        for _, r in policy_df.iterrows()
    ]


def build_recommendations(rec_df: pd.DataFrame) -> list[Recommendation]:
    return [Recommendation(**_row_kwargs(r, RECOMMENDATION_COLS)) for _, r in rec_df.iterrows()]


def build_stress_test_results(stress_df: pd.DataFrame) -> list[StressTestResult]:
    return [StressTestResult(**_row_kwargs(r, STRESS_COLS)) for _, r in stress_df.iterrows()]


def build_segment_metrics(segment_df: pd.DataFrame) -> list[SegmentMetric]:
    return [SegmentMetric(**_row_kwargs(r, SEGMENT_COLS)) for _, r in segment_df.iterrows()]


def build_backtest_results(backtest_df: pd.DataFrame) -> list[BacktestResult]:
    return [BacktestResult(**_row_kwargs(r, BACKTEST_COLS)) for _, r in backtest_df.iterrows()]


def build_fairness_checks(fairness_df: pd.DataFrame) -> list[FairnessCheck]:
    return [FairnessCheck(**_row_kwargs(r, FAIRNESS_COLS)) for _, r in fairness_df.iterrows()]


def sync_run(run_id: str, session) -> None:
    run_dir = os.path.join(RUNS_DIR, run_id)
    manifest = load_run_manifest(run_id)

    existing = session.get(ModelRun, run_id)
    if existing is not None:
        session.delete(existing)  # cascade deletes its children too
        session.flush()

    model_run = build_model_run(manifest)
    model_run.portfolio_runs.append(build_default_portfolio_run(manifest))

    stress_df = pd.DataFrame(manifest["stress_test"])
    if not stress_df.empty:
        model_run.stress_test_results.extend(build_stress_test_results(stress_df))

    rec_df = _read_csv_if_exists(os.path.join(run_dir, "recommendations_raw.csv"))
    if rec_df is not None:
        model_run.recommendations.extend(build_recommendations(rec_df))

    analytics_dir = os.path.join(run_dir, "analytics")
    segment_df = _read_csv_if_exists(os.path.join(analytics_dir, "segment_summary.csv"))
    if segment_df is not None:
        model_run.segment_metrics.extend(build_segment_metrics(segment_df))

    backtest_df = _read_csv_if_exists(os.path.join(analytics_dir, "backtest_window_months.csv"))
    if backtest_df is not None:
        model_run.backtest_results.extend(build_backtest_results(backtest_df))

    fairness_df = _read_csv_if_exists(os.path.join(analytics_dir, "fairness_check.csv"))
    if fairness_df is not None:
        model_run.fairness_checks.extend(build_fairness_checks(fairness_df))

    policy_df = _read_csv_if_exists(os.path.join(analytics_dir, "policy_comparison.csv"))
    if policy_df is not None:
        model_run.portfolio_runs.extend(build_policy_comparison_rows(policy_df))

    session.add(model_run)
    session.commit()

    print(f"  Synced {run_id}: {len(model_run.recommendations)} recommendations, "
          f"{len(model_run.portfolio_runs)} portfolio run(s), "
          f"{len(model_run.stress_test_results)} stress scenario(s), "
          f"{len(model_run.segment_metrics)} segment metric(s), "
          f"{len(model_run.backtest_results)} backtest row(s), "
          f"{len(model_run.fairness_checks)} fairness check(s)")


def main():
    parser = argparse.ArgumentParser(description="Sync an archived run into Postgres")
    parser.add_argument("--run-id", help="Specific run_id to sync (default: the latest run)")
    parser.add_argument("--all", action="store_true", help="Sync every archived run")
    args = parser.parse_args()

    if not os.path.exists(RUNS_INDEX_PATH):
        raise SystemExit(f"{RUNS_INDEX_PATH} not found -- run `python run_all.py` first.")
    index = pd.read_csv(RUNS_INDEX_PATH)

    if args.all:
        run_ids = list(index["run_id"])
    elif args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = [index.iloc[-1]["run_id"]]

    session_factory = get_session_factory(get_engine())
    with session_factory() as session:
        for run_id in run_ids:
            sync_run(run_id, session)


if __name__ == "__main__":
    main()
