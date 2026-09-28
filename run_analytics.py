"""
run_analytics.py
─────────────────
Analytics pass over the latest trained models: segment-level breakdowns
(including a fair-lending disparate-impact check), a cutoff-based PD
"backtest" (see src/analytics/backtest.py's module docstring for why this
dataset doesn't support a classic time-based split), and a policy
comparison across a few named guardrail/budget configurations.

Usage:
    python run_analytics.py          # uses models/*.pkl trained by run_all.py

Requires `python run_all.py` to have been run at least once (reads
models/pd_xgb.pkl, models/pd_calibrator.pkl, models/ead_xgb.pkl, and
data/processed/runs_index.csv to identify which run_id those models came
from). Writes to data/processed/analytics/ ("latest", gitignored like the
rest of data/processed) AND data/processed/runs/<run_id>/analytics/ (tying
this analytics pass to the specific model run it was computed against, the
same way run_all.py's archive_run() does for the core pipeline -- see
sync_run_to_db.py, which reads the per-run copy).
"""

import os
import sys
import time
import shutil
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep import load_uci, basic_clean
from src.features import build_features
from src.decision_engine import recommend_limits
from src.analytics.segments import add_segment_columns, segment_summary, disparate_approval_ratio, SEGMENT_COLUMNS
from src.analytics.backtest import restricted_feature_backtest
from src.analytics.policy_compare import compare_policies
from src.config import EL_BUDGET, EAD_BUDGET, PD_INCREASE_MAX, PD_DECREASE_MIN

RAW_PATH = "data/raw/uci_credit.csv"
MODEL_DIR = "models"
OUT_DIR = "data/processed/analytics"
RUNS_INDEX_PATH = "data/processed/runs_index.csv"
RUNS_DIR = "data/processed/runs"


def _current_run_id() -> str:
    """The run_id of the models/*.pkl currently on disk is (by
    construction -- run_all.py always retrains then archives in the same
    invocation) the last row of runs_index.csv."""
    if not os.path.exists(RUNS_INDEX_PATH):
        raise FileNotFoundError(
            f"{RUNS_INDEX_PATH} not found -- run `python run_all.py` first so "
            "there's a run_id to attach this analytics pass to."
        )
    idx = pd.read_csv(RUNS_INDEX_PATH)
    if idx.empty:
        raise ValueError(f"{RUNS_INDEX_PATH} is empty -- run `python run_all.py` first.")
    return idx.iloc[-1]["run_id"]


def _sep(title=""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print(f"\n{'─'*width}")


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    run_id = _current_run_id()
    print(f"  Attaching this analytics pass to run_id: {run_id}")

    _sep("Load models + data")
    pd_model = joblib.load(os.path.join(MODEL_DIR, "pd_xgb.pkl"))
    pd_calibrator = joblib.load(os.path.join(MODEL_DIR, "pd_calibrator.pkl"))
    ead_model = joblib.load(os.path.join(MODEL_DIR, "ead_xgb.pkl"))

    df = load_uci(RAW_PATH)
    df = basic_clean(df)
    X, y = build_features(df)
    print(f"  {len(X):,} customers, {X.shape[1]} features")

    _sep("Score current recommendations")
    rec = recommend_limits(X, pd_model, pd_calibrator, ead_model)
    segmented_X = add_segment_columns(X)

    _sep("Segment breakdowns")
    all_segments = []
    for col in SEGMENT_COLUMNS:
        summary = segment_summary(rec, segmented_X, col)
        all_segments.append(summary)
        print(f"\n  [{col}]")
        print(summary.to_string(index=False))
    segments_df = pd.concat(all_segments, ignore_index=True)
    segments_path = os.path.join(OUT_DIR, "segment_summary.csv")
    segments_df.to_csv(segments_path, index=False)
    print(f"\n  Saved: {segments_path}")

    _sep("Fair-lending check (disparate approval ratio)")
    fairness_rows = []
    for col in ["sex_segment", "marriage_segment"]:
        summary = segment_summary(rec, segmented_X, col)
        ratio = disparate_approval_ratio(summary)
        flag = "REVIEW" if ratio < 0.8 else "ok"
        fairness_rows.append({"segment": col, "min_max_approval_ratio": ratio, "flag": flag})
        print(f"  {col:18s}: min/max increase-rate ratio = {ratio:.3f}  [{flag}]")
    fairness_df = pd.DataFrame(fairness_rows)
    fairness_path = os.path.join(OUT_DIR, "fairness_check.csv")
    fairness_df.to_csv(fairness_path, index=False)
    print(f"  Saved: {fairness_path}")

    _sep("Cutoff-based PD backtest (window length)")
    backtest_df = restricted_feature_backtest(df, y, window_months=(6, 3, 1))
    print(backtest_df.to_string(index=False))
    backtest_path = os.path.join(OUT_DIR, "backtest_window_months.csv")
    backtest_df.to_csv(backtest_path, index=False)
    print(f"  Saved: {backtest_path}")

    _sep("Policy comparison")
    policies = {
        "current": {},
        "tighter_pd_guardrail": {"pd_increase_max": PD_INCREASE_MAX / 2},
        "looser_pd_guardrail": {"pd_increase_max": PD_INCREASE_MAX * 2},
        "smaller_ead_budget": {"ead_budget": EAD_BUDGET / 2},
        "larger_ead_budget": {"ead_budget": EAD_BUDGET * 2},
    }
    policy_df = compare_policies(X, pd_model, pd_calibrator, ead_model, policies)
    print(policy_df.to_string(index=False))
    policy_path = os.path.join(OUT_DIR, "policy_comparison.csv")
    policy_df.to_csv(policy_path, index=False)
    print(f"  Saved: {policy_path}")

    _sep("Archive to run history")
    run_analytics_dir = os.path.join(RUNS_DIR, run_id, "analytics")
    os.makedirs(run_analytics_dir, exist_ok=True)
    for path in [segments_path, fairness_path, backtest_path, policy_path]:
        shutil.copy2(path, os.path.join(run_analytics_dir, os.path.basename(path)))
    print(f"  Archived to: {run_analytics_dir}")

    _sep("DONE")
    print(f"  Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
