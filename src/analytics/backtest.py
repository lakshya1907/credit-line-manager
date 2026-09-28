"""
src/analytics/backtest.py
──────────────────────────
"Time-based evaluation" for this dataset, honestly scoped.

The UCI credit dataset is a single cross-sectional snapshot: each customer
has 6 months of BILL_AMT/PAY_AMT/PAY_* history (Apr-Sep 2005) and ONE future
target (default in Oct 2005). There is no per-customer signup date and no
repeated observations at different points in time, so a classic "train on
data before date T, test on data after date T" backtest is not meaningful
here -- there's only one T, shared by everyone.

What the data DOES support: asking how much predictive signal comes from
*how much history* you have. build_restricted_features(df, n_months) builds
a small, consistent feature set from only the n_months most recent months
(the same aggregate statistics -- utilization, payment ratio, delinquency,
bill level -- computed over a shorter window), and
restricted_feature_backtest() trains a PD model on each window size and
compares validation performance. This is a real, defensible robustness
check (e.g. "would this still work with only 3 months of history on a new
customer?"), not a workaround dressed up as a time-series backtest.

Note: this uses a deliberately simpler, consistent feature set across all
window sizes (not the full engineered set in features.build_features), so
absolute metrics here are not directly comparable to the production
pd_model's reported metrics -- the point is the *relative* change across
window sizes, isolated from differences in feature engineering.
"""

import numpy as np
import pandas as pd
from ..config import EPS
from ..pd_model import train_pd_model

# PAY_0 is the most recent status column; PAY_1 does not exist in this
# dataset (a known quirk of the UCI file), so the chronological order is
# PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6.
STATUS_COLS_CHRONOLOGICAL = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def build_restricted_features(df: pd.DataFrame, n_months: int) -> pd.DataFrame:
    """A minimal, consistent feature set built from only the n_months most
    recent months (1 <= n_months <= 6)."""
    if not (1 <= n_months <= 6):
        raise ValueError("n_months must be between 1 and 6")

    bill_cols = [f"BILL_AMT{i}" for i in range(1, n_months + 1)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, n_months + 1)]
    stat_cols = STATUS_COLS_CHRONOLOGICAL[:n_months]

    out = pd.DataFrame(index=df.index)
    out["LIMIT_BAL"] = df["LIMIT_BAL"]

    util = df[bill_cols].to_numpy(dtype=float) / (df["LIMIT_BAL"].to_numpy(dtype=float)[:, None] + EPS)
    out["util_mean"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)
    out["util_std"] = util.std(axis=1, ddof=1) if n_months > 1 else 0.0

    pay_ratio = df[pay_amt_cols].to_numpy(dtype=float) / (
        df[bill_cols].clip(lower=1).to_numpy(dtype=float) + EPS
    )
    out["pay_ratio_mean"] = pay_ratio.mean(axis=1)
    out["pay_ratio_min"] = pay_ratio.min(axis=1)

    out["delinq_max"] = df[stat_cols].max(axis=1)
    out["delinq_mean"] = df[stat_cols].mean(axis=1)
    out["delinq_count_pos"] = (df[stat_cols] >= 1).sum(axis=1)

    out["bill_mean"] = df[bill_cols].mean(axis=1)

    return out


def restricted_feature_backtest(df_raw: pd.DataFrame, y: pd.Series, window_months=(6, 3, 1)) -> pd.DataFrame:
    """For each window size in window_months, build a restricted feature
    set and train a PD model (same training routine as production,
    src.pd_model.train_pd_model), returning validation metrics per window
    so they're directly comparable to each other."""
    rows = []
    for w in window_months:
        Xw = build_restricted_features(df_raw, w)
        _, metrics, _ = train_pd_model(Xw, y)
        rows.append({
            "window_months": w,
            "n_features": Xw.shape[1],
            "val_roc_auc": metrics["val_roc_auc"],
            "val_pr_auc": metrics["val_pr_auc"],
        })
    return pd.DataFrame(rows).sort_values("window_months", ascending=False).reset_index(drop=True)
