import numpy as np
import pandas as pd
from .config import EPS

def _slope(vals: np.ndarray) -> float:
    t = np.arange(1, len(vals) + 1)
    if np.std(vals) < 1e-12:
        return 0.0
    return float(np.polyfit(t, vals, 1)[0])

def build_features(df: pd.DataFrame):
    df = df.copy()
    y = df["TARGET"].astype(int)

    bill_cols = [f"BILL_AMT{i}" for i in range(1,7)]
    pay_cols  = [f"PAY_AMT{i}" for i in range(1,7)]
    stat_cols = [c for c in [f"PAY_{i}" for i in range(0, 7)] if c in df.columns]

    # Utilization features
    for i in range(1,7):
        df[f"util_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + EPS)

    util_cols = [f"util_{i}" for i in range(1,7)]
    df["util_mean"]  = df[util_cols].mean(axis=1)
    df["util_max"]   = df[util_cols].max(axis=1)
    df["util_std"]   = df[util_cols].std(axis=1).fillna(0)
    df["util_last"]  = df["util_1"]
    df["util_trend"] = df[util_cols].apply(lambda r: _slope(r.values.astype(float)), axis=1)

    # Payment ratios
    for i in range(1,7):
        df[f"pay_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].clip(lower=1) + EPS)

    pr_cols = [f"pay_ratio_{i}" for i in range(1,7)]
    df["pay_ratio_mean"]  = df[pr_cols].mean(axis=1)
    df["pay_ratio_min"]   = df[pr_cols].min(axis=1)
    df["pay_ratio_std"]   = df[pr_cols].std(axis=1).fillna(0)
    df["pay_ratio_trend"] = df[pr_cols].apply(lambda r: _slope(r.values.astype(float)), axis=1)

    # Delinquency dynamics
    df["delinq_max"] = df[stat_cols].max(axis=1)
    df["delinq_mean"] = df[stat_cols].mean(axis=1)
    df["delinq_std"] = df[stat_cols].std(axis=1).fillna(0)
    df["delinq_count_pos"] = (df[stat_cols] >= 1).sum(axis=1)

    def streak(row):
        s = 0
        for c in stat_cols:  # PAY_0 first (most recent)
            if row[c] >= 1:
                s += 1
            else:
                break
        return s

    df["delinq_streak"] = df[stat_cols].apply(streak, axis=1)

    # Bills
    df["bill_mean"]  = df[bill_cols].mean(axis=1)
    df["bill_std"]   = df[bill_cols].std(axis=1).fillna(0)
    df["bill_trend"] = df[bill_cols].apply(lambda r: _slope(r.values.astype(float)), axis=1)
    df["bill_last"]  = df["BILL_AMT1"]

    # Interactions (helps nonlinear models)
    df["util_x_delinq"] = df["util_last"] * df["delinq_count_pos"]
    df["ratio_x_util"]  = df["pay_ratio_mean"] * df["util_mean"]

    # Rename ID if present
    if "ID" in df.columns:
        df = df.rename(columns={"ID": "customer_id"})

    X = df.drop(columns=["TARGET"])
    return X, y
