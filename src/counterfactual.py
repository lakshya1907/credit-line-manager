import numpy as np
import pandas as pd
from .config import EPS
from .features import _slope, _slope_batch


def _recompute_util_features(frame: pd.DataFrame, new_limit) -> pd.DataFrame:
    """
    Core recompute shared by apply_new_limit_features (single row) and
    build_counterfactual_batch (the whole customer x candidate-limit grid),
    so there is one implementation of "what changes when LIMIT_BAL changes"
    instead of two that can drift apart. `frame` must already have LIMIT_BAL
    set to `new_limit` (scalar or an array/Series aligned with frame's rows).

    util_cols is built explicitly from util_1..util_6 rather than by
    matching the "util_" prefix on frame.columns: frame still holds the
    *old* derived aggregates (util_mean, util_max, util_std, util_last,
    util_trend) and util_x_delinq at this point, all of which also start
    with "util_", so a prefix match would fold those stale values into what
    should be a mean/max/std over just the 6 raw per-period ratios.
    """
    out = frame.copy()
    new_limit_arr = np.asarray(new_limit, dtype=float)

    util_cols = []
    for i in range(1, 7):
        b = f"BILL_AMT{i}"
        u = f"util_{i}"
        if b in out.columns and u in out.columns:
            out[u] = out[b].to_numpy(dtype=float) / (new_limit_arr + EPS)
            util_cols.append(u)

    if util_cols:
        vals = out[util_cols].to_numpy(dtype=float)  # shape (m, 6)
        out["util_mean"] = vals.mean(axis=1)
        out["util_max"] = vals.max(axis=1)
        out["util_std"] = vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else 0.0
        out["util_last"] = out["util_1"]
        out["util_trend"] = _slope_batch(vals)

    if "delinq_count_pos" in out.columns:
        out["util_x_delinq"] = out["util_last"].to_numpy(dtype=float) * out["delinq_count_pos"].to_numpy(dtype=float)
    out["ratio_x_util"] = out["pay_ratio_mean"].to_numpy(dtype=float) * out["util_mean"].to_numpy(dtype=float)
    return out


def apply_new_limit_features(row: pd.Series, new_limit: float) -> pd.Series:
    """Single-customer convenience wrapper (e.g. an interactive what-if
    scoring endpoint) around _recompute_util_features."""
    frame = pd.DataFrame([row])
    frame["LIMIT_BAL"] = float(new_limit)
    frame = _recompute_util_features(frame, float(new_limit))
    out = frame.iloc[0]
    out.name = row.name
    return out


def build_counterfactual_batch(df: pd.DataFrame, multipliers: np.ndarray) -> pd.DataFrame:
    """
    Build the full (customer x candidate-limit) counterfactual feature grid
    in one vectorized pass, instead of calling apply_new_limit_features once
    per customer per candidate (the O(n*k) Python-loop-with-model-calls
    pattern that made decision_engine.py slow). Row order is
    customer-major, candidate-minor: row i*k+j is customer i under
    multipliers[j] -- reshape results to (n, k) with that same order to
    recover per-customer candidates.
    """
    n = len(df)
    k = len(multipliers)
    L0 = df["LIMIT_BAL"].to_numpy(dtype=float)
    new_limits = np.repeat(L0, k) * np.tile(np.asarray(multipliers, dtype=float), n)

    frame = df.loc[df.index.repeat(k)].reset_index(drop=True)
    frame["LIMIT_BAL"] = new_limits
    return _recompute_util_features(frame, new_limits)
