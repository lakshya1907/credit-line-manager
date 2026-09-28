import numpy as np
import pandas as pd
from .config import EPS
from .features import _slope

def apply_new_limit_features(row: pd.Series, new_limit: float) -> pd.Series:
    r = row.copy()
    r["LIMIT_BAL"] = float(new_limit)

    # Recompute the six per-period utilization ratios under the new limit.
    # util_cols is built explicitly (util_1..util_6 only) rather than by
    # matching the "util_" prefix on r.index: at this point r still holds the
    # *old* derived aggregates (util_mean, util_max, util_std, util_last,
    # util_trend) and util_x_delinq, all of which also start with "util_", so
    # a prefix match would fold 6 stale aggregate values into what should be
    # a mean/max/std over just the 6 raw per-period ratios.
    util_cols = []
    for i in range(1,7):
        b = f"BILL_AMT{i}"
        u = f"util_{i}"
        if b in r.index and u in r.index:
            r[u] = float(r[b]) / (float(new_limit) + EPS)
            util_cols.append(u)

    if util_cols:
        vals = np.array([float(r[c]) for c in util_cols], dtype=float)
        r["util_mean"]  = float(vals.mean())
        r["util_max"]   = float(vals.max())
        r["util_std"]   = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        r["util_last"]  = float(r["util_1"])
        r["util_trend"] = _slope(vals)

    if "delinq_count_pos" in r.index:
        r["util_x_delinq"] = float(r["util_last"]) * float(r["delinq_count_pos"])
    r["ratio_x_util"] = float(r["pay_ratio_mean"]) * float(r["util_mean"])
    return r
