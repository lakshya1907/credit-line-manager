import numpy as np
import pandas as pd
from .config import EPS

def apply_new_limit_features(row: pd.Series, new_limit: float) -> pd.Series:
    r = row.copy()
    r["LIMIT_BAL"] = float(new_limit)

    # Update utilization features if present
    for i in range(1, 7):
        b = f"BILL_AMT{i}"
        u = f"util_{i}"
        if b in r.index and u in r.index:
            r[u] = float(r[b]) / (float(new_limit) + EPS)

    util_cols = [c for c in r.index if c.startswith("util_")]
    if len(util_cols) > 0:
        vals = np.array([float(r[c]) for c in util_cols], dtype=float)
        r["util_mean"] = float(np.mean(vals))
        r["util_max"] = float(np.max(vals))
        r["util_std"] = float(np.std(vals))
        r["util_last"] = float(r["util_1"]) if "util_1" in r.index else float(vals[0])
        # trend left unchanged for MVP (optional: recompute slope)

    # Update interaction features if present
    if "delinq_count_pos" in r.index and "util_last" in r.index:
        r["util_x_delinq"] = float(r["util_last"]) * float(r["delinq_count_pos"])
    if "pay_ratio_mean" in r.index and "util_mean" in r.index:
        r["ratio_x_util"] = float(r["pay_ratio_mean"]) * float(r["util_mean"])

    return r
