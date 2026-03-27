import numpy as np
import pandas as pd
from .config import EPS

def apply_new_limit_features(row: pd.Series, new_limit: float) -> pd.Series:
    r = row.copy()
    r["LIMIT_BAL"] = float(new_limit)

    for i in range(1,7):
        b = f"BILL_AMT{i}"
        u = f"util_{i}"
        if b in r.index and u in r.index:
            r[u] = float(r[b]) / (float(new_limit) + EPS)

    util_cols = [c for c in r.index if c.startswith("util_")]
    if util_cols:
        vals = np.array([float(r[c]) for c in util_cols], dtype=float)
        r["util_mean"] = float(vals.mean())
        r["util_max"]  = float(vals.max())
        r["util_std"]  = float(vals.std())
        r["util_last"] = float(r["util_1"])
        # optional recompute trend

    if "delinq_count_pos" in r.index:
        r["util_x_delinq"] = float(r["util_last"]) * float(r["delinq_count_pos"])
    r["ratio_x_util"] = float(r["pay_ratio_mean"]) * float(r["util_mean"])
    return r
