import numpy as np
import pandas as pd

from .config import LIMIT_MULTIPLIERS
from .counterfactual import apply_new_limit_features
from .economics import balance_under_limit, compute_ep
from .calibrate import apply_calibrator

def recommend_limits_v2(X_feat: pd.DataFrame, pd_model, pd_calibrator, ead_model):
    df = X_feat.copy()
    id_col = "customer_id" if "customer_id" in df.columns else None

    def _to_trainable(frame):
        return frame.drop(columns=[id_col]) if id_col else frame

    rec_rows = []

    records = df.to_dict(orient="records")

    for idx, row in enumerate(records):
        if idx % 1000 == 0:
            print(f"Processing row {idx}")

        row_series = pd.Series(row)

        L0 = float(row_series["LIMIT_BAL"])

        row_df = pd.DataFrame([row_series])
        trainable = _to_trainable(row_df)

        base_balance = float(ead_model.predict(trainable)[0])
        base_balance = max(base_balance, 0.0)

        s0 = float(pd_model.predict_proba(trainable)[0, 1])
        pd0 = float(apply_calibrator(pd_calibrator, [s0])[0])

        ead0 = balance_under_limit(base_balance, L0, L0)
        ep0_robust, _ = compute_ep(pd0, ead0)

        best = {"L": L0, "pd": pd0, "ead": ead0, "ep": ep0_robust}

        for m in LIMIT_MULTIPLIERS:
            L1 = L0 * float(m)

            cf_series = apply_new_limit_features(row_series, L1)
            cf_df = pd.DataFrame([cf_series])
            cf_trainable = _to_trainable(cf_df)

            s1 = float(pd_model.predict_proba(cf_trainable)[0, 1])
            pd1 = float(apply_calibrator(pd_calibrator, [s1])[0])

            ead1 = balance_under_limit(base_balance, L0, L1)
            ep1, _ = compute_ep(pd1, ead1)

            if ep1 > best["ep"]:
                best = {"L": L1, "pd": pd1, "ead": ead1, "ep": ep1}

        action = "hold"
        if best["L"] > L0 * 1.001:
            action = "increase"
        elif best["L"] < L0 * 0.999:
            action = "decrease"

        out = {
            "current_limit": L0,
            "recommended_limit": best["L"],
            "action": action,
            "pd_cal_current": pd0,
            "pd_cal_recommended": best["pd"],
            "ead_current": ead0,
            "ead_recommended": best["ead"],
            "ep_current_robust": ep0_robust,
            "ep_recommended_robust": best["ep"],
            "ep_uplift": best["ep"] - ep0_robust,
        }

        if id_col:
            out["customer_id"] = row_series[id_col]

        rec_rows.append(out)

    rec = pd.DataFrame(rec_rows)

    cols = ["customer_id"] if "customer_id" in rec.columns else []
    cols += [c for c in rec.columns if c != "customer_id"]

    return rec[cols]