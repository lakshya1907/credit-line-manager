import pandas as pd
from .config import LIMIT_MULTIPLIERS, PD_INCREASE_MAX, PD_DECREASE_MIN
from .counterfactual import apply_new_limit_features
from .economics import balance_under_limit, robust_ep
from .calibrate import apply_calibrator

def recommend_limits(X_feat, pd_model, pd_calibrator, ead_model):
    df = X_feat.copy()
    id_col = "customer_id" if "customer_id" in df.columns else None

    def trainable(frame):
        return frame.drop(columns=[id_col]) if id_col else frame

    out_rows = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}")

        L0 = float(row["LIMIT_BAL"])

        # Create once (important optimization)
        row_df = pd.DataFrame([row])
        row_train = trainable(row_df)

        # Base balance prediction
        base_balance = float(ead_model.predict(row_train)[0])
        base_balance = max(base_balance, 0.0)

        # Baseline PD
        s0 = float(pd_model.predict_proba(row_train)[0, 1])
        pd0 = float(apply_calibrator(pd_calibrator, [s0])[0])

        # Baseline economics
        ead0 = balance_under_limit(base_balance, L0, L0)
        ep0, _ = robust_ep(pd0, ead0)

        best_L, best_pd, best_ead, best_ep = L0, pd0, ead0, ep0

        for m in LIMIT_MULTIPLIERS:
            L1 = L0 * float(m)

            cf = apply_new_limit_features(row, L1)
            cf_df = pd.DataFrame([cf])
            cf_train = trainable(cf_df)

            s1 = float(pd_model.predict_proba(cf_train)[0, 1])
            pd1 = float(apply_calibrator(pd_calibrator, [s1])[0])

            # Guardrails
            if L1 > L0 and pd1 > PD_INCREASE_MAX:
                continue

            if pd1 > PD_DECREASE_MIN and L1 > L0:
                continue

            ead1 = balance_under_limit(base_balance, L0, L1)
            ep1, _ = robust_ep(pd1, ead1)

            if ep1 > best_ep:
                best_L, best_pd, best_ead, best_ep = L1, pd1, ead1, ep1

        action = "hold"
        if best_L > L0 * 1.001:
            action = "increase"
        elif best_L < L0 * 0.999:
            action = "decrease"

        rec = {
            "current_limit": L0,
            "recommended_limit": best_L,
            "action": action,
            "pd_current": pd0,
            "pd_recommended": best_pd,
            "ead_current": ead0,
            "ead_recommended": best_ead,
            "ep_current": ep0,
            "ep_recommended": best_ep,
            "ep_uplift": best_ep - ep0,
            "el_uplift_proxy": (best_pd * best_ead) - (pd0 * ead0),
            "ead_uplift": best_ead - ead0
        }

        if id_col:
            rec["customer_id"] = row[id_col]

        out_rows.append(rec)

    rec_df = pd.DataFrame(out_rows)

    cols = ["customer_id"] if "customer_id" in rec_df.columns else []
    cols += [c for c in rec_df.columns if c != "customer_id"]

    return rec_df[cols]