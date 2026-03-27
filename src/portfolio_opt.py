import pandas as pd
from .config import EL_BUDGET, EAD_BUDGET

def portfolio_select(rec_df: pd.DataFrame, el_budget=EL_BUDGET, ead_budget=EAD_BUDGET):
    df = rec_df.copy()

    # Only increases consume more exposure; decreases are always "safe" to apply
    inc = df[df["action"] == "increase"].copy()
    dec = df[df["action"] == "decrease"].copy()
    hold = df[df["action"] == "hold"].copy()

    # ROI score (profit per unit risk proxy)
    inc["roi"] = inc["ep_uplift"] / (inc["el_uplift_proxy"].abs() + 1e-6)

    inc = inc.sort_values("roi", ascending=False)

    chosen = []
    used_el = 0.0
    used_ead = 0.0

    for _, r in inc.iterrows():
        d_el = float(max(r["el_uplift_proxy"], 0.0))
        d_ead = float(max(r["ead_uplift"], 0.0))
        if used_el + d_el <= el_budget and used_ead + d_ead <= ead_budget:
            chosen.append(True)
            used_el += d_el
            used_ead += d_ead
        else:
            chosen.append(False)

    inc["approved_by_portfolio"] = chosen
    inc.loc[~inc["approved_by_portfolio"], "action"] = "hold"
    inc.loc[~inc["approved_by_portfolio"], "recommended_limit"] = inc["current_limit"]

    out = pd.concat([inc, dec, hold], axis=0).sort_index()
    summary = {
        "used_el": used_el,
        "used_ead": used_ead,
        "el_budget": el_budget,
        "ead_budget": ead_budget,
        "n_increase_applied": int((out["action"] == "increase").sum()),
        "n_decrease": int((out["action"] == "decrease").sum()),
        "n_hold": int((out["action"] == "hold").sum()),
        "total_ep_uplift": float(out["ep_uplift"].sum())
    }
    return out, summary

