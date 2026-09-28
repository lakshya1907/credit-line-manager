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
        # NOTE: do not clamp d_el to >= 0. el_uplift_proxy = pd1*ead1 - pd0*ead0
        # is frequently negative here (raising a limit lowers utilization, which
        # lowers the counterfactual PD enough to outweigh the EAD increase), and
        # flooring those deltas to 0 before accumulating silently discards real
        # exposure growth from the running EL budget check, letting it never bind.
        # ead_uplift is left unclamped too, for the same reason, though in
        # practice it is already >= 0 for every "increase" row.
        d_el = float(r["el_uplift_proxy"])
        d_ead = float(r["ead_uplift"])
        if used_el + d_el <= el_budget and used_ead + d_ead <= ead_budget:
            chosen.append(True)
            used_el += d_el
            used_ead += d_ead
        else:
            chosen.append(False)

    inc["approved_by_portfolio"] = chosen
    rejected = ~inc["approved_by_portfolio"]
    # A rejected increase reverts to hold -- so every "recommended" figure
    # must also revert to "current", not just action/recommended_limit.
    # Leaving pd_recommended/ead_recommended/ep_uplift/el_uplift_proxy/
    # ead_uplift at their pre-rejection (would-have-been) values means a
    # row now labeled "hold" still reports the profit/risk of the increase
    # that was NOT actually approved -- which silently inflates
    # total_ep_uplift (and any other aggregate over these columns) by
    # exactly the hypothetical uplift of every rejected increase. This was
    # a real bug: e.g. halving EAD_BUDGET cut approved increases from
    # 4,610 to 2,288 while total_ep_uplift stayed bit-for-bit identical,
    # because the rejected increases' stale ep_uplift kept getting summed.
    inc.loc[rejected, "action"] = "hold"
    inc.loc[rejected, "recommended_limit"] = inc.loc[rejected, "current_limit"]
    inc.loc[rejected, "pd_recommended"] = inc.loc[rejected, "pd_current"]
    inc.loc[rejected, "ead_recommended"] = inc.loc[rejected, "ead_current"]
    inc.loc[rejected, "ep_recommended"] = inc.loc[rejected, "ep_current"]
    inc.loc[rejected, "ep_uplift"] = 0.0
    inc.loc[rejected, "el_uplift_proxy"] = 0.0
    inc.loc[rejected, "ead_uplift"] = 0.0

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

