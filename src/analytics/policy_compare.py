"""
src/analytics/policy_compare.py
─────────────────────────────────
Run the decision engine + portfolio selection under several named policy
configurations (budgets, PD guardrails) and compare the resulting
portfolios in one table. Only practical now that decision_engine is
vectorized (each policy is a full ~30k-customer run, ~0.3s -- pre-#1's
performance fix this would have been ~10min per policy).
"""

import pandas as pd
from ..decision_engine import recommend_limits
from ..portfolio_opt import portfolio_select
from ..config import EL_BUDGET, EAD_BUDGET, PD_INCREASE_MAX, PD_DECREASE_MIN


def compare_policies(X, pd_model, pd_calibrator, ead_model, policies: dict) -> pd.DataFrame:
    """
    policies: {policy_name: {el_budget?, ead_budget?, pd_increase_max?,
    pd_decrease_min?}} -- any key omitted falls back to src.config's
    default. Returns one summary row per policy (portfolio_select's
    summary dict, plus the resolved policy parameters), in the order
    `policies` was given.
    """
    rows = []
    for name, overrides in policies.items():
        el_budget = overrides.get("el_budget", EL_BUDGET)
        ead_budget = overrides.get("ead_budget", EAD_BUDGET)
        pd_increase_max = overrides.get("pd_increase_max", PD_INCREASE_MAX)
        pd_decrease_min = overrides.get("pd_decrease_min", PD_DECREASE_MIN)

        rec = recommend_limits(
            X, pd_model, pd_calibrator, ead_model,
            pd_increase_max=pd_increase_max, pd_decrease_min=pd_decrease_min,
        )
        _, summary = portfolio_select(rec, el_budget=el_budget, ead_budget=ead_budget)

        rows.append({
            "policy": name,
            "el_budget": el_budget,
            "ead_budget": ead_budget,
            "pd_increase_max": pd_increase_max,
            "pd_decrease_min": pd_decrease_min,
            "n_increase_applied": summary["n_increase_applied"],
            "n_decrease": summary["n_decrease"],
            "n_hold": summary["n_hold"],
            "used_el": summary["used_el"],
            "used_ead": summary["used_ead"],
            "total_ep_uplift": summary["total_ep_uplift"],
        })
    return pd.DataFrame(rows)
