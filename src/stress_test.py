"""
src/stress_test.py
───────────────────
Two ways to stress-test the portfolio, for two different use cases:

1. run_stress_scenario() — genuine re-simulation. Re-runs the full
   decision engine (recommend_limits) with the PD/EAD shock applied
   *inside* the simulation, so the guardrails and the EP-maximizing
   choice of limit can themselves change under stress (a customer whose
   baseline-optimal action is "increase" may re-optimize to "hold" once
   PD is shocked up). This is correct but slow (~seconds per customer x
   30k customers x 6 candidates), so it's used by the offline batch
   pipeline (run_all.py step_stress_test), not anything interactive.

2. apply_pd_shock() / apply_ead_shock() — fast approximation. Takes an
   *already-decided* recommendation table (whatever action the decision
   engine picked at baseline) and rescales its pd/ead columns by the
   shock factor. This is O(1) per shock level, which is what makes it
   usable behind an interactive slider (see the dashboard's Policy
   Simulator page), but it cannot let a customer's chosen action change
   under stress -- it only asks "how would the numbers on this same
   decision look under stress", not "would we still make this decision".
   Treat its output as a fast, approximate lower/upper bound, not as a
   substitute for run_stress_scenario() in an offline report.
"""

import pandas as pd
from .decision_engine import recommend_limits


def run_stress_scenario(X_feat, pd_model, pd_calibrator, ead_model, pd_shock=0.0, ead_shock=0.0):
    """Re-run the decision engine under a PD/EAD shock. See module docstring."""
    return recommend_limits(
        X_feat, pd_model, pd_calibrator, ead_model,
        pd_shock=pd_shock, ead_shock=ead_shock,
    )


def apply_pd_shock(rec_df: pd.DataFrame, shock=0.2):
    """Fast approximation: rescale pd_current/pd_recommended on an
    already-decided recommendation table. Does not re-optimize the
    chosen action. See module docstring."""
    df = rec_df.copy()
    df["pd_current"] = (df["pd_current"] * (1 + shock)).clip(0, 1)
    df["pd_recommended"] = (df["pd_recommended"] * (1 + shock)).clip(0, 1)
    df["el_uplift_proxy"] = (df["pd_recommended"] * df["ead_recommended"]) - (df["pd_current"] * df["ead_current"])
    return df


def apply_ead_shock(rec_df: pd.DataFrame, shock=0.1):
    """Fast approximation: rescale ead_current/ead_recommended on an
    already-decided recommendation table. Does not re-optimize the
    chosen action. See module docstring."""
    df = rec_df.copy()
    df["ead_current"] *= (1 + shock)
    df["ead_recommended"] *= (1 + shock)
    df["ead_uplift"] = df["ead_recommended"] - df["ead_current"]
    df["el_uplift_proxy"] = (df["pd_recommended"] * df["ead_recommended"]) - (df["pd_current"] * df["ead_current"])
    return df
