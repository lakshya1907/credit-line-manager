import pandas as pd

def apply_pd_shock(rec_df: pd.DataFrame, shock=0.2):
    df = rec_df.copy()
    df["pd_current"] = (df["pd_current"] * (1 + shock)).clip(0, 1)
    df["pd_recommended"] = (df["pd_recommended"] * (1 + shock)).clip(0, 1)
    # Recompute risk proxy uplift only (MVP); full recompute would re-run simulation
    df["el_uplift_proxy"] = (df["pd_recommended"] * df["ead_recommended"]) - (df["pd_current"] * df["ead_current"])
    return df

def apply_ead_shock(rec_df: pd.DataFrame, shock=0.1):
    df = rec_df.copy()
    df["ead_current"] *= (1 + shock)
    df["ead_recommended"] *= (1 + shock)
    df["ead_uplift"] = df["ead_recommended"] - df["ead_current"]
    df["el_uplift_proxy"] = (df["pd_recommended"] * df["ead_recommended"]) - (df["pd_current"] * df["ead_current"])
    return df
