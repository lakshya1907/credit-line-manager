"""
src/analytics/segments.py
──────────────────────────
Segment-level breakdowns of the decision engine's recommendations.

Two kinds of segment:
  - Behavioral (util_tier, delinquency_tier): the model's own risk signal,
    bucketed for readability. No fairness concern here.
  - Demographic (sex, education, marriage, age_bucket): SEX, EDUCATION, and
    MARRIAGE are fed to pd_model as raw numeric features today (see
    build_features / pd_model.py) -- i.e. the PD model can and does use sex
    and marital status directly as predictors. That is a real fair-lending
    concern in a lot of jurisdictions (in the US, ECOA prohibits using sex
    or marital status as a factor in a credit decision), independent of
    whether it's technically predictive in this dataset. These segment
    breakdowns exist partly to make that checkable -- see
    disparate_approval_ratio() -- not just to slice the data for its own
    sake.
"""

import numpy as np
import pandas as pd

SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate_school", 2: "university", 3: "high_school", 4: "other"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]

UTIL_BINS = [-np.inf, 0.3, 0.7, np.inf]
UTIL_LABELS = ["low(<30%)", "medium(30-70%)", "high(>70%)"]

DELINQ_BINS = [-np.inf, 0, 2, np.inf]
DELINQ_LABELS = ["none", "mild(1-2mo)", "severe(3mo+)"]


def add_segment_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of X (a build_features() output, or anything with the
    same SEX/EDUCATION/MARRIAGE/AGE/util_mean/delinq_max columns) with
    added *_segment columns. Unrecognized/undocumented codes (this dataset
    has a handful of SEX/EDUCATION/MARRIAGE values outside the documented
    UCI coding) map to "unknown" rather than being dropped, so segment
    counts always sum to len(X).
    """
    out = X.copy()

    if "SEX" in out.columns:
        out["sex_segment"] = out["SEX"].map(SEX_LABELS).fillna("unknown")
    if "EDUCATION" in out.columns:
        out["education_segment"] = out["EDUCATION"].map(EDUCATION_LABELS).fillna("unknown")
    if "MARRIAGE" in out.columns:
        out["marriage_segment"] = out["MARRIAGE"].map(MARRIAGE_LABELS).fillna("unknown")
    if "AGE" in out.columns:
        out["age_segment"] = pd.cut(out["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    if "util_mean" in out.columns:
        out["util_tier"] = pd.cut(out["util_mean"], bins=UTIL_BINS, labels=UTIL_LABELS).astype(str)
    if "delinq_max" in out.columns:
        out["delinquency_tier"] = pd.cut(out["delinq_max"], bins=DELINQ_BINS, labels=DELINQ_LABELS).astype(str)

    return out


SEGMENT_COLUMNS = ["sex_segment", "education_segment", "marriage_segment",
                    "age_segment", "util_tier", "delinquency_tier"]


def segment_summary(rec_df: pd.DataFrame, segmented_X: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """
    Join a decision-engine recommendation table (rec_df, from
    decision_engine.recommend_limits / recommendations_raw.csv) to a
    feature table that's already been through add_segment_columns, on
    customer_id, and summarize outcomes per segment value.

    Both frames must have a customer_id column.
    """
    if "customer_id" not in rec_df.columns or "customer_id" not in segmented_X.columns:
        raise ValueError("both rec_df and segmented_X must have a customer_id column")
    if segment_col not in segmented_X.columns:
        raise ValueError(f"{segment_col} not found; call add_segment_columns first")

    merged = rec_df.merge(segmented_X[["customer_id", segment_col]], on="customer_id", how="inner")

    grouped = merged.groupby(segment_col, observed=True)
    n = grouped.size()

    summary = pd.DataFrame({
        "n_customers": n,
        "increase_rate": grouped["action"].apply(lambda s: (s == "increase").mean()),
        "decrease_rate": grouped["action"].apply(lambda s: (s == "decrease").mean()),
        "hold_rate": grouped["action"].apply(lambda s: (s == "hold").mean()),
        "avg_pd_current": grouped["pd_current"].mean(),
        "avg_ep_uplift": grouped["ep_uplift"].mean(),
        "total_ep_uplift": grouped["ep_uplift"].sum(),
    }).reset_index().rename(columns={segment_col: "segment_value"})
    summary.insert(0, "segment", segment_col)
    return summary.sort_values("n_customers", ascending=False).reset_index(drop=True)


def disparate_approval_ratio(summary: pd.DataFrame) -> float:
    """
    Given a segment_summary() output for a protected-characteristic
    segment (sex_segment or marriage_segment), returns the ratio of the
    lowest to the highest increase_rate across segment values (the "four-
    fifths rule" quantity US fair-lending review commonly checks: a ratio
    below 0.8 is a common trigger for further review, though it is a rule
    of thumb, not a legal determination on its own). Segments with fewer
    than 30 customers are excluded as too small to draw a rate from.
    """
    eligible = summary[summary["n_customers"] >= 30]
    if len(eligible) < 2:
        return float("nan")
    rates = eligible["increase_rate"]
    if rates.max() == 0:
        return float("nan")
    return float(rates.min() / rates.max())
