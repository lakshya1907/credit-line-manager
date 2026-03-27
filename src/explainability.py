"""
src/explainability.py
─────────────────────
SHAP-based explainability for the Credit Line Manager.

Provides:
    - Batch SHAP computation (cached explainer, one pass over all rows)
    - Per-customer top-feature extraction (robust to shap version differences)
    - Human-readable reason codes mapped from feature names
    - Direction-aware reasons (e.g. "rising" vs "falling" utilization)
    - Audit log row builder (full decision record for governance)
    - Lightweight summary: global feature importance from SHAP
"""

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Reason code map
# Each entry: (feature_substr, high_direction, reason_high, reason_low)
# high_direction: sign of SHAP value that triggers the "high" message
# ─────────────────────────────────────────────
_REASON_MAP = [
    # feature substring        high msg                                    low msg
    ("delinq_streak",          "Active delinquency streak",                "No recent delinquency streak"),
    ("delinq_count_pos",       "Frequent late payments on record",         "Consistently on-time payments"),
    ("delinq_max",             "Severe past delinquency",                  "No severe past delinquency"),
    ("delinq_mean",            "Elevated average delinquency status",      "Low average delinquency"),
    ("util_trend",             "Utilization trending upward",              "Utilization trending downward"),
    ("util_std",               "High utilization volatility",              "Stable utilization pattern"),
    ("util_max",               "Peak utilization is very high",            "Peak utilization is low"),
    ("util_mean",              "High average utilization",                 "Low average utilization"),
    ("util_last",              "Current utilization is elevated",          "Current utilization is low"),
    ("pay_ratio_trend",        "Payment ratio deteriorating over time",    "Payment ratio improving over time"),
    ("pay_ratio_min",          "Minimum payment ratio is very low",        "Minimum payment ratio is healthy"),
    ("pay_ratio_mean",         "Average payment ratio is low",             "Strong average payment coverage"),
    ("bill_trend",             "Bills increasing rapidly",                 "Bills stable or declining"),
    ("bill_std",               "High billing volatility",                  "Stable billing pattern"),
    ("util_x_delinq",          "High utilization combined with delinquency","Low combined utilization-delinquency risk"),
    ("ratio_x_util",           "Low payment ratio at high utilization",    "Good payment coverage at current utilization"),
    ("LIMIT_BAL",              "High credit limit increases exposure",     "Low credit limit reduces exposure"),
]


def _match_reason(feat: str, shap_val: float) -> str:
    """Return human reason string for a feature given its SHAP direction."""
    for substr, msg_high, msg_low in _REASON_MAP:
        if substr in feat:
            return msg_high if shap_val > 0 else msg_low
    # fallback: generic
    direction = "increases" if shap_val > 0 else "decreases"
    return f"{feat} {direction} default risk"


# ─────────────────────────────────────────────
# Explainer builder  (call once, reuse)
# ─────────────────────────────────────────────
def build_explainer(pd_model, X_background: pd.DataFrame = None):
    """
    Build a SHAP TreeExplainer.
    Passing X_background (a sample ~200 rows) enables interventional
    SHAP which is more accurate for correlated features.
    Without it, TreeExplainer uses the faster path-dependent method.
    """
    import shap
    if X_background is not None:
        bg = X_background.sample(min(200, len(X_background)), random_state=42)
        explainer = shap.TreeExplainer(pd_model, bg, feature_perturbation="interventional")
    else:
        explainer = shap.TreeExplainer(pd_model)
    return explainer


# ─────────────────────────────────────────────
# Batch SHAP (compute once for all rows)
# ─────────────────────────────────────────────
def compute_shap_batch(explainer, X_trainable: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for all rows in one pass.
    Returns array of shape (n_samples, n_features).
    Handles both old shap (<0.40) and new shap (>=0.40) return formats.
    """
    sv = explainer.shap_values(X_trainable)

    # shap >= 0.40 returns Explanation object or list
    if hasattr(sv, "values"):
        # Explanation object
        arr = sv.values
        if arr.ndim == 3:
            # multiclass: take class-1 slice
            arr = arr[:, :, 1]
        return arr

    if isinstance(sv, list):
        # binary classifier returns [shap_class0, shap_class1]
        if len(sv) == 2:
            return np.array(sv[1])   # class 1 = default
        return np.array(sv[0])

    # plain ndarray (regression or older binary)
    arr = np.array(sv)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr


# ─────────────────────────────────────────────
# Per-row top features + reasons
# ─────────────────────────────────────────────
def top_features_for_row(shap_row: np.ndarray, feature_names, topk: int = 5):
    """
    Given SHAP values for one row, return top-k (feature, shap_val) pairs
    sorted by |shap|.
    """
    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:topk]


def reason_codes_for_row(shap_row: np.ndarray, feature_names, topk: int = 5):
    """
    Return list of human-readable reason strings for one customer.
    Direction-aware: rising util vs falling util give different messages.
    """
    top = top_features_for_row(shap_row, feature_names, topk=topk)
    reasons = []
    seen_substrs = set()
    for feat, val in top:
        # Deduplicate if multiple util_* features all say same thing
        key = next((s for s, _, _ in _REASON_MAP if s in feat), feat)
        if key not in seen_substrs:
            reasons.append(_match_reason(feat, val))
            seen_substrs.add(key)
    return reasons[:3] if reasons else ["Model-driven risk pattern detected"]


# ─────────────────────────────────────────────
# Batch annotation: attach top features + reasons to rec_df
# ─────────────────────────────────────────────
def annotate_decisions(
    rec_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names,
    topk: int = 5,
) -> pd.DataFrame:
    """
    Add columns to rec_df:
        top_features   : comma-separated top SHAP feature names
        reason_codes   : pipe-separated human reason strings
        shap_pd_contrib: SHAP value of top feature (magnitude proxy)
    """
    df = rec_df.copy()

    top_feat_list   = []
    reason_list     = []
    top_shap_list   = []

    for i in range(len(shap_values)):
        row_sv = shap_values[i]
        top    = top_features_for_row(row_sv, feature_names, topk=topk)
        codes  = reason_codes_for_row(row_sv, feature_names, topk=topk)

        top_feat_list.append(", ".join(f for f, _ in top))
        reason_list.append(" | ".join(codes))
        top_shap_list.append(float(top[0][1]) if top else 0.0)

    df["top_features"]    = top_feat_list
    df["reason_codes"]    = reason_list
    df["top_shap_value"]  = top_shap_list

    return df


# ─────────────────────────────────────────────
# Global feature importance (mean |SHAP|)
# ─────────────────────────────────────────────
def global_importance(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """
    Returns DataFrame with mean absolute SHAP per feature, sorted descending.
    Useful for the dashboard model diagnostics page.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":    list(feature_names),
        "mean_shap":  mean_abs,
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Audit log builder
# ─────────────────────────────────────────────
def build_audit_log(
    rec_df: pd.DataFrame,
    pd_model_version: str = "pd_xgb_v1",
    calibrator_version: str = "isotonic_v1",
    ead_model_version: str  = "ead_xgb_v1",
    policy_constraints: dict = None,
) -> pd.DataFrame:
    """
    Build a full governance audit log — one row per customer decision.
    Contains everything needed to reconstruct or challenge a decision.
    """
    df = rec_df.copy()
    ts = datetime.datetime.utcnow().isoformat()

    df["audit_timestamp"]        = ts
    df["pd_model_version"]       = pd_model_version
    df["calibrator_version"]     = calibrator_version
    df["ead_model_version"]      = ead_model_version

    if policy_constraints:
        df["policy_el_budget"]   = policy_constraints.get("el_budget", None)
        df["policy_ead_budget"]  = policy_constraints.get("ead_budget", None)

    # Decision rationale string (human-readable summary)
    def _rationale(row):
        parts = [
            f"Action: {row.get('action', 'unknown')}",
            f"PD: {row.get('pd_current', 0):.4f} -> {row.get('pd_recommended', 0):.4f}",
            f"Limit: {row.get('current_limit', 0):,.0f} -> {row.get('recommended_limit', 0):,.0f}",
            f"EP uplift: {row.get('ep_uplift', 0):,.2f}",
        ]
        if "reason_codes" in row.index:
            parts.append(f"Reasons: {row['reason_codes']}")
        return " | ".join(parts)

    df["decision_rationale"] = df.apply(_rationale, axis=1)

    # Reorder columns: identifiers first, then decision, then risk, then audit
    id_cols    = [c for c in ["customer_id"] if c in df.columns]
    dec_cols   = [c for c in ["action", "current_limit", "recommended_limit",
                               "pd_current", "pd_recommended",
                               "ead_current", "ead_recommended",
                               "ep_current", "ep_recommended", "ep_uplift",
                               "el_uplift_proxy"] if c in df.columns]
    expl_cols  = [c for c in ["top_features", "reason_codes", "top_shap_value"] if c in df.columns]
    audit_cols = [c for c in ["audit_timestamp", "pd_model_version", "calibrator_version",
                               "ead_model_version", "policy_el_budget", "policy_ead_budget",
                               "decision_rationale"] if c in df.columns]
    other_cols = [c for c in df.columns
                  if c not in id_cols + dec_cols + expl_cols + audit_cols]

    ordered = id_cols + dec_cols + expl_cols + audit_cols + other_cols
    return df[ordered]


# ─────────────────────────────────────────────
# Convenience: run full explainability pass
# ─────────────────────────────────────────────
def run_explainability(
    pd_model,
    X_trainable: pd.DataFrame,
    rec_df: pd.DataFrame,
    topk: int = 5,
    sample_n: int = None,
    save_global_path: str = None,
    save_audit_path: str = None,
    policy_constraints: dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Full explainability pass. Returns:
        annotated_rec  : rec_df with top_features + reason_codes added
        global_imp     : global feature importance DataFrame
        shap_values    : raw SHAP array (n_samples, n_features)

    Args:
        sample_n       : if set, only compute SHAP on a random sample for speed.
                         Annotations for the rest get 'N/A'.
        save_global_path : if set, saves global importance CSV here.
        save_audit_path  : if set, saves full audit log CSV here.
    """
    n = len(X_trainable)
    feature_names = list(X_trainable.columns)

    # Build explainer (use background sample for interventional SHAP)
    print(f"  [Explainability] Building SHAP TreeExplainer on {n:,} rows…")
    explainer = build_explainer(pd_model, X_background=X_trainable)

    # Optionally subsample for speed
    if sample_n and sample_n < n:
        print(f"  [Explainability] Subsampling to {sample_n:,} rows for SHAP computation…")
        idx = np.random.default_rng(42).choice(n, size=sample_n, replace=False)
        idx_sorted = np.sort(idx)
        X_sub = X_trainable.iloc[idx_sorted]
    else:
        idx_sorted = np.arange(n)
        X_sub = X_trainable

    shap_values_sub = compute_shap_batch(explainer, X_sub)

    # Map back to full index (non-sampled rows get zeros → reason = generic)
    shap_values = np.zeros((n, len(feature_names)), dtype=np.float32)
    shap_values[idx_sorted] = shap_values_sub

    # Annotate decisions
    annotated = annotate_decisions(rec_df, shap_values, feature_names, topk=topk)

    # Global importance
    global_imp = global_importance(shap_values_sub, feature_names)
    print(f"  [Explainability] Top 5 global features by |SHAP|:")
    for _, row in global_imp.head(5).iterrows():
        print(f"      {row['feature']:30s}  {row['mean_shap']:.4f}")

    # Build + save audit log
    audit_log = build_audit_log(
        annotated,
        policy_constraints=policy_constraints,
    )

    if save_global_path:
        os.makedirs(os.path.dirname(save_global_path) or ".", exist_ok=True)
        global_imp.to_csv(save_global_path, index=False)
        print(f"  [Explainability] Saved global importance: {save_global_path}")

    if save_audit_path:
        os.makedirs(os.path.dirname(save_audit_path) or ".", exist_ok=True)
        audit_log.to_csv(save_audit_path, index=False)
        print(f"  [Explainability] Saved audit log: {save_audit_path}")

    return annotated, global_imp, shap_values