import pandas as pd
import shap

REASON_MAP = [
    ("delinq_streak", "Increasing delinquency streak"),
    ("delinq_count_pos", "Frequent late payments"),
    ("util_std", "High utilization volatility"),
    ("util_trend", "Rising utilization trend"),
    ("pay_ratio_trend", "Payment ratio deteriorating"),
    ("bill_trend", "Bills increasing rapidly"),
    ("util_x_delinq", "High utilization + delinquency interaction"),
]

def shap_reasons(pd_model, X_trainable: pd.DataFrame, X_row_trainable: pd.DataFrame, topk=5):
    explainer = shap.TreeExplainer(pd_model)
    sv = explainer.shap_values(X_row_trainable)
    vals = sv[0]
    feats = X_row_trainable.columns
    imp = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:topk]
    return [f for f, _ in imp]

def reason_codes_from_features(top_features):
    codes = []
    for f, msg in REASON_MAP:
        if f in top_features:
            codes.append(msg)
    return codes[:3] if codes else ["Model-driven risk pattern detected"]


