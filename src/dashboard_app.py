"""
src/dashboard_app.py
Adaptive Credit Line Manager — Product Dashboard

Run with:
    streamlit run src/dashboard_app.py
    or via run_all.py after pipeline completes.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── path fix so imports work whether called as src/dashboard_app.py or standalone ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio_opt import portfolio_select
from src.stress_test import apply_pd_shock, apply_ead_shock
from src.config import (
    EL_BUDGET, EAD_BUDGET,
    APR_ANNUAL_SCENARIOS, LGD_SCENARIOS,
    LIMIT_MULTIPLIERS,
)

# ─────────────────────────────────────────────
# Constants / paths
# ─────────────────────────────────────────────
REC_PATH        = "data/processed/recommendations_raw.csv"
PD_MODEL_PATH   = "models/pd_xgb.pkl"
CALIB_PATH      = "models/pd_calibrator.pkl"
EAD_MODEL_PATH  = "models/ead_xgb.pkl"

# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────
@st.cache_data
def load_rec(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    pd_model     = joblib.load(PD_MODEL_PATH)   if os.path.exists(PD_MODEL_PATH)   else None
    pd_calibrator = joblib.load(CALIB_PATH)      if os.path.exists(CALIB_PATH)      else None
    ead_model    = joblib.load(EAD_MODEL_PATH)   if os.path.exists(EAD_MODEL_PATH)  else None
    return pd_model, pd_calibrator, ead_model

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fmt_currency(v: float, decimals: int = 0) -> str:
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:.{decimals}f}"

def action_color(action: str) -> str:
    return {"increase": "#22c55e", "decrease": "#ef4444", "hold": "#94a3b8"}.get(action, "#94a3b8")

def metric_delta_color(v: float) -> str:
    return "normal" if v >= 0 else "inverse"


# ─────────────────────────────────────────────
# Profit-vs-limit curve for a single customer
# (recomputes EP across candidate multipliers
#  using the economics formulas directly — no
#  model re-inference needed for the chart)
# ─────────────────────────────────────────────
def profit_vs_limit_chart(row: pd.Series) -> go.Figure:
    from src.economics import balance_under_limit, scenario_eps
    from src.config import EAD_ELASTICITY

    L0 = float(row["current_limit"])
    base_ead = float(row.get("ead_current", L0 * 0.3))
    pd0 = float(row.get("pd_current", 0.05))

    multipliers = sorted(set(LIMIT_MULTIPLIERS + [1.0]))
    results = []

    for m in multipliers:
        L1 = L0 * m
        ead1 = balance_under_limit(base_ead, L0, L1)
        eps = scenario_eps(pd0, ead1)
        results.append({
            "limit":       L1,
            "multiplier":  m,
            "ep_worst":    min(eps),
            "ep_mean":     float(np.mean(eps)),
            "ep_best":     max(eps),
        })

    df_c = pd.DataFrame(results)
    rec_L = float(row.get("recommended_limit", L0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_c["limit"], y=df_c["ep_best"],
        fill=None, mode="lines",
        line=dict(color="rgba(34,197,94,0.3)", width=0),
        name="Best scenario"
    ))
    fig.add_trace(go.Scatter(
        x=df_c["limit"], y=df_c["ep_worst"],
        fill="tonexty", mode="lines",
        line=dict(color="rgba(239,68,68,0.3)", width=0),
        fillcolor="rgba(148,163,184,0.15)",
        name="Scenario range"
    ))
    fig.add_trace(go.Scatter(
        x=df_c["limit"], y=df_c["ep_mean"],
        mode="lines+markers",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=7),
        name="Mean EP across scenarios"
    ))
    # Mark current and recommended
    fig.add_vline(x=L0,    line_dash="dot",  line_color="#94a3b8", annotation_text="Current")
    fig.add_vline(x=rec_L, line_dash="dash", line_color="#22c55e", annotation_text="Recommended")

    fig.update_layout(
        title="Expected Profit vs Credit Limit (scenario band)",
        xaxis_title="Credit Limit",
        yaxis_title="Expected Profit",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────
# Utilization / delinquency trend chart
# ─────────────────────────────────────────────
def behavior_trend_chart(row: pd.Series) -> go.Figure:
    months = list(range(1, 7))
    util_vals = [row.get(f"util_{i}", np.nan) for i in months]
    delinq_vals = [row.get(f"PAY_{i-1}", np.nan) if i > 1 else row.get("PAY_0", np.nan)
                   for i in months]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=months, y=util_vals, name="Utilization",
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=7)
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=months, y=delinq_vals, name="Delinquency status",
        marker_color="#ef4444", opacity=0.5
    ), secondary_y=True)

    fig.update_layout(
        title="Utilization & Delinquency Trend (month 1 = most recent)",
        xaxis_title="Month (1=recent)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Utilization ratio", secondary_y=False)
    fig.update_yaxes(title_text="Delinquency status", secondary_y=True)
    return fig


# ─────────────────────────────────────────────
# Scenario table for a single customer
# ─────────────────────────────────────────────
def scenario_table(row: pd.Series) -> pd.DataFrame:
    from src.economics import balance_under_limit

    L0 = float(row["current_limit"])
    rec_L = float(row.get("recommended_limit", L0))
    base_ead = float(row.get("ead_current", L0 * 0.3))
    pd0 = float(row.get("pd_current", 0.05))

    ead_rec = balance_under_limit(base_ead, L0, rec_L)

    records = []
    for apr_a in APR_ANNUAL_SCENARIOS:
        apr_m = apr_a / 12.0
        for lgd in LGD_SCENARIOS:
            er = apr_m * ead_rec
            el = pd0 * ead_rec * lgd
            records.append({
                "APR":    f"{int(apr_a*100)}%",
                "LGD":    f"{int(lgd*100)}%",
                "EAD":    f"{ead_rec:,.0f}",
                "ER":     f"{er:,.2f}",
                "EL":     f"{el:,.2f}",
                "EP":     f"{er - el:,.2f}",
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# PAGE 1 — Portfolio Overview
# ─────────────────────────────────────────────
def page_portfolio_overview(final_plan: pd.DataFrame, summary: dict, raw_rec: pd.DataFrame):
    st.header("📊 Portfolio Overview")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    n_total = len(final_plan)
    n_inc = int((final_plan["action"] == "increase").sum())
    n_dec = int((final_plan["action"] == "decrease").sum())
    n_hold = int((final_plan["action"] == "hold").sum())
    total_ep = float(final_plan["ep_uplift"].sum())
    baseline_ep = float(raw_rec["ep_current"].sum()) if "ep_current" in raw_rec.columns else 0.0

    col1.metric("Total Customers", f"{n_total:,}")
    col2.metric("Increase",        f"{n_inc:,}",  delta=f"{n_inc/n_total*100:.1f}%")
    col3.metric("Decrease",        f"{n_dec:,}",  delta=f"-{n_dec/n_total*100:.1f}%", delta_color="inverse")
    col4.metric("Hold",            f"{n_hold:,}")
    col5.metric("Total EP Uplift", fmt_currency(total_ep), delta=f"vs baseline {fmt_currency(baseline_ep)}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Action distribution pie
        action_counts = final_plan["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        color_map = {"increase": "#22c55e", "decrease": "#ef4444", "hold": "#94a3b8"}
        fig_pie = px.pie(
            action_counts, names="action", values="count",
            color="action", color_discrete_map=color_map,
            title="Recommended Action Distribution"
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=320,
                               paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # EP uplift distribution
        fig_hist = px.histogram(
            final_plan, x="ep_uplift", nbins=60,
            color="action", color_discrete_map=color_map,
            title="EP Uplift Distribution by Action",
            labels={"ep_uplift": "Expected Profit Uplift"}
        )
        fig_hist.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=320,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Risk (PD) distribution
        fig_pd = px.histogram(
            final_plan, x="pd_current", nbins=50,
            title="Current PD Distribution",
            labels={"pd_current": "Calibrated PD"},
            color_discrete_sequence=["#6366f1"]
        )
        fig_pd.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pd, use_container_width=True)

    with col_r2:
        # EL budget gauge
        el_used = summary.get("used_el", 0.0)
        el_budget = summary.get("el_budget", EL_BUDGET)
        pct_used = el_used / max(el_budget, 1.0)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct_used * 100,
            title={"text": "EL Budget Utilisation (%)"},
            delta={"reference": 80, "increasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 60],  "color": "#d1fae5"},
                    {"range": [60, 80], "color": "#fef9c3"},
                    {"range": [80, 100],"color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 3}, "value": 90},
            },
            number={"suffix": "%", "valueformat": ".1f"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                                 paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Portfolio Budget Summary")
    st.json(summary)


# ─────────────────────────────────────────────
# PAGE 2 — Action Queue
# ─────────────────────────────────────────────
def page_action_queue(final_plan: pd.DataFrame):
    st.header("📋 Action Queue")

    col1, col2, col3 = st.columns(3)
    action_filter = col1.selectbox("Filter by action", ["All", "increase", "decrease", "hold"])
    sort_by = col2.selectbox("Sort by", ["ep_uplift", "pd_current", "current_limit", "recommended_limit"])
    sort_asc = col3.checkbox("Ascending", value=False)

    df = final_plan.copy()
    if action_filter != "All":
        df = df[df["action"] == action_filter]
    df = df.sort_values(sort_by, ascending=sort_asc)

    display_cols = [c for c in [
        "customer_id", "current_limit", "recommended_limit", "action",
        "pd_current", "pd_recommended", "ep_uplift",
        "el_uplift_proxy", "ead_uplift"
    ] if c in df.columns]

    # Format for readability
    fmt_df = df[display_cols].head(500).copy()
    for c in ["current_limit", "recommended_limit", "ead_uplift"]:
        if c in fmt_df.columns:
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:,.0f}")
    for c in ["pd_current", "pd_recommended"]:
        if c in fmt_df.columns:
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:.4f}")
    for c in ["ep_uplift", "el_uplift_proxy"]:
        if c in fmt_df.columns:
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:,.2f}")

    st.dataframe(fmt_df, use_container_width=True, height=520)

    # Download
    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered queue as CSV", csv, "action_queue.csv", "text/csv")


# ─────────────────────────────────────────────
# PAGE 3 — Customer Drilldown
# ─────────────────────────────────────────────
def page_customer_drilldown(final_plan: pd.DataFrame, raw_rec: pd.DataFrame):
    st.header("🔍 Customer Drilldown")

    id_col = "customer_id" if "customer_id" in final_plan.columns else None

    if id_col:
        cid_options = final_plan[id_col].astype(str).tolist()
        selected_id = st.selectbox("Select Customer ID", cid_options[:500])
        row_final = final_plan[final_plan[id_col].astype(str) == selected_id].iloc[0]
    else:
        idx = st.number_input("Row index", min_value=0, max_value=len(final_plan)-1, value=0, step=1)
        row_final = final_plan.iloc[int(idx)]

    # Try to get raw features row for behaviour chart
    if id_col and id_col in raw_rec.columns:
        row_raw = raw_rec[raw_rec[id_col].astype(str) == str(row_final[id_col])]
        row_raw = row_raw.iloc[0] if len(row_raw) > 0 else row_final
    else:
        row_raw = row_final

    action = str(row_final.get("action", "hold"))
    color = action_color(action)

    # Header card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}18 0%, {color}05 100%);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    ">
        <b>Action:</b> <span style="color:{color}; font-size:1.1em; font-weight:700">
            {action.upper()}
        </span> &nbsp;|&nbsp;
        <b>Current Limit:</b> {fmt_currency(float(row_final.get('current_limit', 0)))} &nbsp;→&nbsp;
        <b>Recommended:</b> {fmt_currency(float(row_final.get('recommended_limit', 0)))} &nbsp;|&nbsp;
        <b>PD (current):</b> {float(row_final.get('pd_current', 0)):.4f} &nbsp;→&nbsp;
        <b>PD (rec):</b> {float(row_final.get('pd_recommended', 0)):.4f}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EP Uplift",     fmt_currency(float(row_final.get("ep_uplift", 0)), 2))
    col2.metric("EAD (current)", fmt_currency(float(row_final.get("ead_current", 0))))
    col3.metric("EAD (rec)",     fmt_currency(float(row_final.get("ead_recommended", 0))))
    col4.metric("EL proxy Δ",    fmt_currency(float(row_final.get("el_uplift_proxy", 0)), 2))

    st.divider()

    # Charts row
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(behavior_trend_chart(row_raw), use_container_width=True)
    with col_b:
        st.plotly_chart(profit_vs_limit_chart(row_final), use_container_width=True)

    st.divider()

    # Scenario table
    st.subheader("Scenario Analysis (Recommended Limit)")
    st.dataframe(scenario_table(row_final), use_container_width=True, hide_index=True)

    st.subheader("Risk Reason Codes")
    
    if "reason_codes" in row_final.index and pd.notna(row_final["reason_codes"]):
        for code in str(row_final["reason_codes"]).split(" | "):
            st.markdown(f"- {code}")

    st.subheader("Top SHAP Features")
    if "top_features" in row_final.index and pd.notna(row_final["top_features"]):
        st.code(row_final["top_features"])

    # Raw record
    with st.expander("Full record (raw fields)"):
        st.json(row_final.to_dict())


# ─────────────────────────────────────────────
# PAGE 4 — Policy Simulator (what-if)
# ─────────────────────────────────────────────
def page_policy_simulator(raw_rec: pd.DataFrame):
    st.header("⚙️ Policy Simulator")
    st.caption("Adjust parameters and recompute the portfolio plan in real time.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Budget Controls")
        el_budget  = st.slider("EL Budget (M)",  0.5, 50.0, float(EL_BUDGET/1e6),  0.5) * 1e6
        ead_budget = st.slider("EAD Budget (M)", 1.0, 100.0, float(EAD_BUDGET/1e6), 1.0) * 1e6

    with col2:
        st.subheader("Macro Stress Shocks")
        pd_shock  = st.slider("PD shock (%)",  0, 50, 0, 5) / 100.0
        ead_shock = st.slider("EAD shock (%)", 0, 30, 0, 5) / 100.0

    rec2 = raw_rec.copy()
    if pd_shock  > 0: rec2 = apply_pd_shock(rec2, pd_shock)
    if ead_shock > 0: rec2 = apply_ead_shock(rec2, ead_shock)

    final, summary = portfolio_select(rec2, el_budget=el_budget, ead_budget=ead_budget)

    st.divider()
    st.subheader("Simulated Portfolio Outcome")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Increases approved",  int((final["action"] == "increase").sum()))
    m2.metric("Decreases",           int((final["action"] == "decrease").sum()))
    m3.metric("Holds",               int((final["action"] == "hold").sum()))
    m4.metric("Total EP Uplift",     fmt_currency(float(final["ep_uplift"].sum())))
    m5.metric("EL Used / Budget",
              f"{summary['used_el']/1e6:.2f}M / {summary['el_budget']/1e6:.2f}M")

    col_left, col_right = st.columns(2)
    with col_left:
        # Action counts under shock
        ac = final["action"].value_counts().reset_index()
        ac.columns = ["action", "count"]
        color_map = {"increase": "#22c55e", "decrease": "#ef4444", "hold": "#94a3b8"}
        fig = px.bar(ac, x="action", y="count", color="action",
                     color_discrete_map=color_map,
                     title="Simulated Action Distribution")
        fig.update_layout(showlegend=False, height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # EP uplift scatter (PD vs uplift, coloured by action)
        scatter_df = final.sample(min(1500, len(final)), random_state=42)
        fig_s = px.scatter(
            scatter_df, x="pd_current", y="ep_uplift",
            color="action", color_discrete_map=color_map,
            opacity=0.5, title="EP Uplift vs Current PD",
            labels={"pd_current": "Calibrated PD", "ep_uplift": "EP Uplift"},
        )
        fig_s.update_layout(height=300,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s, use_container_width=True)

    # Stress test summary table
    st.subheader("Portfolio Stress Summary")
    stress_rows = []
    for s in [0.0, 0.10, 0.20, 0.30]:
        tmp = apply_pd_shock(raw_rec.copy(), s)
        if ead_shock > 0:
            tmp = apply_ead_shock(tmp, ead_shock)
        _, smry = portfolio_select(tmp, el_budget=el_budget, ead_budget=ead_budget)
        stress_rows.append({
            "PD Shock": f"+{int(s*100)}%",
            "Increases": smry["n_increase_applied"],
            "Total EP Uplift": fmt_currency(smry["total_ep_uplift"]),
            "EL Used":  fmt_currency(smry["used_el"]),
            "Budget %": f"{smry['used_el']/max(el_budget,1)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 5 — Model Diagnostics
# ─────────────────────────────────────────────
def page_model_diagnostics(final_plan: pd.DataFrame):
    st.header("🧪 Model Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        # PD before vs after recommendation
        st.subheader("PD: Current vs Recommended")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=final_plan["pd_current"],
                                    name="PD current", opacity=0.65,
                                    marker_color="#6366f1"))
        if "pd_recommended" in final_plan.columns:
            fig.add_trace(go.Histogram(x=final_plan["pd_recommended"],
                                        name="PD recommended", opacity=0.65,
                                        marker_color="#22c55e"))
        fig.update_layout(barmode="overlay", height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # EP uplift by PD bucket
        st.subheader("EP Uplift by PD Bucket")
        tmp = final_plan.copy()
        tmp["pd_bucket"] = pd.cut(tmp["pd_current"],
                                   bins=[0, 0.05, 0.10, 0.20, 0.40, 1.0],
                                   labels=["<5%", "5–10%", "10–20%", "20–40%", ">40%"])
        bucket_summary = tmp.groupby("pd_bucket", observed=True)["ep_uplift"].mean().reset_index()
        fig2 = px.bar(bucket_summary, x="pd_bucket", y="ep_uplift",
                      title="Mean EP Uplift by PD Bucket",
                      labels={"pd_bucket": "PD Bucket", "ep_uplift": "Mean EP Uplift"},
                      color="ep_uplift",
                      color_continuous_scale=["#ef4444", "#fef9c3", "#22c55e"])
        fig2.update_layout(height=320,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Limit delta distribution
    st.subheader("Recommended Limit Change Distribution")
    tmp2 = final_plan.copy()
    tmp2["limit_delta_pct"] = (tmp2["recommended_limit"] - tmp2["current_limit"]) / tmp2["current_limit"] * 100
    fig3 = px.histogram(
        tmp2, x="limit_delta_pct", nbins=40,
        color="action",
        color_discrete_map={"increase": "#22c55e", "decrease": "#ef4444", "hold": "#94a3b8"},
        title="Limit Change % Distribution",
        labels={"limit_delta_pct": "Limit Change (%)"},
    )
    fig3.update_layout(height=300,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    # Summary statistics table
    st.subheader("Portfolio Statistics by Action")
    stat_cols = [c for c in [
        "pd_current", "pd_recommended",
        "ead_current", "ead_recommended",
        "ep_uplift", "el_uplift_proxy"
    ] if c in final_plan.columns]
    st.dataframe(
        final_plan.groupby("action")[stat_cols].describe().round(4),
        use_container_width=True,
    )
    # Global SHAP importance chart
    GLOBAL_IMP_PATH = "reports/shap_global_importance.csv"
    if os.path.exists(GLOBAL_IMP_PATH):
        st.subheader("Global Feature Importance (mean |SHAP|)")
        imp_df = pd.read_csv(GLOBAL_IMP_PATH).head(20)
        fig_imp = px.bar(
            imp_df,
            x="mean_shap",
            y="feature",
            orientation="h",
            title="Top 20 Features by Mean |SHAP| Value",
            labels={"mean_shap": "Mean |SHAP|", "feature": "Feature"},
            color="mean_shap",
            color_continuous_scale=["#e0f2fe", "#6366f1", "#1e1b4b"],
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ─────────────────────────────────────────────
# Main app entrypoint
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Credit Line Manager",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS — clean look
    st.markdown("""
    <style>
        [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
        [data-testid="stMetricDelta"] { font-size: 0.8rem; }
        .stDataFrame { border-radius: 8px; }
        section[data-testid="stSidebar"] { background-color: #0f172a; }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 💳 Credit Line Manager")
        st.markdown("---")
        page = st.radio("Navigate", [
            "📊 Portfolio Overview",
            "📋 Action Queue",
            "🔍 Customer Drilldown",
            "⚙️ Policy Simulator",
            "🧪 Model Diagnostics",
        ])

        st.markdown("---")
        st.caption("Budget (default run)")
        sidebar_el  = st.number_input("EL Budget", value=float(EL_BUDGET),  step=1e6, format="%.0f")
        sidebar_ead = st.number_input("EAD Budget", value=float(EAD_BUDGET), step=1e6, format="%.0f")

    # ── Load data ──
    if not os.path.exists(REC_PATH):
        st.error(f"Recommendations file not found: `{REC_PATH}`\n\nRun `python run_all.py` first.")
        st.stop()

    raw_rec = load_rec(REC_PATH)
    final_plan, summary = portfolio_select(raw_rec, el_budget=sidebar_el, ead_budget=sidebar_ead)

    # ── Route to page ──
    if page == "📊 Portfolio Overview":
        page_portfolio_overview(final_plan, summary, raw_rec)

    elif page == "📋 Action Queue":
        page_action_queue(final_plan)

    elif page == "🔍 Customer Drilldown":
        page_customer_drilldown(final_plan, raw_rec)

    elif page == "⚙️ Policy Simulator":
        page_policy_simulator(raw_rec)

    elif page == "🧪 Model Diagnostics":
        page_model_diagnostics(final_plan)


if __name__ == "__main__":
    main()