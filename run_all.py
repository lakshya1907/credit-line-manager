"""
run_all.py
──────────
Single entrypoint for the full Credit Line Manager pipeline.

Usage:
    python run_all.py                         # uses default data path
    python run_all.py --data path/to/file.csv # custom path

Runs in order:
    1. Load + clean data
    2. Build feature store
    3. Train PD model (XGBoost) + calibrate
    4. Train EAD model (XGBoost regression)
    5. Run per-customer decision engine (counterfactual + robust EP)
    6. Save raw recommendations CSV + all model artefacts
    7. Print summary + portfolio snapshot
"""

import os
import sys
import time
import json
import shutil
import uuid
import datetime
import subprocess
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from src.explainability import run_explainability

warnings.filterwarnings("ignore")

# ── ensure src/ is importable ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    RANDOM_SEED, EL_BUDGET, EAD_BUDGET, EAD_ELASTICITY, ROBUST_MODE,
    LIMIT_MULTIPLIERS, LGD_SCENARIOS, APR_ANNUAL_SCENARIOS,
    PD_INCREASE_MAX, PD_DECREASE_MIN,
)
from src.data_prep      import load_uci, basic_clean
from src.features       import build_features
from src.pd_model       import train_pd_model
from src.calibrate      import calibrate_pd, apply_calibrator
from src.ead_model      import make_balance_target, train_ead_model
from src.decision_engine import recommend_limits
from src.portfolio_opt  import portfolio_select
from src.stress_test    import run_stress_scenario

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DEFAULT_RAW  = "data/raw/uci_credit.csv"
PROC_DIR     = "data/processed"
MODEL_DIR    = "models"
REPORT_DIR   = "reports"

REC_RAW_PATH     = os.path.join(PROC_DIR, "recommendations_raw.csv")
REC_FINAL_PATH   = os.path.join(PROC_DIR, "recommendations_final.csv")
STRESS_PATH      = os.path.join(PROC_DIR, "stress_test_results.csv")
FEAT_PATH        = os.path.join(PROC_DIR, "features.parquet")
METRICS_PATH     = os.path.join(REPORT_DIR, "pipeline_metrics.txt")
GLOBAL_IMP_PATH = os.path.join(REPORT_DIR, "shap_global_importance.csv")
AUDIT_LOG_PATH  = os.path.join(PROC_DIR, "audit_log.csv")

# Every run also gets archived here (see archive_run()) so history survives
# the next run overwriting the paths above -- the "latest run" flat files
# stay for backward compatibility (the dashboard reads them directly), but
# they're no longer the only record of what a run produced.
RUNS_DIR         = os.path.join(PROC_DIR, "runs")
RUNS_INDEX_PATH  = os.path.join(PROC_DIR, "runs_index.csv")

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def _sep(title: str = ""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print(f"\n{'─'*width}")

def _fmt(v, unit=""):
    if isinstance(v, float):
        return f"{v:,.4f}{unit}"
    return str(v)

def _ensure_dirs():
    for d in [PROC_DIR, MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

def _new_run_id(started_at: datetime.datetime) -> str:
    # Timestamp prefix keeps runs sortable/human-readable; a short random
    # suffix avoids a collision if two runs somehow start in the same
    # second (e.g. kicked off from two terminals at once).
    return f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None

def _config_snapshot() -> dict:
    """A run's config is part of what has to be reproducible/auditable about
    it (it directly determines the guardrails and budgets that produced the
    recommendations), not just its outputs -- so it goes in the manifest."""
    return {
        "random_seed": RANDOM_SEED,
        "limit_multipliers": LIMIT_MULTIPLIERS,
        "apr_annual_scenarios": APR_ANNUAL_SCENARIOS,
        "lgd_scenarios": LGD_SCENARIOS,
        "robust_mode": ROBUST_MODE,
        "ead_elasticity": EAD_ELASTICITY,
        "el_budget": EL_BUDGET,
        "ead_budget": EAD_BUDGET,
        "pd_increase_max": PD_INCREASE_MAX,
        "pd_decrease_min": PD_DECREASE_MIN,
    }

def archive_run(
    run_id, started_at, finished_at, raw_path,
    pd_metrics, brier_raw, brier_cal, ead_metrics,
    portfolio_summary, stress_df,
):
    """
    Copy this run's outputs into data/processed/runs/<run_id>/, write a
    run_manifest.json (config snapshot + every metric, so a run is
    reproducible/explainable on its own, not just diffable against the
    files that happen to exist right now), and append a summary row to
    data/processed/runs_index.csv. Both runs/ and runs_index.csv are
    gitignored (they live under data/processed/), same as every other
    generated artifact -- this is plain per-run files, not a database; see
    CLAUDE.md for why that's the deliberate scope here.
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    for src_path in [REC_RAW_PATH, REC_FINAL_PATH, STRESS_PATH, AUDIT_LOG_PATH,
                      METRICS_PATH, GLOBAL_IMP_PATH]:
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(run_dir, os.path.basename(src_path)))

    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_time_seconds": (finished_at - started_at).total_seconds(),
        "git_commit": _git_commit_hash(),
        "raw_data_path": raw_path,
        "config": _config_snapshot(),
        "pd_model_metrics": {
            "val_roc_auc": pd_metrics["val_roc_auc"],
            "val_pr_auc": pd_metrics["val_pr_auc"],
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
        },
        "ead_model_metrics": {"val_mae": ead_metrics["val_mae"]},
        "portfolio_summary": portfolio_summary,
        "stress_test": stress_df.to_dict(orient="records"),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    index_row = pd.DataFrame([{
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": manifest["wall_time_seconds"],
        "git_commit": manifest["git_commit"],
        "roc_auc": pd_metrics["val_roc_auc"],
        "pr_auc": pd_metrics["val_pr_auc"],
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "ead_mae": ead_metrics["val_mae"],
        "el_budget": portfolio_summary["el_budget"],
        "ead_budget": portfolio_summary["ead_budget"],
        "used_el": portfolio_summary["used_el"],
        "used_ead": portfolio_summary["used_ead"],
        "n_increase_applied": portfolio_summary["n_increase_applied"],
        "n_decrease": portfolio_summary["n_decrease"],
        "n_hold": portfolio_summary["n_hold"],
        "total_ep_uplift": portfolio_summary["total_ep_uplift"],
    }])
    write_header = not os.path.exists(RUNS_INDEX_PATH)
    index_row.to_csv(RUNS_INDEX_PATH, mode="a", index=False, header=write_header)

    print(f"  Archived run   : {run_dir}")
    print(f"  Appended to    : {RUNS_INDEX_PATH}")

# ─────────────────────────────────────────────
# Step 1: Load + clean
# ─────────────────────────────────────────────
def step_load(raw_path: str) -> pd.DataFrame:
    _sep("STEP 1 — Load & Clean")
    t0 = time.time()
    df = load_uci(raw_path)
    df = basic_clean(df)
    print(f"  Shape          : {df.shape}")
    print(f"  Target balance : {df['TARGET'].mean():.4f}  ({df['TARGET'].sum()} defaults)")
    print(f"  LIMIT_BAL range: {df['LIMIT_BAL'].min():,.0f} – {df['LIMIT_BAL'].max():,.0f}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return df

# ─────────────────────────────────────────────
# Step 2: Feature engineering
# ─────────────────────────────────────────────
def step_features(df: pd.DataFrame):
    _sep("STEP 2 — Feature Engineering")
    t0 = time.time()
    X, y = build_features(df)
    print(f"  Features shape : {X.shape}")
    print(f"  Feature groups : util_*, pay_ratio_*, delinq_*, bill_*, interactions")
    # persist for reference
    X.to_parquet(FEAT_PATH, index=False)
    print(f"  Saved to       : {FEAT_PATH}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return X, y

# ─────────────────────────────────────────────
# Step 3: PD model + calibration
# ─────────────────────────────────────────────
def step_pd_model(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame):
    _sep("STEP 3 — PD Model + Calibration")
    t0 = time.time()

    pd_model, pd_metrics, splits = train_pd_model(X, y)
    X_train, X_val, y_train, y_val = splits

    # Calibrate on validation
    val_raw_scores = pd_model.predict_proba(X_val)[:, 1]
    pd_calibrator  = calibrate_pd(val_raw_scores, y_val, method="isotonic")

    # Calibrated PD for entire dataset
    id_col = "customer_id" if "customer_id" in X.columns else None
    X_trainable = X.drop(columns=[id_col]) if id_col else X
    all_raw = pd_model.predict_proba(X_trainable)[:, 1]
    pd_cal  = apply_calibrator(pd_calibrator, all_raw)

    # Calibration quality — Brier score (manual, no sklearn needed)
    from sklearn.metrics import brier_score_loss
    brier_raw = brier_score_loss(y_val, val_raw_scores)
    brier_cal = brier_score_loss(y_val, apply_calibrator(pd_calibrator, val_raw_scores))

    print(f"  ROC-AUC (val)  : {pd_metrics['val_roc_auc']:.4f}")
    print(f"  PR-AUC  (val)  : {pd_metrics['val_pr_auc']:.4f}")
    print(f"  Brier raw / cal: {brier_raw:.4f} / {brier_cal:.4f}")
    print(f"  PD range (cal) : {pd_cal.min():.4f} – {pd_cal.max():.4f}  mean={pd_cal.mean():.4f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    joblib.dump(pd_model,     os.path.join(MODEL_DIR, "pd_xgb.pkl"))
    joblib.dump(pd_calibrator, os.path.join(MODEL_DIR, "pd_calibrator.pkl"))
    print(f"  Saved: models/pd_xgb.pkl, models/pd_calibrator.pkl")

    return pd_model, pd_calibrator, pd_cal, pd_metrics, brier_raw, brier_cal

# ─────────────────────────────────────────────
# Step 4: EAD model
# ─────────────────────────────────────────────
def step_ead_model(X: pd.DataFrame, df_raw: pd.DataFrame):
    _sep("STEP 4 — EAD Model (Balance Forecast)")
    t0 = time.time()

    y_balance = make_balance_target(df_raw)
    print(f"  Balance target : mean={y_balance.mean():,.0f}  std={y_balance.std():,.0f}")

    ead_model, ead_metrics = train_ead_model(X, y_balance)

    print(f"  MAE (val)      : {ead_metrics['val_mae']:,.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    joblib.dump(ead_model, os.path.join(MODEL_DIR, "ead_xgb.pkl"))
    print(f"  Saved: models/ead_xgb.pkl")

    return ead_model, ead_metrics

# ─────────────────────────────────────────────
# Step 5: Decision engine (per-customer)
# ─────────────────────────────────────────────
def step_decisions(X: pd.DataFrame, pd_model, pd_calibrator, ead_model):
    _sep("STEP 5 — Per-Customer Decision Engine")
    t0 = time.time()
    print(f"  Simulating {len(X):,} customers across {len(LIMIT_MULTIPLIERS)} candidate limits…")

    rec = recommend_limits(X, pd_model, pd_calibrator, ead_model)

    n_inc  = int((rec["action"] == "increase").sum())
    n_dec  = int((rec["action"] == "decrease").sum())
    n_hold = int((rec["action"] == "hold").sum())
    total_up = float(rec["ep_uplift"].sum())

    print(f"  Actions        : increase={n_inc}  decrease={n_dec}  hold={n_hold}")
    print(f"  Total EP uplift: {total_up:,.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    rec.to_csv(REC_RAW_PATH, index=False)
    print(f"  Saved: {REC_RAW_PATH}")
    return rec

# ─────────────────────────────────────────────
# Step 6: Portfolio optimisation
# ─────────────────────────────────────────────
def step_portfolio(rec: pd.DataFrame):
    _sep("STEP 6 — Portfolio Constrained Optimisation")
    t0 = time.time()

    final, summary = portfolio_select(rec, el_budget=EL_BUDGET, ead_budget=EAD_BUDGET)

    print(f"  EL budget used : {summary['used_el']:,.2f} / {summary['el_budget']:,.2f}"
          f"  ({summary['used_el']/max(summary['el_budget'],1)*100:.1f}%)")
    print(f"  EAD budget used: {summary['used_ead']:,.2f} / {summary['ead_budget']:,.2f}"
          f"  ({summary['used_ead']/max(summary['ead_budget'],1)*100:.1f}%)")
    print(f"  Approved inc   : {summary['n_increase_applied']}")
    print(f"  Decreases      : {summary['n_decrease']}")
    print(f"  Holds          : {summary['n_hold']}")
    print(f"  Total EP uplift: {summary['total_ep_uplift']:,.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    final.to_csv(REC_FINAL_PATH, index=False)
    print(f"  Saved: {REC_FINAL_PATH}")
    return final, summary

# ─────────────────────────────────────────────
# Step 7: Stress testing
# ─────────────────────────────────────────────
def step_stress_test(X, pd_model, pd_calibrator, ead_model):
    _sep("STEP 7 — Stress Testing")
    t0 = time.time()

    rows = []
    shocks = [0.0, 0.10, 0.20, 0.30]
    for s in shocks:
        # Genuine re-simulation: the decision engine re-optimizes each
        # customer's recommended limit under the shocked PD, so the action
        # itself (not just its reported numbers) can change under stress.
        # See src/stress_test.py for why this differs from the dashboard's
        # apply_pd_shock/apply_ead_shock fast-approximation path.
        shocked_rec = run_stress_scenario(X, pd_model, pd_calibrator, ead_model, pd_shock=s)
        _, smry = portfolio_select(shocked_rec, el_budget=EL_BUDGET, ead_budget=EAD_BUDGET)
        rows.append({
            "pd_shock":       f"+{int(s*100)}%",
            "n_increase":     smry["n_increase_applied"],
            "n_decrease":     smry["n_decrease"],
            "n_hold":         smry["n_hold"],
            "total_ep_uplift": smry["total_ep_uplift"],
            "el_used":        smry["used_el"],
            "el_budget_pct":  smry["used_el"] / max(smry["el_budget"], 1) * 100,
        })
        print(f"  PD +{int(s*100):2d}% shock → "
              f"inc={smry['n_increase_applied']:4d}  "
              f"EP uplift={smry['total_ep_uplift']:,.2f}  "
              f"EL used={smry['used_el']:,.2f}")

    st_df = pd.DataFrame(rows)
    st_df.to_csv(STRESS_PATH, index=False)
    print(f"  Saved: {STRESS_PATH}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return st_df

# ─────────────────────────────────────────────
# Step 8: Write metrics report
# ─────────────────────────────────────────────
def step_write_report(
    pd_metrics, brier_raw, brier_cal,
    ead_metrics, portfolio_summary, stress_df
):
    _sep("STEP 8 — Report")

    lines = [
        "=" * 62,
        "CREDIT LINE MANAGER — PIPELINE METRICS REPORT",
        "=" * 62,
        "",
        "[PD Model — XGBoost]",
        f"  ROC-AUC (val)       : {pd_metrics['val_roc_auc']:.4f}",
        f"  PR-AUC  (val)       : {pd_metrics['val_pr_auc']:.4f}",
        f"  Brier (uncalibrated): {brier_raw:.4f}",
        f"  Brier (calibrated)  : {brier_cal:.4f}",
        "",
        "[EAD Model — XGBoost Regressor]",
        f"  MAE (val balance proxy): {ead_metrics['val_mae']:,.2f}",
        "",
        "[Portfolio Optimisation (no stress)]",
        f"  EL used / budget    : {portfolio_summary['used_el']:,.2f} / {portfolio_summary['el_budget']:,.2f}",
        f"  EAD used / budget   : {portfolio_summary['used_ead']:,.2f} / {portfolio_summary['ead_budget']:,.2f}",
        f"  Increases approved  : {portfolio_summary['n_increase_applied']}",
        f"  Decreases           : {portfolio_summary['n_decrease']}",
        f"  Total EP uplift     : {portfolio_summary['total_ep_uplift']:,.2f}",
        "",
        "[Stress Test — PD Shock]",
    ]
    for _, row in stress_df.iterrows():
        lines.append(
            f"  {row['pd_shock']:6s} → "
            f"inc={int(row['n_increase']):4d}  "
            f"EP={row['total_ep_uplift']:,.2f}  "
            f"EL used={row['el_used']:,.2f}  "
            f"({row['el_budget_pct']:.1f}% budget)"
        )
    lines += ["", "=" * 62]
    report = "\n".join(lines)

    with open(METRICS_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"\n  Saved: {METRICS_PATH}")

def step_explainability(X, rec_raw, pd_model):
    from src.explainability import run_explainability
    _sep("STEP 6b — Explainability (SHAP + Reason Codes)")
    t0 = time.time()
 
    id_col = "customer_id" if "customer_id" in X.columns else None
    X_trainable = X.drop(columns=[id_col]) if id_col else X
 
    annotated, global_imp, shap_values = run_explainability(
        pd_model       = pd_model,
        X_trainable    = X_trainable,
        rec_df         = rec_raw,
        topk           = 5,
        sample_n       = 5000,          # cap at 5k rows for speed; remove for full SHAP
        save_global_path = os.path.join(REPORT_DIR, "shap_global_importance.csv"),
        save_audit_path  = os.path.join(PROC_DIR,   "audit_log.csv"),
        policy_constraints = {"el_budget": EL_BUDGET, "ead_budget": EAD_BUDGET},
    )
 
    # Overwrite raw recs with annotated version (adds top_features + reason_codes)
    annotated.to_csv(REC_RAW_PATH, index=False)
    print(f"  Updated recommendations with reason codes: {REC_RAW_PATH}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return annotated, global_imp

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(raw_path: str):
    t_start = time.time()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    run_id = _new_run_id(started_at)
    _ensure_dirs()

    _sep("CREDIT LINE MANAGER — FULL PIPELINE")
    print(f"  Run ID: {run_id}")

    df                                                      = step_load(raw_path)
    X, y                                                    = step_features(df)
    pd_model, pd_calibrator, pd_cal, pd_metrics, brier_raw, brier_cal \
                                                            = step_pd_model(X, y, df)
    ead_model, ead_metrics                                  = step_ead_model(X, df)
    rec_raw                                                 = step_decisions(X, pd_model, pd_calibrator, ead_model)
    rec_raw, global_imp                                     = step_explainability(X, rec_raw, pd_model)
    final_plan, portfolio_summary                           = step_portfolio(rec_raw)
    stress_df                                               = step_stress_test(X, pd_model, pd_calibrator, ead_model)
    step_write_report(pd_metrics, brier_raw, brier_cal,
                      ead_metrics, portfolio_summary, stress_df)

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    _sep("STEP 9 — Archive Run")
    archive_run(run_id, started_at, finished_at, raw_path,
                pd_metrics, brier_raw, brier_cal, ead_metrics,
                portfolio_summary, stress_df)

    _sep("DONE")
    print(f"  Run ID         : {run_id}")
    print(f"  Total wall time: {time.time()-t_start:.1f}s")
    print()
    print("  ► Launch dashboard:")
    print("    streamlit run src/dashboard_app.py")
    print()

    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Credit Line Manager pipeline")
    parser.add_argument("--data", default=DEFAULT_RAW,
                        help=f"Path to UCI credit CSV (default: {DEFAULT_RAW})")
    args = parser.parse_args()
    main(args.data)