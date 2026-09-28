# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Adaptive credit line manager built on the UCI "Default of Credit Card Clients" dataset (`data/raw/uci_credit.csv`). It predicts default probability (PD) and exposure (EAD), simulates alternative credit limits per customer, picks the limit that maximizes risk-adjusted expected profit, then applies portfolio-level budget constraints and stress tests.

## Commands

```bash
pip install -r requirements-dev.txt      # includes requirements.txt + pytest; venv/ is gitignored
python run_all.py                        # full pipeline (run from repo root)
python run_all.py --data path/to/file.csv
streamlit run src/dashboard_app.py       # dashboard; needs run_all.py output first
pytest                                   # tests/*.py, pure-function unit tests only (no model training)
pytest tests/test_portfolio_opt.py -q    # single file
```

There is no linter or build step. Run everything from the repo root — `run_all.py` imports `src.*` as a package and all paths (`data/processed`, `models`, `reports`) are relative to the cwd.

The `tests/` suite covers the pure-function modules (`economics.py`, `calibrate.py`, `counterfactual.py`, `portfolio_opt.py`, `decision_engine.py` with fake PD/EAD models) — it does not train real models or run the pipeline end-to-end. `pyproject.toml` sets `pythonpath = ["."]` so `from src... import ...` resolves without installing the package.

The Step 5 decision engine loops row-by-row over all ~30k customers × 6 limit candidates with per-row model calls, so a full run is slow. SHAP is capped at `sample_n=5000` in `step_explainability`.

## Pipeline architecture (`run_all.py`)

Steps run in this order (note explainability runs before portfolio, despite its "6b" label):

1. `data_prep` — load CSV, rename target (`default.payment.next.month` → `TARGET`), clip negatives.
2. `features.build_features` — utilization, payment-ratio, delinquency, bill-trend and interaction features; renames `ID` → `customer_id`. Writes `data/processed/features.parquet`.
3. `pd_model` (XGBClassifier) + `calibrate` (isotonic on the validation split). A calibrator is a `(kind, obj)` tuple — always apply it via `apply_calibrator`, never call `.predict` on it directly.
4. `ead_model` (XGBRegressor) — target is a balance proxy `0.7*BILL_AMT1 + 0.3*BILL_AMT2`.
5. `decision_engine.recommend_limits` — for each customer, for each `LIMIT_MULTIPLIERS` value: rebuild limit-dependent features (`counterfactual.apply_new_limit_features`), re-score PD, scale balance by a log-elasticity (`economics.balance_under_limit`), compute EP. Increases are blocked by PD guardrails (`PD_INCREASE_MAX`, `PD_DECREASE_MIN`). Writes `recommendations_raw.csv`.
6. `explainability.run_explainability` — SHAP top features + human-readable reason codes (`_REASON_MAP`, matched by feature-name substring); overwrites `recommendations_raw.csv` with the annotated version and writes `audit_log.csv` and `reports/shap_global_importance.csv`.
7. `portfolio_opt.portfolio_select` — greedy selection of increases by ROI (`ep_uplift / el_uplift_proxy`) under `EL_BUDGET` / `EAD_BUDGET`; unapproved increases revert to hold. Decreases/holds always pass. Writes `recommendations_final.csv`.
8. `stress_test` — multiplies PD/EAD columns in the recommendation table and re-runs `portfolio_select` (does **not** re-simulate the decision engine). Writes `stress_test_results.csv` and `reports/pipeline_metrics.txt`.

### Key cross-file invariants

- **Customer ID handling**: `customer_id` stays in `X` but every model call drops it first (the `id_col` / `trainable()` pattern repeated in `pd_model`, `ead_model`, `decision_engine`, `run_all`). Models are trained on the column set without it, so column order/names must match exactly.
- **Counterfactual features must mirror `build_features`**: any new feature derived from `LIMIT_BAL` (utilization and interactions) must also be recomputed in `counterfactual.apply_new_limit_features`, or counterfactual PD scores will be stale. Note `util_trend` is currently not recomputed there.
- **Economics** (`economics.py`): EP per scenario = `(APR/12)·EAD − PD·EAD·LGD` over the APR × LGD grid in `config.py`; `ROBUST_MODE="worst_case"` takes the min across scenarios.
- **All tunables live in `src/config.py`** (limit multipliers, APR/LGD scenarios, elasticity, budgets, guardrails, seed).

## Dashboard (`src/dashboard_app.py`)

Reads only `data/processed/recommendations_raw.csv` (and optionally the pickled models in `models/`), then re-runs `portfolio_select` live using sidebar EL/EAD budget inputs. So changes to columns in the decision engine / explainability output affect the dashboard. Pages: Portfolio Overview, Action Queue, Customer Drilldown, Policy Simulator, Model Diagnostics.

## Repo notes

- Generated artifacts (`data/processed/*`, `models/*.pkl`, `reports/*`, `src/__pycache__/`) are committed to git, so a pipeline run produces a large diff. `models/pd.pkl` and `models/calibrator.pkl` are legacy artifacts not written by the current pipeline.
- Git history is organized by "Milestones" (1: PD prediction, 2: counterfactual decision engine, 3: portfolio optimization, stress test, explainability, dashboard).
